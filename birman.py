#!/usr/bin/env python3
"""Start a llama.cpp binary with flags chosen for this machine.

    ./birman.py --start [-m MODEL]                                                  # same, in the background (pid logs/birman.pid)
    ./birman.py --status                                                            # running? pid, model, parameters, health
    ./birman.py --stop                                                              # stop what --start launched
    ./birman.py -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf              # llama-server
    ./birman.py --tool cli -m MODEL -p "Hello" -n 64                           # any other llama-* tool
    ./birman.py --check -m MODEL                                               # report only, run nothing
    ./birman.py                                                                # no -m: pick the best model for this machine

It inspects the machine and adds what the measurements in the README support:

  -t N              the number of physical cores this process may use (all sockets), capped by a cgroup CPU
                    quota. Generation peaked there on both machines tested (8 on an i9-9900; 32 on a 2-socket
                    Xeon with --numa distribute); SMT threads only helped prompt processing, and not reliably.
  --numa distribute on Linux machines with more than one NUMA node (see numa_distribute.py): about 1.1-2x faster
                    generation on a 2-socket Xeon. --numa-evict drops the model from the page cache first, so
                    the pinned threads place its pages (worth doing once after copying or downloading a model).
  -ngl 0            CPU only.
  --chat-template-file   the 2B-4T chat template, for the 2B-4T model when running the server or cli.

Without -m it also chooses the model (model_picker.py): the canonical GGUFs under models/ that fit in the available
memory (a cgroup limit counts), tried best first - chat-capable models, then more parameters; --prefer largest ignores
the chat preference - and the first whose measured generation speed (a few-second llama-bench probe run with the
flags below) reaches --min-tps (default 10 t/s) is used, with the reasoning printed. --no-probe skips the
measurement, --reprobe ignores the cache (kept 7 days; the speed depends on the load at the time) and
--models-dir looks elsewhere. --check still runs the probe when it has to choose.

It also warns about things that make I2_S slow or wrong: no AVX2 on x86 (the scalar fallback runs; ARM has NEON
kernels), and too little free RAM for the model. On macOS -t is the performance-core count (sysctl
hw.perflevel0.physicalcpu) and memory comes from sysctl and vm_stat. Anything it does not recognise is
passed to the llama binary unchanged, and a flag you give yourself (-t is ours; --numa, -ngl and
--chat-template-file are yours) is never overridden.

Opt-outs: --no-numa / BITNET_NUMA=0; --numa-evict / BITNET_NUMA_EVICT=1; -t to choose the thread count.
"""
import argparse
import glob
import json
import math
import os
import platform
import re
import shlex
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import model_picker as picker  # noqa: E402
import numa_distribute as numa  # noqa: E402

TOOLS = {
    "server": "llama-server",
    "cli": "llama-cli",
    "completion": "llama-completion",
    "bench": "llama-bench",
    "perplexity": "llama-perplexity",
}
CHAT_TEMPLATE = os.path.join(ROOT, "chat-templates", "bitnet-b1.58-2B-4T.jinja")
SYS_CPU = "/sys/devices/system/cpu"


def _read(path):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return None


def allowed_cpus():
    try:
        return sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return list(range(os.cpu_count() or 1))


def topology(cpus, sys_cpu=SYS_CPU):
    """(sockets, physical cores) among `cpus`; a CPU whose topology cannot be read counts as its own core."""
    sockets, cores = set(), set()
    for c in cpus:
        pkg = _read("%s/cpu%d/topology/physical_package_id" % (sys_cpu, c))
        core = _read("%s/cpu%d/topology/core_id" % (sys_cpu, c))
        if pkg is None or core is None:
            sockets.add(("?",))
            cores.add(("?", c))
        else:
            sockets.add(pkg.strip())
            cores.add((pkg.strip(), core.strip()))
    return len(sockets), len(cores)


def cgroup_cpu_limit(root="/sys/fs/cgroup"):
    """CPUs allowed by a cgroup quota, or None."""
    v2 = _read(os.path.join(root, "cpu.max"))
    if v2:
        quota, _, period = v2.strip().partition(" ")
        if quota != "max" and period.isdigit() and int(period) > 0:
            return int(quota) / int(period)
    q = _read(os.path.join(root, "cpu", "cpu.cfs_quota_us"))
    p = _read(os.path.join(root, "cpu", "cpu.cfs_period_us"))
    if q and p and int(q) > 0 and int(p) > 0:
        return int(q) / int(p)
    return None


def cgroup_mem_available(root="/sys/fs/cgroup"):
    """Bytes left under a cgroup memory limit (which /proc/meminfo does not reflect), or None."""
    limit, cur = _read(os.path.join(root, "memory.max")), _read(os.path.join(root, "memory.current"))
    if limit is None:
        limit = _read(os.path.join(root, "memory", "memory.limit_in_bytes"))
        cur = _read(os.path.join(root, "memory", "memory.usage_in_bytes"))
    if limit and cur and limit.strip().isdigit() and cur.strip().isdigit() and int(limit) < (1 << 60):
        return max(0, int(limit) - int(cur))
    return None


def meminfo_kb():
    out = {}
    for line in (_read("/proc/meminfo") or "").splitlines():
        m = re.match(r"(\w+):\s+(\d+)", line)
        if m:
            out[m.group(1)] = int(m.group(2))
    return out


def cpu_model_and_simd():
    info = _read("/proc/cpuinfo") or ""
    m = re.search(r"^(?:model name|Model name|Hardware)\s*:\s*(.+)$", info, re.MULTILINE)
    model = m.group(1).strip() if m else platform.processor() or platform.machine()
    flags = set()
    for key in ("flags", "Features"):
        m = re.search(r"^%s\s*:\s*(.+)$" % key, info, re.MULTILINE)
        if m:
            flags |= set(m.group(1).split())
    return model, flags


def sysctl(name):
    try:
        out = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return out or None


def darwin_mem_kb():
    """macOS: {"MemTotal", "MemAvailable"} in kB from sysctl and vm_stat (free + inactive + speculative pages)."""
    out = {}
    total = sysctl("hw.memsize")
    if total and total.isdigit():
        out["MemTotal"] = int(total) // 1024
    try:
        vm = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        vm = ""
    m = re.search(r"page size of (\d+) bytes", vm)
    pages = [re.search(r"^Pages %s:\s+(\d+)" % k, vm, re.MULTILINE) for k in ("free", "inactive", "speculative")]
    if m and all(pages):
        out["MemAvailable"] = sum(int(x.group(1)) for x in pages) * int(m.group(1)) // 1024
    return out


def detect():
    cpus = allowed_cpus()
    limit = cgroup_cpu_limit()
    if platform.system() == "Darwin":
        # Apple Silicon: threads = performance cores. The efficiency cores slow the barrier-synchronized threads
        # down (2B-4T generation on an M1 Max: about 64 t/s at 8 threads, 14 t/s at 10).
        sockets = 1
        phys, perf = sysctl("hw.physicalcpu"), sysctl("hw.perflevel0.physicalcpu")
        cores = int(phys) if phys and phys.isdigit() else len(cpus)
        threads = int(perf) if perf and perf.isdigit() else cores
        mem = darwin_mem_kb()
    else:
        sockets, cores = topology(cpus)
        threads = cores
        mem = meminfo_kb()
    if limit is not None:
        threads = max(1, min(threads, int(math.floor(limit))))
    avail_kb = mem.get("MemAvailable", 0)
    cg_mem = cgroup_mem_available()
    if cg_mem is not None:
        avail_kb = min(avail_kb, cg_mem // 1024)
    model, flags = cpu_model_and_simd()
    if platform.system() == "Darwin":
        model = sysctl("machdep.cpu.brand_string") or model
    return {
        "arch": platform.machine(), "cpu": model, "flags": flags,
        "logical": len(cpus), "cores": cores, "sockets": sockets, "numa_nodes": numa.numa_node_count(),
        "cgroup_limit": limit, "threads": threads,
        "mem_total_gb": mem.get("MemTotal", 0) / 1048576.0, "mem_avail_gb": avail_kb / 1048576.0,
        "cgroup_mem": cg_mem is not None,
    }


def warnings_for(m, model_bytes):
    out = []
    arch = m["arch"].lower()
    if arch in ("x86_64", "amd64") and "avx2" not in m["flags"]:
        out.append("this CPU has no AVX2: I2_S runs the scalar fallback, which is correct but slow")
    if model_bytes and m["mem_avail_gb"] and model_bytes / 1073741824.0 * 1.1 > m["mem_avail_gb"]:
        out.append("the model is %.1f GiB but only %.1f GiB of RAM is available: expect swapping or a failed load"
                   % (model_bytes / 1073741824.0, m["mem_avail_gb"]))
    return out


def has_flag(args, *names):
    return any(a == n or a.startswith(n + "=") for a in args for n in names)


def build_command(binary, model, passthrough, threads, numa_enabled, tool):
    """Return (command, notes). Flags the user gave are never overridden."""
    cmd = [binary]
    notes = []
    if model:
        cmd += ["-m", model]
    cmd += ["-t", str(threads)]
    notes.append("threads = %d" % threads)
    if not has_flag(passthrough, "-ngl", "--gpu-layers", "--n-gpu-layers"):
        cmd += ["-ngl", "0"]
    if (tool in ("server", "cli") and model and "2B-4T" in model
            and not has_flag(passthrough, "--chat-template-file", "--chat-template") and os.path.exists(CHAT_TEMPLATE)):
        cmd += ["--chat-template-file", CHAT_TEMPLATE]
        notes.append("2B-4T chat template")
    cmd += passthrough
    extra, reason = numa.explain(cmd, numa_enabled)
    cmd += extra
    notes.append("NUMA: " + ("adding `%s` (%s)" % (" ".join(extra), reason) if extra else "none (%s)" % reason))
    return cmd, notes


def parse(argv):
    p = argparse.ArgumentParser(description="Start a llama.cpp binary with flags chosen for this machine.",
                                epilog="Unrecognised arguments are passed to the llama binary unchanged.",
                                allow_abbrev=False)
    p.add_argument("--tool", choices=sorted(TOOLS), default="server", help="which binary to run (default: server)")
    p.add_argument("-m", "--model", help="path to the GGUF model (default: choose the best one for this machine from models/)")
    p.add_argument("-t", "--threads", type=int, help="thread count (default: physical cores this process may use)")
    p.add_argument("--start", action="store_true", help="run in the background: log to logs/birman.log, pid in logs/birman.pid (refuses if one is running)")
    p.add_argument("--status", action="store_true", help="show whether the --start instance is running: pid, uptime, model, command line, health (exit 0 if running, 1 if not)")
    p.add_argument("--stop", action="store_true", help="stop the instance --start launched (SIGTERM, SIGKILL after 15 s) and exit")
    p.add_argument("--check", "--dry-run", dest="check", action="store_true", help="report the machine and the command, run nothing")
    p.add_argument("--no-numa", action="store_true", help="do not add --numa distribute on multi-socket machines (also: BITNET_NUMA=0)")
    p.add_argument("--numa-evict", action="store_true", help="evict the model from the page cache first (multi-socket only; also: BITNET_NUMA_EVICT=1)")
    p.add_argument("--models-dir", default=os.path.join(ROOT, "models"), help="where to look for models when -m is not given (default: models/)")
    p.add_argument("--prefer", choices=("chat", "largest"), default="chat", help="model choice when -m is not given: chat-capable models first, then more parameters (default), or just the most parameters")
    p.add_argument("--min-tps", type=float, default=picker.DEFAULT_MIN_TPS, help="minimum measured generation speed for an automatically chosen model (default: %(default)s t/s)")
    p.add_argument("--no-probe", action="store_true", help="choose the model by memory fit only, without measuring its speed")
    p.add_argument("--reprobe", action="store_true", help="ignore cached speed measurements")
    p.add_argument("--bin-dir", default=os.path.join(ROOT, "build", "bin"), help="directory with the llama binaries")
    return p.parse_known_args(argv)


def choose_model(args, m, threads):
    """Pick a model from models/ for this machine (see model_picker.py). Returns (candidate or None, evicted paths)."""
    cands = []
    for path in picker.find_models(args.models_dir):
        try:
            cands.append(picker.describe(path))
        except (OSError, ValueError) as e:
            print("  skipping %s: %s" % (path, e), file=sys.stderr)
    if not cands:
        print("  no models found under %s (run ./build.sh, or pass -m MODEL.gguf)" % args.models_dir, file=sys.stderr)
        return None, set()
    mem = int(m["mem_avail_gb"] * 1073741824)
    if mem <= 0:  # not readable on this OS (no /proc/meminfo): do not conclude that nothing fits
        print("  available memory could not be read on this system: memory fit is not checked", file=sys.stderr)
        mem = 1 << 62
    bench = os.path.join(args.bin_dir, TOOLS["bench"])
    numa_extra, _reason = numa.explain([], not args.no_numa)
    evicted = set()
    probe = None
    if args.no_probe:
        print("  speed is not measured (--no-probe): choosing by memory fit only", file=sys.stderr)
    elif not os.path.isfile(bench):
        print("  %s not found, so speed cannot be measured: choosing by memory fit only" % bench, file=sys.stderr)
    else:
        key = "%s|%d|%d|%d" % (m["cpu"], m["cores"], m["sockets"], m["numa_nodes"])
        evict = numa.evict_requested(args.numa_evict) and bool(numa_extra)

        def probe(c):
            def before():
                if evict:
                    numa.evict_from_page_cache(c["path"])
                    evicted.add(c["path"])
            return picker.probe_tps(bench, c["path"], threads, numa_extra, key, use_cache=not args.reprobe, before=before)

    chosen, rows = picker.select(cands, mem, args.prefer, args.min_tps, probe)
    mem_text = "%.1f GiB available" % (mem / 1073741824.0) if mem < (1 << 61) else "memory unknown"
    print("Model selection (%s; at least %.0f t/s measured; needs 1.2x the file + 0.5 GiB, %s):"
          % ("chat-capable models first, then more parameters" if args.prefer == "chat" else "most parameters first",
             args.min_tps, mem_text), file=sys.stderr)
    for c, status in rows:
        print("  %-30s %5.2f B  %5.2f GiB  %-5s %s%s" % (c["label"][:30], c["params"] / 1e9, c["size"] / 1073741824.0, "chat" if c["chat"] else "",
                                                      status, "   <-- chosen" if chosen and c["path"] == chosen["path"] else ""), file=sys.stderr)
    return chosen, evicted


PID_FILE = os.path.join(ROOT, "logs", "birman.pid")
LOG_FILE = os.path.join(ROOT, "logs", "birman.log")
STATE_FILE = os.path.join(ROOT, "logs", "birman.json")


def running_pid():
    """The pid in PID_FILE if that process is alive, else None (a stale file is removed)."""
    try:
        pid = int(open(PID_FILE).read().split()[0])
        os.kill(pid, 0)
        return pid
    except (OSError, ValueError, IndexError):
        for f in (PID_FILE, STATE_FILE):
            if os.path.exists(f):
                os.remove(f)
        return None


def stop(timeout=15.0):
    pid = running_pid()
    if pid is None:
        print("no instance started by --start is running (%s)" % PID_FILE, file=sys.stderr)
        return 1
    os.kill(pid, signal.SIGTERM)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            break
        time.sleep(0.2)
    else:
        os.kill(pid, signal.SIGKILL)
        print("pid %d ignored SIGTERM for %.0f s, sent SIGKILL" % (pid, timeout), file=sys.stderr)
    for f in (PID_FILE, STATE_FILE):
        if os.path.exists(f):
            os.remove(f)
    print("stopped pid %d" % pid, file=sys.stderr)
    return 0


def flag_value(cmd, name, default):
    for i, a in enumerate(cmd):
        if a == name and i + 1 < len(cmd):
            return cmd[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return default


def status():
    pid = running_pid()
    if pid is None:
        print("not running (no instance started by --start; see %s for the last log)" % LOG_FILE)
        return 1
    try:
        st = json.load(open(STATE_FILE))
    except (OSError, ValueError):
        st = {}
    cmd = st.get("command") or []
    print("running, pid %d" % pid)
    if st.get("started"):
        up = int(time.time() - st["started"])
        print("  started: %s (up %dd %02d:%02d:%02d)" % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st["started"])),
                                                         up // 86400, up % 86400 // 3600, up % 3600 // 60, up % 60))
    model = flag_value(cmd, "-m", None) or flag_value(cmd, "--model", None)
    if model:
        size = os.path.getsize(model) if os.path.isfile(model) else 0
        print("  model:   %s%s" % (model, " (%.0f MB)" % (size / 1e6) if size else ""))
    for label, names, default in (("threads", ("-t", "--threads"), None), ("numa", ("--numa",), "off"),
                                  ("gpu layers", ("-ngl", "--n-gpu-layers"), None), ("context", ("-c", "--ctx-size"), "model default"),
                                  ("chat template", ("--chat-template-file",), "model's own")):
        v = next((x for x in (flag_value(cmd, n, None) for n in names) if x is not None), default)
        if v is not None:
            print("  %-15s %s" % (label + ":", v))
    host, port = flag_value(cmd, "--host", "127.0.0.1"), flag_value(cmd, "--port", "8080")
    try:
        with urllib.request.urlopen("http://%s:%s/health" % (host, port), timeout=2) as r:
            health = json.load(r).get("status", "?")
    except Exception as e:
        health = "unreachable (%s)" % getattr(e, "reason", e)
    print("  listen:  %s:%s, health: %s" % (host, port, health))
    print("  command: " + shlex.join(cmd) if cmd else "  command: unknown (state file missing)")
    print("  log:     " + LOG_FILE)
    return 0


def start_background(cmd):
    pid = running_pid()
    if pid is not None:
        print("error: already running as pid %d (./birman.py --stop first)" % pid, file=sys.stderr)
        return 2
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "ab") as log:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
    with open(PID_FILE, "w") as f:
        f.write("%d\n" % proc.pid)
    with open(STATE_FILE, "w") as f:
        json.dump({"pid": proc.pid, "started": time.time(), "command": cmd}, f)
    time.sleep(2)
    if proc.poll() is not None:
        os.remove(PID_FILE)
        os.remove(STATE_FILE)
        print("error: exited with status %d straight away, see %s" % (proc.returncode, LOG_FILE), file=sys.stderr)
        return 2
    print("started pid %d, log %s (stop with ./birman.py --stop)" % (proc.pid, LOG_FILE), file=sys.stderr)
    return 0


def main(argv=None):
    args, passthrough = parse(sys.argv[1:] if argv is None else argv)
    if args.status:
        return status()
    if args.stop:
        return stop()
    m = detect()
    threads = args.threads if args.threads else m["threads"]

    print("Machine: %s (%s), %d socket(s), %d physical cores (%d logical CPUs usable), %d NUMA node(s), RAM %.0f GiB (%.0f GiB free%s)"
          % (m["cpu"], m["arch"], m["sockets"], m["cores"], m["logical"], m["numa_nodes"], m["mem_total_gb"], m["mem_avail_gb"],
             ", limited by a cgroup" if m["cgroup_mem"] else ""), file=sys.stderr)
    if m["cgroup_limit"] is not None:
        print("  cgroup CPU limit: %.1f CPUs" % m["cgroup_limit"], file=sys.stderr)

    evicted = set()
    if not args.model:
        chosen, evicted = choose_model(args, m, threads)
        if chosen:
            args.model = chosen["path"]

    model_bytes = os.path.getsize(args.model) if args.model and os.path.isfile(args.model) else 0
    for w in warnings_for(m, model_bytes):
        print("  WARNING: " + w, file=sys.stderr)

    binary = os.path.join(args.bin_dir, TOOLS[args.tool])
    problems = []
    if not os.path.isfile(binary):
        problems.append("%s not found: run ./build.sh first" % binary)
    if args.tool != "bench" or args.model:
        if not args.model:
            problems.append("no model given or found (-m MODEL.gguf)")
        elif not os.path.isfile(args.model):
            problems.append("model not found: %s" % args.model)

    cmd, notes = build_command(binary, args.model, passthrough, threads, not args.no_numa, args.tool)
    print("  " + "; ".join(notes), file=sys.stderr)
    print("  command: " + shlex.join(cmd), file=sys.stderr)

    if problems and not args.check:
        for pr in problems:
            print("error: " + pr, file=sys.stderr)
        return 2
    if args.check:
        for pr in problems:
            print("  PROBLEM: " + pr, file=sys.stderr)
        return 0 if not problems else 2

    if numa.evict_requested(args.numa_evict) and "--numa" in cmd and args.model and args.model not in evicted:
        print("  %s %s from the page cache" % ("evicted" if numa.evict_from_page_cache(args.model) else "could not evict", args.model),
              file=sys.stderr)
    if args.start:
        return start_background(cmd)
    os.execv(binary, cmd)


if __name__ == "__main__":
    sys.exit(main())
