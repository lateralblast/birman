#!/usr/bin/env python3
"""Start a llama.cpp binary with flags chosen for this machine.

    ./start_llama.py -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf              # llama-server
    ./start_llama.py --tool cli -m MODEL -p "Hello" -n 64                           # any other llama-* tool
    ./start_llama.py --check -m MODEL                                               # report only, run nothing
    ./start_llama.py                                                                # no -m: pick the best model for this machine

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
import secrets
import shutil
import signal
import shlex
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
try:
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "VERSION")) as _f:
        VERSION = _f.read().strip()
except OSError:
    VERSION = "unknown"
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


DEFAULT_KEY_FILE = os.path.join(os.path.expanduser("~"), ".cache", "start_llama", "api_key")


def key_file_arg(args):
    """Path given with --api-key-file (either spelling), or None."""
    for i, a in enumerate(args):
        if a == "--api-key-file" and i + 1 < len(args):
            return args[i + 1]
        if a.startswith("--api-key-file="):
            return a.split("=", 1)[1]
    return None


def ensure_api_key(passthrough, dry_run):
    """For --open without a key: use the key file (given, or ~/.cache/start_llama/api_key), creating it with a fresh
    random key when it is missing or empty. Returns (passthrough with --api-key-file, key or None, path or None)."""
    if has_flag(passthrough, "--api-key") or os.environ.get("LLAMA_API_KEY"):
        return passthrough, None, None
    path = key_file_arg(passthrough) or DEFAULT_KEY_FILE
    key = None
    try:
        with open(path) as f:
            key = next((ln.strip() for ln in f if ln.strip() and not ln.startswith("#")), None)
    except OSError:
        pass
    if key is None and not dry_run:
        key = secrets.token_urlsafe(32)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(key + "\n")
        print("  generated a new API key in %s" % path, file=sys.stderr)
    if not has_flag(passthrough, "--api-key-file"):
        passthrough = passthrough + ["--api-key-file", path]
    return passthrough, key, path


def server_port(args):
    """Port the server will listen on: --port/-p from the arguments, else llama-server's default 8080."""
    for i, a in enumerate(args):
        if a in ("--port", "-p") and i + 1 < len(args) and args[i + 1].isdigit():
            return int(args[i + 1])
        if a.startswith("--port=") and a[7:].isdigit():
            return int(a[7:])
    return 8080


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None


FIREWALL_STATE = os.path.join(os.path.expanduser("~"), ".cache", "start_llama", "firewall.json")


def firewall_plan(port):
    """(name, already-open check, open command, close command) for the enabled host firewall, or None (Linux: ufw,
    firewalld). Plain iptables/nftables are not touched: their rule sets are too varied to extend safely."""
    if sys.platform != "linux":
        return None
    sudo = [] if os.geteuid() == 0 else ["sudo", "-n"]
    if shutil.which("ufw"):
        enabled = False
        try:
            with open("/etc/ufw/ufw.conf") as f:
                enabled = any(ln.strip().lower() == "enabled=yes" for ln in f)
        except OSError:
            r = _run(sudo + ["ufw", "status"])
            enabled = bool(r and "Status: active" in r.stdout)
        if enabled:
            return ("ufw", None, sudo + ["ufw", "allow", "%d/tcp" % port, "comment", "start_llama"],
                    sudo + ["ufw", "delete", "allow", "%d/tcp" % port])
    if shutil.which("firewall-cmd"):
        r = _run(["firewall-cmd", "--state"])
        if r and r.stdout.strip() == "running":
            return ("firewalld", ["firewall-cmd", "--query-port=%d/tcp" % port], sudo + ["firewall-cmd", "--add-port=%d/tcp" % port],
                    sudo + ["firewall-cmd", "--remove-port=%d/tcp" % port])
    return None


def _save_state(entries):
    try:
        os.makedirs(os.path.dirname(FIREWALL_STATE), exist_ok=True)
        if entries:
            with open(FIREWALL_STATE, "w") as f:
                json.dump(entries, f)
        elif os.path.exists(FIREWALL_STATE):
            os.remove(FIREWALL_STATE)
    except OSError:
        pass


def _load_state():
    try:
        with open(FIREWALL_STATE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return []


def open_firewall(port, dry_run):
    """Open the port if a firewall is enabled and does not already allow it. Returns True when a rule was added
    (recorded in FIREWALL_STATE so close_firewall can remove exactly that rule)."""
    plan = firewall_plan(port)
    if not plan:
        print("  firewall: none detected (ufw/firewalld), nothing to open", file=sys.stderr)
        return False
    name, query, cmd, undo = plan
    if dry_run:
        print("  firewall: %s is enabled; would run `%s`" % (name, shlex.join(cmd)), file=sys.stderr)
        return False
    if query:
        q = _run(query)
        if q and q.stdout.strip() == "yes":
            print("  firewall: %s already allows tcp/%d; left as it is" % (name, port), file=sys.stderr)
            return False
    r = _run(cmd)
    if not (r and r.returncode == 0):
        print("  WARNING: %s is enabled but `%s` failed (%s); run it as root or the port stays blocked"
              % (name, shlex.join(cmd), (r.stderr.strip() if r else "could not run")[:120] or "no detail"), file=sys.stderr)
        return False
    if "existing rule" in (r.stdout + r.stderr).lower():
        print("  firewall: %s already allows tcp/%d; left as it is" % (name, port), file=sys.stderr)
        return False
    _save_state(_load_state() + [{"name": name, "port": port, "undo": undo}])
    print("  firewall: %s now allows tcp/%d%s; it is removed again when the server exits (or: start_llama.py --close)"
          % (name, port, " (runtime only)" if name == "firewalld" else ""), file=sys.stderr)
    return True


def close_firewall(dry_run=False):
    """Remove every rule open_firewall added (restores the firewall to how it was). Returns 0, or 1 if one failed."""
    entries = _load_state()
    if not entries:
        print("  firewall: no rule added by start_llama.py is recorded; nothing to close", file=sys.stderr)
        return 0
    left, rc = [], 0
    for e in entries:
        if dry_run:
            print("  firewall: would run `%s`" % shlex.join(e["undo"]), file=sys.stderr)
            left.append(e)
            continue
        r = _run(e["undo"])
        if r and r.returncode == 0:
            print("  firewall: %s no longer allows tcp/%d" % (e["name"], e["port"]), file=sys.stderr)
        else:
            rc = 1
            left.append(e)
            print("  WARNING: `%s` failed (%s); the rule is still in place, retry with --close"
                  % (shlex.join(e["undo"]), (r.stderr.strip() if r else "could not run")[:120] or "no detail"), file=sys.stderr)
    if not dry_run:
        _save_state(left)
    return rc


def run_and_close(binary, cmd):
    """Run the server as a child so the firewall rule can be removed when it exits (Ctrl-C and SIGTERM included).
    A SIGKILL of this script cannot be caught: use --close then."""
    proc = subprocess.Popen(cmd)

    def forward(signum, _frame):
        try:
            proc.send_signal(signum)
        except OSError:
            pass
    old = {sig: signal.signal(sig, forward) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        return proc.wait()
    finally:
        for sig, h in old.items():
            signal.signal(sig, h)
        close_firewall()


def build_command(binary, model, passthrough, threads, numa_enabled, tool, host=None):
    """Return (command, notes). Flags the user gave are never overridden. host: "127.0.0.1" or "0.0.0.0" (server only)."""
    cmd = [binary]
    notes = []
    if host:
        cmd += ["--host", host]
        notes.append("listening on %s%s" % (host, " (all interfaces, no authentication unless --api-key is given)"
                                            if host == "0.0.0.0" and not has_flag(passthrough, "--api-key", "--api-key-file") else ""))
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
    p.add_argument("--version", action="version", version="start_llama.py " + VERSION)
    p.add_argument("--tool", choices=sorted(TOOLS), default="server", help="which binary to run (default: server)")
    p.add_argument("-m", "--model", help="path to the GGUF model (default: choose the best one for this machine from models/)")
    p.add_argument("-t", "--threads", type=int, help="thread count (default: physical cores this process may use)")
    p.add_argument("--check", "--dry-run", dest="check", action="store_true", help="report the machine and the command, run nothing")
    p.add_argument("--no-numa", action="store_true", help="do not add --numa distribute on multi-socket machines (also: BITNET_NUMA=0)")
    p.add_argument("--numa-evict", action="store_true", help="evict the model from the page cache first (multi-socket only; also: BITNET_NUMA_EVICT=1)")
    p.add_argument("--models-dir", default=os.path.join(ROOT, "models"), help="where to look for models when -m is not given (default: models/)")
    p.add_argument("--prefer", choices=("chat", "largest"), default="chat", help="model choice when -m is not given: chat-capable models first, then more parameters (default), or just the most parameters")
    p.add_argument("--min-tps", type=float, default=picker.DEFAULT_MIN_TPS, help="minimum measured generation speed for an automatically chosen model (default: %(default)s t/s)")
    p.add_argument("--no-probe", action="store_true", help="choose the model by memory fit only, without measuring its speed")
    p.add_argument("--reprobe", action="store_true", help="ignore cached speed measurements")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--local", action="store_true", help="server: listen on localhost only (127.0.0.1; llama-server's own default)")
    g.add_argument("--open", action="store_true", help="server: listen on all interfaces (0.0.0.0) so other machines can reach the API; combine with --api-key KEY")
    p.add_argument("--close", action="store_true", help="remove the firewall rules a previous --open run added, then exit (they are also removed when the server exits)")
    p.add_argument("--no-firewall", action="store_true", help="with --open: do not add a firewall rule for the server port (ufw/firewalld)")
    p.add_argument("--bin-dir", default=os.path.join(ROOT, "build", "bin"), help="directory with the llama binaries")
    args, rest = p.parse_known_args(argv)
    if (args.local or args.open) and args.tool != "server":
        p.error("--local/--open only apply to --tool server")
    if (args.local or args.open) and has_flag(rest, "--host"):
        p.error("--local/--open conflict with --host")
    return args, rest


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


def main(argv=None):
    args, passthrough = parse(sys.argv[1:] if argv is None else argv)
    if args.close:
        return close_firewall(args.check)
    fw_added = False
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

    if args.open:
        passthrough, key, key_path = ensure_api_key(passthrough, args.check)
        if key_path:
            print("  API key (%s): %s" % (key_path, key or "<would be generated on start>"), file=sys.stderr)
            print("  clients send it as:  Authorization: Bearer <key>", file=sys.stderr)
        if not args.no_firewall:
            fw_added = open_firewall(server_port(passthrough), args.check)

    binary = os.path.join(args.bin_dir, TOOLS[args.tool])
    problems = []
    if not os.path.isfile(binary):
        problems.append("%s not found: run ./build.sh first" % binary)
    if args.tool != "bench" or args.model:
        if not args.model:
            problems.append("no model given or found (-m MODEL.gguf)")
        elif not os.path.isfile(args.model):
            problems.append("model not found: %s" % args.model)

    cmd, notes = build_command(binary, args.model, passthrough, threads, not args.no_numa, args.tool,
                              "0.0.0.0" if args.open else "127.0.0.1" if args.local else None)
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
    if fw_added:
        return run_and_close(binary, cmd)
    os.execv(binary, cmd)


if __name__ == "__main__":
    sys.exit(main())
