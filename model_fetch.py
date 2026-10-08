"""Find, judge and install a model that is not under models/ yet (used by birman.py for `-m NAME`).

The catalogue is the model list of setup_env.py (Hugging Face repos it can convert to I2_S) plus the two BitNet
embedding models (a ready GGUF, downloaded as it is). Installing runs the same steps as by hand:

  conversion models   python setup_env.py -hr REPO -q i2_s     (download, convert to f32, quantize to I2_S, rebuild)
  embedding models    huggingface-cli download REPO --local-dir models/DIR

Before anything is downloaded the model is judged for this machine from estimates (parameter count -> sizes); a model
that does not look suitable is only installed after the user confirms. The estimates come from what was measured here:
final I2_S file about 0.38 GiB per billion parameters (2.61 GiB for 7.46 B), download about 2 GB per billion (bf16
safetensors), conversion peak memory about 4.5 GiB per billion (36 GB for Llama3-8B), the f32 intermediate 4 GB per
billion, and generation speed scaling inversely with file size (it is memory-bandwidth-bound).
"""
import os
import re
import shutil
import subprocess
import sys

import model_picker as picker

ROOT = os.path.dirname(os.path.abspath(__file__))
GIB = 1073741824.0

# Parameter counts (billions) where the name does not say; everything else is read from the name ("7B", "1B", ...).
KNOWN_PARAMS = {"bitnet_b1_58-large": 0.73, "bitnet-b1.58-2b-4t": 2.41, "bitnet-embedding-0.6b": 0.6, "bitnet-embedding-270m": 0.27}
# Ready-made GGUF repos (no conversion): the embedding models.
EMBEDDING = {"microsoft/BitNet-embedding-0.6B": "bitnet-embedding-0.6b", "microsoft/BitNet-embedding-270M": "bitnet-embedding-270m"}


# Nominal size in the name -> parameters it really has (measured here: Falcon3-1B 1.67, Falcon3-7B 7.46, Llama3-8B 8.03).
ACTUAL = {1.0: 1.7, 3.0: 3.2, 7.0: 7.46, 8.0: 8.03, 10.0: 10.3}


def params_from_name(name):
    if name.lower() in KNOWN_PARAMS:
        return KNOWN_PARAMS[name.lower()]
    m = re.search(r"(\d+(?:\.\d+)?)b(?![a-z])", name.lower().replace("-", " "))
    if not m:
        return None
    nominal = float(m.group(1))
    return ACTUAL.get(nominal, nominal)


def catalogue(include_embedding):
    """List of dicts: name (directory under models/), repo, params (billions or None), kind, chat."""
    out = []
    if include_embedding:
        for repo, name in EMBEDDING.items():
            out.append({"name": name, "repo": repo, "params": params_from_name(name), "kind": "embedding", "chat": False})
        return out
    try:
        import setup_env
        repos = {r: v["model_name"] for r, v in setup_env.SUPPORTED_HF_MODELS.items() if "embedding" not in r.lower()}
    except Exception:  # the catalogue is a convenience: without it nothing can be fetched
        return []
    for repo, name in repos.items():
        low = name.lower()
        out.append({"name": name, "repo": repo, "params": params_from_name(name), "kind": "convert",
                    "chat": any(h in low for h in picker.CHAT_HINTS)})
    return out


def find(pattern, entries, prefer):
    """Entries whose name (or repo) contains every word of pattern, best first: chat-capable then more parameters
    (prefer "chat"), or just more parameters (prefer "largest")."""
    hit = [e for e in entries if picker.matches(os.path.join(e["name"], e["repo"]), pattern)]
    key = (lambda e: (e["chat"], e["params"] or 0)) if prefer == "chat" else (lambda e: (e["params"] or 0,))
    return sorted(hit, key=key, reverse=True)


def sizes(e):
    """Estimated sizes in GiB: final file, download, peak memory while converting, disk needed in total."""
    p = e["params"]
    if p is None:
        return None
    if e["kind"] == "embedding":
        final = p * 0.8 + 0.1
        return {"final": final, "download": final, "peak": final, "disk": final * 1.1}
    final = p * 0.38 + 0.3
    dl = p * 1.9
    return {"final": final, "download": dl, "peak": p * 4.5, "disk": (dl + p * 4.0 + final) * 1.1}


def swap_free_gib():
    try:
        with open("/proc/meminfo") as f:
            for ln in f:
                if ln.startswith("SwapFree:"):
                    return int(ln.split()[1]) / (1024.0 * 1024.0)
    except OSError:
        pass
    return 0.0


def judge(e, machine, models_dir, min_tps, est_bytes_per_s):
    """(problems, notes): problems are reasons the model does not look suitable for this machine."""
    sz = sizes(e)
    if sz is None:
        return ["the size of %s is unknown (no parameter count in its name), so suitability cannot be judged" % e["name"]], []
    problems, notes = [], []
    mem = machine["mem_avail_gb"]
    if mem > 0:
        need = sz["final"] * picker.MEM_FACTOR + picker.MEM_OVERHEAD / GIB
        if need > mem:
            problems.append("running it needs about %.1f GiB of RAM and %.1f GiB is available" % (need, mem))
        if e["kind"] == "convert" and sz["peak"] > mem + swap_free_gib():
            problems.append("converting it peaks at about %.0f GiB of memory; %.1f GiB RAM + %.1f GiB free swap is available"
                            % (sz["peak"], mem, swap_free_gib()))
        elif e["kind"] == "convert" and sz["peak"] > mem:
            notes.append("converting peaks at about %.0f GiB, more than the %.1f GiB of free RAM: it will use swap and be slow"
                         % (sz["peak"], mem))
    probe_dir = models_dir if os.path.isdir(models_dir) else (os.path.dirname(os.path.abspath(models_dir)) or ".")
    try:
        free = shutil.disk_usage(probe_dir).free / GIB
        if sz["disk"] > free:
            problems.append("it needs about %.0f GiB of disk while converting and %.0f GiB is free in %s" % (sz["disk"], free, probe_dir))
    except OSError:
        pass
    if e["kind"] == "convert" and est_bytes_per_s:
        tps = est_bytes_per_s / (sz["final"] * GIB)
        notes.append("estimated generation speed here: about %.0f t/s (from the speed measured on a local model; measured after install)" % tps)
        if tps < min_tps:
            problems.append("its estimated generation speed (about %.1f t/s) is below the %.0f t/s minimum" % (tps, min_tps))
    return problems, notes


def confirm(question, assume_yes):
    if assume_yes:
        print("%s yes (--yes)" % question, file=sys.stderr)
        return True
    if not sys.stdin.isatty():
        print("%s no (not a terminal; pass --yes to proceed)" % question, file=sys.stderr)
        return False
    try:
        return input("%s [y/N] " % question).strip().lower() in ("y", "yes")
    except EOFError:
        return False


def _env():
    env = dict(os.environ)
    venv = os.path.join(ROOT, ".venv", "bin")
    if os.path.isdir(venv):
        env["PATH"] = venv + os.pathsep + env.get("PATH", "")
    gguf_py = os.path.join(ROOT, "3rdparty", "llama.cpp", "gguf-py")
    env["PYTHONPATH"] = gguf_py + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def install(e, models_dir):
    """Download and build the model into models_dir/<name>/. Returns True on success."""
    py = os.path.join(ROOT, ".venv", "bin", "python")
    py = py if os.path.isfile(py) else sys.executable
    if e["kind"] == "embedding":
        cmd = ["huggingface-cli", "download", e["repo"], "--local-dir", os.path.join(models_dir, e["name"])]
    else:
        cmd = [py, os.path.join(ROOT, "setup_env.py"), "-hr", e["repo"], "-q", "i2_s", "-md", models_dir]
    print("  running: " + " ".join(cmd), file=sys.stderr)
    try:
        rc = subprocess.run(cmd, cwd=ROOT, env=_env()).returncode
    except OSError as ex:
        print("  could not run %s: %s" % (cmd[0], ex), file=sys.stderr)
        return False
    if rc != 0:
        print("  installing %s failed (exit %d); see logs/ for the step that failed" % (e["name"], rc), file=sys.stderr)
        return False
    if e["kind"] == "convert":  # the f32 GGUF is only an intermediate (4 GB per billion parameters): drop it
        d = os.path.join(models_dir, e["name"])
        f32, i2s = os.path.join(d, "ggml-model-f32.gguf"), os.path.join(d, "ggml-model-i2_s.gguf")
        if os.path.isfile(f32) and os.path.isfile(i2s) and os.path.getsize(i2s) > 0:
            size = os.path.getsize(f32)
            os.remove(f32)
            print("  removed the f32 intermediate (%.1f GiB)" % (size / GIB), file=sys.stderr)
    return True
