"""Choose the best model for this machine from the GGUF files under models/.

Used by start_llama.py when no -m is given. The policy:

  1. Candidates are the canonical quantized outputs of setup_env.py, models/*/ggml-model-{i2_s,tl1,tl2}.gguf and
     the -f16emb variant (f32 files, embedding models and stale *.bad/*.old/... variants are ignored; where both
     exist the -f16emb file of a directory wins). Parameter counts are read from the GGUF header.
  2. A model fits when 1.2 x its file size + 0.5 GiB is within the memory that is actually available (MemAvailable,
     lowered by a cgroup memory limit).
  3. Candidates are tried best first. The ranking is "chat-capable first, then more parameters" (--prefer chat, the
     default) or just "more parameters" (--prefer largest). Each is measured with a short llama-bench generation
     probe run with the flags the server will use (threads, --numa); the first one that reaches the minimum
     generation speed (default 10 t/s) is chosen. Generation is memory-bandwidth-bound, so a model at least as large
     as one that was too slow is not probed. If none is fast enough the smallest model that fits is used, with a warning.
  4. Probe results (kept for 7 days; speed depends on the load at the time, --reprobe to redo) and parsed headers
     are cached in ~/.cache/start_llama/, keyed by machine settings and file size/mtime, so later starts are instant.

"Chat-capable" is a name heuristic (names containing "instruct", and the official BitNet-b1.58-2B-4T release);
parameter count is only a proxy for quality. Pass -m to choose a model yourself.
"""
import json
import os
import re
import struct
import subprocess
import time

CANONICAL = re.compile(r"^ggml-model-(i2_s|tl1|tl2)(-f16emb)?\.gguf$")
CHAT_HINTS = ("instruct", "2b-4t")
MEM_FACTOR = 1.2
MEM_OVERHEAD = 512 * 1024 * 1024
DEFAULT_MIN_TPS = 10.0
PROBE_TTL = 7 * 24 * 3600  # seconds: a cached speed measurement is ignored after this
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "start_llama")

_FIXED = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4), 5: ("<i", 4),
          6: ("<f", 4), 7: ("<?", 1), 10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8)}
_KEEP = {"general.architecture", "general.name", "general.size_label"}


# --- GGUF header ---------------------------------------------------------------------------------------------

def _u32(f):
    return struct.unpack("<I", f.read(4))[0]


def _u64(f):
    return struct.unpack("<Q", f.read(8))[0]


_FILE_SIZE = 0  # size of the file being parsed: lengths and counts beyond it are corrupt


def _check(n):
    if n > _FILE_SIZE:
        raise ValueError("corrupt GGUF (a length or count exceeds the file size)")
    return n


def _string(f, keep=True):
    n = _check(_u64(f))
    if keep:
        return f.read(n).decode("utf-8", "replace")
    f.seek(n, 1)
    return None


def _value(f, t, keep):
    if t in _FIXED:
        fmt, size = _FIXED[t]
        return struct.unpack(fmt, f.read(size))[0]
    if t == 8:
        return _string(f, keep)
    if t == 9:
        et, n = _u32(f), _check(_u64(f))
        if et in _FIXED:
            f.seek(_FIXED[et][1] * n, 1)
        else:
            for _ in range(n):
                _value(f, et, False)
        return None
    raise ValueError("unknown GGUF value type %d" % t)


def read_gguf_header(path):
    """(metadata subset, parameter count) read from the GGUF header; no tensor data is touched.
    Raises ValueError for anything that is not a well-formed GGUF file."""
    global _FILE_SIZE
    _FILE_SIZE = os.path.getsize(path)
    try:
        return _read_gguf_header(path)
    except (struct.error, OverflowError, MemoryError, UnicodeError) as e:
        raise ValueError("truncated or corrupt GGUF (%s)" % e)


def _read_gguf_header(path):
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        version = _u32(f)
        if version < 2:
            raise ValueError("GGUF version %d is not supported" % version)
        n_tensors, n_kv = _check(_u64(f)), _check(_u64(f))
        meta = {}
        for _ in range(n_kv):
            key = _string(f)
            t = _u32(f)
            v = _value(f, t, key in _KEEP)
            if key in _KEEP:
                meta[key] = v
        params = 0
        for _ in range(n_tensors):
            _string(f, False)
            nd = _u32(f)
            if nd > 8:
                raise ValueError("corrupt GGUF (tensor with %d dimensions)" % nd)
            n = 1
            for _ in range(nd):
                n *= _u64(f)
            f.seek(4 + 8, 1)  # type, offset
            params += n
    return meta, params


# --- cache ---------------------------------------------------------------------------------------------------

def _load_cache():
    try:
        with open(os.path.join(CACHE_DIR, "cache.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_cache(cache):
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(os.path.join(CACHE_DIR, "cache.json"), "w") as f:
            json.dump(cache, f)
    except OSError:
        pass


def _file_key(path):
    st = os.stat(path)
    return "%s|%d|%d" % (os.path.abspath(path), st.st_size, int(st.st_mtime))


# --- candidates ----------------------------------------------------------------------------------------------

def find_models(models_dir):
    """Canonical model files under models_dir; the -f16emb variant wins within a directory."""
    by_dir = {}
    for root, _dirs, files in os.walk(models_dir):
        for name in files:
            if CANONICAL.match(name):
                by_dir.setdefault(root, []).append(name)
    out = []
    for root, names in sorted(by_dir.items()):
        pick = next((n for n in names if "-f16emb" in n), sorted(names)[0])
        out.append(os.path.join(root, pick))
    return out


def describe(path, cache=None):
    """Dict with name, params, size and chat for a model file (header parsed once, then cached)."""
    cache = _load_cache() if cache is None else cache
    key = "hdr|" + _file_key(path)
    info = cache.get(key)
    if info is None:
        meta, params = read_gguf_header(path)
        info = {"params": params, "arch": meta.get("general.architecture", ""), "gname": meta.get("general.name", "")}
        cache[key] = info
        _save_cache(cache)
    label = os.path.basename(os.path.dirname(path)) or os.path.basename(path)
    hay = (label + " " + info.get("gname", "")).lower()
    return {"path": path, "label": label, "params": info["params"], "size": os.path.getsize(path),
            "chat": any(h in hay for h in CHAT_HINTS)}


def fits(size, mem_avail_bytes):
    return size * MEM_FACTOR + MEM_OVERHEAD <= mem_avail_bytes


def rank_key(c, prefer):
    return (c["chat"], c["params"]) if prefer == "chat" else (c["params"],)


# --- probe ---------------------------------------------------------------------------------------------------

def probe_tps(bench, model, threads, extra_args, machine_key, use_cache=True, timeout=180, before=None):
    """Generation speed (t/s) measured with llama-bench; (value, from_cache). None if the probe failed."""
    cache = _load_cache()
    key = "tps|%s|%d|%s|%s" % (machine_key, threads, " ".join(extra_args), _file_key(model))
    hit = cache.get(key)
    if use_cache and isinstance(hit, list) and len(hit) == 2 and time.time() - hit[1] < PROBE_TTL:
        return hit[0], True
    if before:
        before()  # only when a real probe will run (not on a cache hit)
    # tg4 first warms the page cache under the same threads/NUMA flags, tg16 is the reading
    cmd = [bench, "-m", model, "-p", "0", "-n", "4,16", "-r", "1", "-t", str(threads), "-ngl", "0"] + list(extra_args)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None, False
    tps = None
    for line in r.stdout.splitlines():
        cells = [c.strip() for c in line.split("|")]
        if "tg16" in cells:
            try:
                tps = float(cells[-2].split("±")[0])
            except (ValueError, IndexError):
                pass
    if tps is None:
        return None, False
    cache[key] = [tps, time.time()]
    _save_cache(cache)
    return tps, False


# --- selection -----------------------------------------------------------------------------------------------

def select(cands, mem_avail_bytes, prefer, min_tps, probe):
    """Pick a model. probe(candidate) -> (tps or None, from_cache) or None to skip probing.

    Returns (chosen candidate or None, rows) where each row is (candidate, status text)."""
    order = sorted(cands, key=lambda c: rank_key(c, prefer), reverse=True)
    rows = {c["path"]: "not tried (a higher-ranked model was chosen)" for c in order}
    too_slow_size = None
    chosen = None
    for c in order:
        if not fits(c["size"], mem_avail_bytes):
            rows[c["path"]] = "does not fit: needs %.1f GiB, %.1f GiB available" % (
                (c["size"] * MEM_FACTOR + MEM_OVERHEAD) / 1073741824.0, mem_avail_bytes / 1073741824.0)
            continue
        if too_slow_size is not None and c["size"] >= too_slow_size:
            rows[c["path"]] = "skipped: at least as large as a model that was too slow"
            continue
        if probe is None:
            chosen = c
            rows[c["path"]] = "fits (speed not measured)"
            break
        tps, cached = probe(c)
        if tps is None:
            rows[c["path"]] = "probe failed"
            too_slow_size = c["size"] if too_slow_size is None else min(too_slow_size, c["size"])
            continue
        note = "%.1f t/s%s" % (tps, " (cached)" if cached else "")
        if tps >= min_tps:
            rows[c["path"]] = note + ", meets %.0f t/s" % min_tps
            chosen = c
            break
        rows[c["path"]] = note + ", below %.0f t/s" % min_tps
        too_slow_size = c["size"] if too_slow_size is None else min(too_slow_size, c["size"])
    if chosen is None:
        fitting = [c for c in order if fits(c["size"], mem_avail_bytes)]
        if fitting:
            chosen = min(fitting, key=lambda c: c["size"])
            rows[chosen["path"]] += "; fallback: smallest model that fits"
    return chosen, [(c, rows[c["path"]]) for c in order]
