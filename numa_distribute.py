"""Run llama.cpp with `--numa distribute` on multi-socket (multi-NUMA-node) machines.

By default the model's pages end up on one NUMA node and the threads float between nodes, so generation is
limited by one node's memory bandwidth. `--numa distribute` pins the threads evenly across the nodes; each
thread then first-touches (and so places, in local memory) the rows of the weights it always handles, and
every read is local. On a 2-socket Xeon (see the README) it made generation 1.1-2.0x faster at every thread
count for four models (about 2x at 32-64 threads) and usually sped up prompt processing too. The scripts that
launch llama.cpp binaries add the flag automatically when it applies:

  - Linux, with more than one NUMA node that has CPUs (nodes that only have memory do not count);
  - the command does not already contain --numa.

Where the pages are first touched matters: right after a model file is copied or downloaded its pages sit on
one node, and a run then gains less (about 15% instead of 2x). Evict the file from the page cache once and let
the next run populate it: `--numa-evict` (or BITNET_NUMA_EVICT=1) does that before launching, at the cost of
re-reading the file from disk.

Disable with BITNET_NUMA=0 (or off/false/no), or --no-numa on the scripts.

Command line: `python numa_distribute.py --args` prints the llama.cpp arguments to add (nothing when it does
not apply), which is what the shell scripts use; `--explain` says why or why not.
"""
import glob
import os
import platform
import re
import sys

ENV_VAR = "BITNET_NUMA"
EVICT_ENV_VAR = "BITNET_NUMA_EVICT"
ARGS = ["--numa", "distribute"]
_OFF = {"0", "off", "false", "no"}
_ON = {"1", "on", "true", "yes"}


def numa_node_count(node_root="/sys/devices/system/node"):
    """Number of NUMA nodes that have CPUs (nodes with memory only, e.g. CXL or HBM, are not counted)."""
    count = 0
    for node in glob.glob(os.path.join(node_root, "node[0-9]*")):
        if not re.fullmatch(r"node\d+", os.path.basename(node)):
            continue
        try:
            with open(os.path.join(node, "cpulist")) as f:
                if f.read().strip():
                    count += 1
        except OSError:
            pass
    return count


def explain(command=(), enabled=True):
    """Return (extra_args, reason)."""
    if not enabled or os.environ.get(ENV_VAR, "").strip().lower() in _OFF:
        return [], "disabled (--no-numa or %s=0)" % ENV_VAR
    if platform.system() != "Linux":
        return [], "not Linux"
    if "--numa" in command:
        return [], "--numa is already given"
    nodes = numa_node_count()
    if nodes < 2:
        return [], "single NUMA node"
    return list(ARGS), "%d NUMA nodes" % nodes


def numa_args(command=(), enabled=True):
    return explain(command, enabled)[0]


def evict_from_page_cache(path):
    """Drop a file's pages from the page cache (unprivileged) so the next run places them itself."""
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)  # clean pages only: a file that was just written must be flushed first
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
        return True
    except (OSError, AttributeError):
        return False


def evict_requested(flag=False):
    """True when eviction was asked for with a flag or BITNET_NUMA_EVICT=1."""
    return flag or os.environ.get(EVICT_ENV_VAR, "").strip().lower() in _ON


def with_numa(command, enabled=True, model=None, evict=False, quiet=False):
    """Return `command` with --numa distribute added when it applies, evicting `model` first if asked."""
    extra, reason = explain(command, enabled)
    if not extra:
        return list(command)
    if not quiet:
        print("Multi-socket machine (%s): adding `%s` (--no-numa or %s=0 to disable)"
              % (reason, " ".join(extra), ENV_VAR), file=sys.stderr)
    if model and evict_requested(evict):
        ok = evict_from_page_cache(model)
        if not quiet:
            print("Evicted %s from the page cache" % model if ok else "Could not evict %s" % model, file=sys.stderr)
    return list(command) + extra


if __name__ == "__main__":
    if "--explain" in sys.argv:
        extra, reason = explain()
        print(("applies: " if extra else "does not apply: ") + reason)
    elif "--args" in sys.argv:
        print(" ".join(numa_args()))
    else:
        print(__doc__)
