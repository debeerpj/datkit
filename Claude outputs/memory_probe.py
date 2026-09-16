"""Memory probe that works inside a capped container. Standard library only."""

import os

# A cgroup with no cap set reports a sentinel near 2**63, not a missing file.
# Anything above this is "unlimited", so fall back to the host's total.
_NO_LIMIT = 2**62

# cgroup v2 first, then v1. JupyterHub images run either.
_LIMIT_FILES = ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes")
_USAGE_FILES = ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes")


def _read_first(paths):
    """Return the int in the first readable file, or None."""
    for path in paths:
        try:
            with open(path) as fh:
                raw = fh.read().strip()
        except OSError:
            continue  # wrong cgroup version, or not Linux
        if raw == "max":  # cgroup v2 spells "no limit" as the word max
            return None
        try:
            value = int(raw)
        except ValueError:
            continue
        return None if value >= _NO_LIMIT else value
    return None


def _host_total():
    """Total RAM of the machine, from /proc/meminfo."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024  # reported in kB
    except OSError:
        pass
    return None


def _host_available():
    """RAM the host considers available (free plus reclaimable cache)."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def memory_status():
    """Report the ceiling that actually applies, and how much of it is used.

    Returns
    -------
    dict
        limit_bytes, used_bytes, available_bytes, source.
        `source` is 'cgroup' when a container cap applies, else 'host'.
    """
    limit = _read_first(_LIMIT_FILES)
    if limit is not None:
        used = _read_first(_USAGE_FILES) or 0
        return {"limit_bytes": limit, "used_bytes": used,
                "available_bytes": limit - used, "source": "cgroup"}

    total = _host_total()
    avail = _host_available()
    used = total - avail if total and avail else None
    return {"limit_bytes": total, "used_bytes": used,
            "available_bytes": avail, "source": "host"}


def peak_rss_bytes():
    """High-water mark of this process's resident memory, or None on Windows."""
    try:
        import resource  # Unix only; absent on Windows
    except ImportError:
        return None
    # ru_maxrss is kB on Linux, bytes on macOS.
    import sys
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def gb(n):
    """Bytes as GB, or '?' when unknown."""
    return "?" if n is None else f"{n / 1024**3:.2f} GB"


if __name__ == "__main__":
    st = memory_status()
    print(f"source    {st['source']}")
    print(f"limit     {gb(st['limit_bytes'])}")
    print(f"used      {gb(st['used_bytes'])}")
    print(f"available {gb(st['available_bytes'])}")
    print(f"peak RSS  {gb(peak_rss_bytes())}")
