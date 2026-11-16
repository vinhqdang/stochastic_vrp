"""Measurement helpers for the pilot timing stage (EXPERIMENT_A.md sections 7.2 and 7.3, P5).

``traced_peak`` reports the peak of Python and NumPy allocations inside a block
(tracemalloc tracks NumPy buffers). ``process_peak_rss_mb`` reports the process
high-water mark (resource.getrusage, kilobytes on Linux; GetProcessMemoryInfo
peak working set on Windows, where ``resource`` does not exist). ``Stopwatch``
records named wall-clock intervals. None of them changes any computed value.
"""
from __future__ import annotations

import time
import tracemalloc
from contextlib import contextmanager

try:
    import resource
except ImportError:  # Windows
    resource = None

MIB = float(2 ** 20)


def process_peak_rss_mb() -> float:
    """Process peak resident set size in MiB (ru_maxrss is in KiB on Linux)."""
    if resource is not None:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return _windows_peak_working_set_mb()


def _windows_peak_working_set_mb() -> float:
    """Peak working set of the current process in MiB, via psapi GetProcessMemoryInfo."""
    import ctypes
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters),
                                           wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(ProcessMemoryCounters)
    if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters),
                                      counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return counters.PeakWorkingSetSize / MIB


@contextmanager
def traced_peak():
    """Yields a dict that is filled on exit with ``peak_mb`` (peak traced
    allocation above the level at entry) and ``retained_mb`` (allocation still
    alive at exit, above the level at entry). Starts tracemalloc if needed."""
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    tracemalloc.reset_peak()
    base, _ = tracemalloc.get_traced_memory()
    box: dict = {}
    try:
        yield box
    finally:
        cur, peak = tracemalloc.get_traced_memory()
        box["peak_mb"] = (peak - base) / MIB
        box["retained_mb"] = (cur - base) / MIB
        if not was_tracing:
            tracemalloc.stop()


class Stopwatch:
    """Named wall-clock intervals: ``with sw.lap("gen"): ...`` then ``sw.seconds``."""

    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}

    @contextmanager
    def lap(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[name] = self.seconds.get(name, 0.0) + (time.perf_counter() - t0)
