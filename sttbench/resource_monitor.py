"""Lightweight per-process resource monitor.
Samples the CURRENT process (PID) at interval and aggregates min/avg/max CPU%, RSS, and GPU if available.
"""
from __future__ import annotations
import os, threading, time, atexit
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import psutil

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML = True
except Exception:
    _NVML = False

@dataclass
class ResourceStats:
    duration_sec: float = 0.0
    cpu_avg_percent: float = 0.0
    cpu_max_percent: float = 0.0
    rss_avg_mb: float = 0.0
    rss_max_mb: float = 0.0
    gpu_mem_max_mb: Optional[float] = None
    gpu_util_max_percent: Optional[float] = None
    samples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()

class ResourceMonitor:
    def __init__(self, interval: float = 0.2):
        self.interval = interval
        self.proc = psutil.Process(os.getpid())
        self.stats = ResourceStats()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        # Prime CPU percent measurement
        self.proc.cpu_percent(None)
        self._t0 = time.time()
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        self._thread.join()
        self.stats.duration_sec = time.time() - self._t0

    def _run(self):
        rss_sum = 0.0
        cpu_sum = 0.0
        count = 0
        gpu_mem_max = None
        gpu_util_max = None
        handle = None
        if _NVML:
            try:
                h = pynvml.nvmlDeviceGetHandleByIndex(0)
                handle = h
            except Exception:
                handle = None
        while not self._stop.is_set():
            try:
                rss = self.proc.memory_info().rss / (1024 ** 2)
                cpu = self.proc.cpu_percent(None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            self.stats.rss_max_mb = max(self.stats.rss_max_mb, rss)
            self.stats.cpu_max_percent = max(self.stats.cpu_max_percent, cpu)
            rss_sum += rss
            cpu_sum += cpu
            count += 1

            if handle is not None:
                try:
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_mem = mem.used / (1024 ** 2)
                    gpu_u = util.gpu
                    gpu_mem_max = gpu_mem if gpu_mem_max is None else max(gpu_mem_max, gpu_mem)
                    gpu_util_max = gpu_u if gpu_util_max is None else max(gpu_util_max, gpu_u)
                except Exception:
                    pass

            time.sleep(self.interval)
        if count:
            self.stats.rss_avg_mb = rss_sum / count
            self.stats.cpu_avg_percent = cpu_sum / count
        if gpu_mem_max is not None:
            self.stats.gpu_mem_max_mb = gpu_mem_max
        if gpu_util_max is not None:
            self.stats.gpu_util_max_percent = gpu_util_max

# Ensure NVML shutdown on exit
if _NVML:
    atexit.register(lambda: pynvml.nvmlShutdown())