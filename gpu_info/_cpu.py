import os
import platform

from ._types import GPUInfo

BACKEND = "cpu"


def _get_memory_info():
    if platform.system() == "Windows":
        # Windows
        total_memory = int(
            os.popen("wmic os get TotalVisibleMemorySize").read().split("=")[1].strip()
        )
        free_memory = int(
            os.popen("wmic os get FreePhysicalMemory").read().split("=")[1].strip()
        )
    elif platform.system() == "Darwin":
        # macOS
        total_memory = int(os.popen("sysctl -n hw.memsize").read().strip())
        free_memory = int(os.popen("sysctl -n hw.physmem").read().strip())
    else:
        # Linux
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total_memory = int(line.split(":")[1].strip().split(" ")[0]) * 1024
                elif line.startswith("MemAvailable:"):
                    free_memory = int(line.split(":")[1].strip().split(" ")[0]) * 1024
    return total_memory, free_memory


def get_gpu_info():
    total_memory_gb, free_memory_gb = _get_memory_info()
    return [(total_memory_gb, free_memory_gb)]


def get_info():
    total_memory_gb, free_memory_gb = _get_memory_info()
    return [
        GPUInfo(
            backend=BACKEND,
            total_memory=total_memory_gb,
            free_memory=free_memory_gb,
        )
    ]
