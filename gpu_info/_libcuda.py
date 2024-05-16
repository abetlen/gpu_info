import ctypes
import typing

from ._exceptions import GPUInfoProviderNotAvailable

try:
    libcuda = ctypes.CDLL("libcuda.so")
except OSError:
    raise GPUInfoProviderNotAvailable()

libcuda.cuInit.argtypes = [ctypes.c_uint]
libcuda.cuInit.restype = ctypes.c_int

rc = libcuda.cuInit(0) != 0
if rc != 0:
    raise RuntimeError(f"cuInit failed with error code {rc}")

libcuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
libcuda.cuDeviceGet.restype = ctypes.c_int

libcuda.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
libcuda.cuDeviceGetCount.restype = ctypes.c_int

libcuda.cuCtxCreate_v3.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
libcuda.cuCtxCreate_v3.restype = ctypes.c_int

libcuda.cuMemGetInfo_v2.argtypes = [
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_size_t),
]
libcuda.cuMemGetInfo_v2.restype = ctypes.c_int

libcuda.cuCtxDestroy.argtypes = [ctypes.c_void_p]
libcuda.cuCtxDestroy.restype = ctypes.c_int


def get_device_count() -> int:
    count = ctypes.c_int()
    rc = libcuda.cuDeviceGetCount(ctypes.byref(count))
    if rc != 0:
        raise RuntimeError(f"cuDeviceGetCount failed with error code {rc}")
    return count.value


def get_device_vram(device: int) -> typing.Tuple[int, int]:
    handle = ctypes.c_void_p()
    rc = libcuda.cuCtxCreate_v3(ctypes.byref(handle), None, 0, 0, device)
    if rc != 0:
        raise RuntimeError(f"cuCtxCreate_v3 failed with error code {rc}")

    def cleanup():
        rc = libcuda.cuCtxDestroy(handle)
        if rc != 0:
            raise RuntimeError(f"cuCtxDestroy failed with error code {rc}")

    total = ctypes.c_size_t()
    free = ctypes.c_size_t()
    rc = libcuda.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total))
    if rc != 0:
        cleanup()
        raise RuntimeError(f"cuMemGetInfo_v2 failed with error code {rc}")
    cleanup()
    return total.value, free.value


def get_gpu_info() -> typing.List[typing.Tuple[int, int]]:
    count = get_device_count()
    return [get_device_vram(i) for i in range(count)]
