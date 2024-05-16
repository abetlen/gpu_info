import ctypes
import typing

libcudart = ctypes.CDLL("libcudart.so")

libcudart.cudaSetDevice.argtypes = [ctypes.c_int]
libcudart.cudaSetDevice.restype = ctypes.c_int

libcudart.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
libcudart.cudaGetDeviceCount.restype = ctypes.c_int

libcudart.cudaMemGetInfo.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
libcudart.cudaMemGetInfo.restype = ctypes.c_int

def get_device_count() -> int:
    count = ctypes.c_int()
    rc = libcudart.cudaGetDeviceCount(ctypes.byref(count))
    if rc != 0:
        raise RuntimeError(f"cudaGetDeviceCount failed with error code {rc}")
    return count.value

def get_device_vram(device: int) -> typing.Tuple[int, int]:
    rc = libcudart.cudaSetDevice(device)
    if rc != 0:
        raise RuntimeError(f"cudaSetDevice failed with error code {rc}")
    total = ctypes.c_size_t()
    free = ctypes.c_size_t()
    rc = libcudart.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total))
    if rc != 0:
        raise RuntimeError(f"cudaMemGetInfo failed with error code {rc}")
    return total.value, free.value
