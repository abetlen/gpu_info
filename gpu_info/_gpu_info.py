from __future__ import annotations

import typing

from ._exceptions import GPUInfoProviderNotAvailable

from ._cpu import get_gpu_info as cpu_get_gpu_info

try:
    from ._libcuda import get_gpu_info as cuda_get_gpu_info
except GPUInfoProviderNotAvailable:
    cuda_get_gpu_info = None

try:
    from ._libcudart import get_gpu_info as cudart_get_gpu_info
except GPUInfoProviderNotAvailable:
    cudart_get_gpu_info = None

try:
    from ._vulkan import get_gpu_info as vulkan_get_gpu_info
except GPUInfoProviderNotAvailable:
    vulkan_get_gpu_info = None


class GPUInfo(typing.NamedTuple):
    backend: str
    total: int
    free: int


def get_gpu_info(include_cpu: bool = False) -> typing.List[GPUInfo]:
    gpu_info: typing.List[GPUInfo] = []

    if include_cpu:
        gpu_info.extend(GPUInfo("cpu", *info) for info in cpu_get_gpu_info())

    if cuda_get_gpu_info is not None:
        gpu_info.extend(GPUInfo("cuda", *info) for info in cuda_get_gpu_info())
    elif cudart_get_gpu_info is not None:
        gpu_info.extend(GPUInfo("cuda", *info) for info in cudart_get_gpu_info())
    
    if vulkan_get_gpu_info is not None:
        gpu_info.extend(GPUInfo("vulkan", *info) for info in vulkan_get_gpu_info())

    return gpu_info
