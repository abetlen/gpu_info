from __future__ import annotations

import typing

from ._types import GPUInfo
from ._exceptions import GPUInfoProviderNotAvailable

from .providers._cpu import get_info as cpu_get_gpu_info

try:
    from .providers._libcuda import get_info as cuda_get_gpu_info
except GPUInfoProviderNotAvailable:
    cuda_get_gpu_info = None

try:
    from .providers._libcudart import get_info as cudart_get_gpu_info
except GPUInfoProviderNotAvailable:
    cudart_get_gpu_info = None

try:
    from .providers._vulkan import get_info as vulkan_get_gpu_info
except GPUInfoProviderNotAvailable:
    vulkan_get_gpu_info = None

try:
    from .providers._metal_info import get_info as metal_get_gpu_info
except GPUInfoProviderNotAvailable:
    metal_get_gpu_info = None


def get_gpu_info(include_cpu: bool = False) -> typing.List[GPUInfo]:
    gpu_info: typing.List[GPUInfo] = []

    if include_cpu:
        gpu_info.extend(cpu_get_gpu_info())

    if cuda_get_gpu_info is not None:
        gpu_info.extend(cuda_get_gpu_info())
    elif cudart_get_gpu_info is not None:
        gpu_info.extend(cudart_get_gpu_info())

    if vulkan_get_gpu_info is not None:
        gpu_info.extend(vulkan_get_gpu_info())

    if metal_get_gpu_info is not None:
        gpu_info.extend(metal_get_gpu_info())

    return gpu_info
