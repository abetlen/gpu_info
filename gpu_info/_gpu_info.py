from __future__ import annotations

import typing

from ._cudart import get_device_count, get_device_vram

class GPUInfo(typing.NamedTuple):
    total: int
    free: int

def get_gpu_info() -> typing.List[GPUInfo]:
    count = get_device_count()
    return [GPUInfo(*get_device_vram(i)) for i in range(count)]
