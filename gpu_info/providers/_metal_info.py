from .._types import GPUInfo
from .._exceptions import GPUInfoProviderNotAvailable

BACKEND = "metal"
PROVIDER = __name__

try:
    from metal_info import (
        recommended_max_vram,
        physical_memory,
        current_allocated_size,
    )
except ImportError:
    raise GPUInfoProviderNotAvailable()


def get_info(
    use_recommended_max_vram: bool = True,
):
    return [
        GPUInfo(
            backend=BACKEND,
            provider=PROVIDER,
            total_memory=(
                physical_memory()
                if not use_recommended_max_vram
                else recommended_max_vram()
            ),
            free_memory=current_allocated_size(),
        )
    ]
