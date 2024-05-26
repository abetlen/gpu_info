from dataclasses import dataclass, asdict


@dataclass
class GPUInfo:
    """Dataclass for GPU information"""

    backend: str
    provider: str
    total_memory: int
    free_memory: int

    def to_dict(self):
        return asdict(self)
