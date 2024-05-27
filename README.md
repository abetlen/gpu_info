# `gpu_info`: Small cross-platform Python package to get GPU information

## Installation

```bash
pip install gpu_info git+https://github.com/abetlen/gpu_info.git
```

## Usage

```python
import gpu_info

print(gpu_info.get_info())
```

## Providers

| Backend | Provider    | Package                                             |
| ------- | ----------- | --------------------------------------------------- |
| CPU     | `cpu`       | `included`                                          |
| CUDA    | `libcuda`   | `included`                                          |
| CUDA    | `libcudart` | `included`                                          |
| Vulkan  | `vulkan`    | [vulkan](https://pypi.org/project/vulkan/)          |
| Metal   | `metal`     | [metal_info](https://github.com/abetlen/metal_info) |
