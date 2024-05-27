import sys
from collections import defaultdict

import gguf

import gpu_info

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <gguf-file>")
        sys.exit(1)

    filepath = sys.argv[-1]
    gguf_file = gguf.GGUFReader(filepath)

    total_size = 0
    layer_size = defaultdict(int)
    total_layer_size = 0
    for tensor in gguf_file.tensors:
        total_size += tensor.n_bytes
        if tensor.name.startswith("blk."):
            layer_number = int(tensor.name.split(".")[1])
            layer_size[layer_number] += tensor.n_bytes
            total_layer_size += tensor.n_bytes
    
    print(f"Total size: {total_size / 1024**2:.2f} MB")
    for layer_number, size in sorted(layer_size.items()):
        print(f"Layer {layer_number}: {size / 1024**2:.2f} MB")


    info = gpu_info.get_gpu_info(include_cpu=False)
    if len(info) == 0:
        print("No GPU information available")
    
    gpu = info[0]
    print(f"GPU: {gpu.to_dict()}")

    total_memory = gpu.total_memory

    # Estimate the maximum number of layers that can fit in the GPU memory
    available_size = total_memory
    n_gpu_layers = 0
    for layer_number, size in sorted(layer_size.items()):
        if available_size < size:
            break
        available_size -= size
        n_gpu_layers += 1

    print(f"Estimated number of layers that can fit in the GPU memory: {n_gpu_layers}")