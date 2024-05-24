from . import get_gpu_info


def main():
    gpu_info = get_gpu_info()
    for i, info in enumerate(gpu_info):
        print(f"GPU {i}: {info}")