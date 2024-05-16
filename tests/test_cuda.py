import subprocess
import xml.etree.ElementTree as ET

import pytest

from gpu_info import get_gpu_info


def has_nvidia_smi():
    try:
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False


def get_nvidia_smi_xml():
    result = subprocess.run(["nvidia-smi", "-q", "-x"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return ET.fromstring(result.stdout)

def parse_memory_str(s: str):
    if s.endswith(" MiB"):
        return int(s[:-3]) * 1024 * 1024
    elif s.endswith(" KiB"):
        return int(s[:-3]) * 1024
    elif s.endswith(" GiB"):
        return int(s[:-3]) * 1024 * 1024 * 1024
    else:
        raise ValueError("Unknown unit")

@pytest.mark.skipif(not has_nvidia_smi(), reason="No NVIDIA GPU detected")
def test_cuda():
    # Test against information from nvidia-smi
    info = get_nvidia_smi_xml()
    gpus = info.findall("gpu")
    gpu_count = len(gpus)
    assert gpu_count > 0
    assert gpu_count == int(info.find("attached_gpus").text)
    total = [parse_memory_str(gpu.find("fb_memory_usage").find("total").text) for gpu in gpus]
    used = [parse_memory_str(gpu.find("fb_memory_usage").find("used").text) for gpu in gpus]
    free = [parse_memory_str(gpu.find("fb_memory_usage").find("free").text) for gpu in gpus]
    for i in range(gpu_count):
        assert total[i] >= used[i] + free[i]

    gpu_info = get_gpu_info()
    assert len(gpu_info) == gpu_count
    for i in range(gpu_count):
        assert abs(gpu_info[i].total - total[i]) <= 1024 * 1024 * 512 # 512 MiB tolerance
        assert abs(gpu_info[i].free - free[i]) <= 1024 * 1024 * 512 # 512 MiB tolerance
