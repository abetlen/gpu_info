import argparse

from . import get_gpu_info, __version__


def main():
    parser = argparse.ArgumentParser(description="Get GPU information")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    parser.add_argument("--no-cpu", action="store_true", help="Do not include CPU information")
    args = parser.parse_args()

    gpu_info = get_gpu_info(include_cpu=not args.no_cpu)

    if args.json:
        import json
        print(json.dumps([info._asdict() for info in gpu_info], indent=4))
    else:
        for i, info in enumerate(gpu_info):
            print(f"Device {i}: {info}")