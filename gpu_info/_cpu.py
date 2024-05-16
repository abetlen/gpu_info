import platform
import subprocess

def get_ram_info():
    if platform.system() == "Linux":
        with open('/proc/meminfo', 'r') as f:
            lines = f.readlines()
            total_ram = int(lines[0].split(':')[1].strip().split()[0]) * 1024
            free_ram = int(lines[2].split(':')[1].strip().split()[0]) * 1024
    elif platform.system() == "Windows":
        total_ram = int(subprocess.check_output(['wmic', 'os', 'get', '/value', 'TotalVisibleMemorySize']).decode().strip().split('=')[1])
        free_ram = int(subprocess.check_output(['wmic', 'os', 'get', '/value', 'FreePhysicalMemory']).decode().strip().split('=')[1])
    elif platform.system() == "Darwin":
        total_ram = int(subprocess.check_output(['sysctl', '-n', 'hw.memsize']).decode().strip())
        free_ram = int(subprocess.check_output(['sysctl', '-n', 'vm.vmtotal']).decode().strip())
    else:
        raise NotImplementedError("Unsupported platform")
    return total_ram, free_ram
