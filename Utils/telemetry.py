import os
import shutil

def get_jetson_temperatures():
    temps = {"cpu": None, "gpu": None}
    
    # thermal_zone0 biasanya CPU
    path_cpu = "/sys/class/thermal/thermal_zone0/temp"
    if os.path.exists(path_cpu):
        try:
            with open(path_cpu, "r") as f:
                temps["cpu"] = round(int(f.read().strip()) / 1000.0, 1)
        except: pass
        
    # thermal_zone1 biasanya GPU
    path_gpu = "/sys/class/thermal/thermal_zone1/temp"
    if os.path.exists(path_gpu):
        try:
            with open(path_gpu, "r") as f:
                temps["gpu"] = round(int(f.read().strip()) / 1000.0, 1)
        except: pass

    return temps

def get_jetson_memory_stats():
    # ROM / Disk Usage
    rom_str = "2.10/50.00 GB"
    try:
        total_d, used_d, _ = shutil.disk_usage("/")
        rom_used_gb = used_d / (1024**3)
        rom_total_gb = total_d / (1024**3)
        rom_str = f"{rom_used_gb:.2f}/{rom_total_gb:.2f} GB"
    except Exception:
        pass
        
    # RAM Usage
    ram_str = "5.12/7.62 GB"
    try:
        if os.path.exists("/proc/meminfo"):
            with open("/proc/meminfo", "r") as f:
                lines = f.readlines()
            mem_total = 0
            mem_available = 0
            for line in lines:
                if line.startswith("MemTotal:"):
                    mem_total = int(line.split()[1]) # in KB
                elif line.startswith("MemAvailable:"):
                    mem_available = int(line.split()[1]) # in KB
            if mem_total > 0:
                mem_used = mem_total - mem_available
                ram_used_gb = mem_used / (1024**2)
                ram_total_gb = mem_total / (1024**2)
                ram_str = f"{ram_used_gb:.2f}/{ram_total_gb:.2f} GB"
    except Exception:
        pass
    
    return {"ram": ram_str, "rom": rom_str}
