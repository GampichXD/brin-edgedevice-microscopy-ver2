#!/usr/bin/env python3
"""
====================================================================
MAIN ENTRY POINT - EDGE HARDWARE CORE SYSTEM (JETSON ORIN NANO)
====================================================================
File ini berfungsi sebagai gerbang utama untuk menyalakan seluruh 
subsistem mekatronika, driver device, dan jembatan data internet (VPS).
"""

import sys
import os
import asyncio

# ====================================================================
# FIX RADIKAL: FORCE ROOT PATH RESOLUTION UNTUK WINDOWS & LINUX
# ====================================================================
# Ambil path absolut folder induk 'Tugas Akhir'
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Bersihkan path lokal folder 'Hardware' dari sys.path jika ada di urutan pertama
if sys.path and sys.path[0] == os.path.dirname(os.path.abspath(__file__)):
    sys.path.pop(0)

# Masukkan folder root 'Tugas Akhir' ke urutan paling depan pencarian Python
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Sekarang import absolut dengan nama 'Hardware' dijamin 100% aman dan terbaca!
from Hardware.Backend_Connector.vps_bridge import hardware_control_loop
from Hardware.Devices.motor import motor_core
from Hardware.Devices.camera import camera_core
import threading
from Hardware.Backend_Connector.vps_syncer import run_daemon

def print_banner():
    """Mencetak identitas sistem saat startup di terminal Jetson Orin Nano."""
    print("="*65)
    print("       EDGE COMPUTING MICROSCOPE HARDWARE SYSTEM INTERFACE      ")
    print("         Concentration: Information Technology - UNDIP         ")
    print("="*65)
    print("[SYSTEM] Menginisialisasi dependensi lokal...")

def shutdown_system():
    """Memastikan pemutusan koneksi fisik dan pembersihan memori berjalan steril."""
    print("\n" + "="*65)
    print("[SHUTDOWN] Menghentikan sirkuit komputasi edge...")
    
    try:
        camera_core.close()
    except Exception as e:
        print(f"[SHUTDOWN ERROR] Gagal melepas sensor kamera: {e}")
        
    try:
        motor_core.close()
    except Exception as e:
        print(f"[SHUTDOWN ERROR] Gagal memutuskan serial port GRBL: {e}")
        
    print("[SHUTDOWN SUCCESS] Seluruh pin hardware aman. Sesi terminal ditutup.")
    print("="*65)

if __name__ == "__main__":
    print_banner()
    
    # Jalankan daemon sinkronisasi cloud di background (non-blocking)
    syncer_thread = threading.Thread(target=run_daemon, daemon=True)
    syncer_thread.start()
    
    try:
        # Menjalankan loop utama asinkron multi-tasking dari vps_bridge
        asyncio.run(hardware_control_loop())
        
    except KeyboardInterrupt:
        # Menangkap interupsi tombol CTRL + C dari terminal user
        print("\n[SYSTEM WARNING] Menerima sinyal interupsi manual (SIGINT).")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Sistem mengalami kegagalan fatal: {e}")
    finally:
        # Sirkuit penutup wajib untuk mencegah locked-port di OS Linux
        shutdown_system()