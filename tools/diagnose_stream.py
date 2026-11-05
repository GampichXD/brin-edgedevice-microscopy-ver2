#!/usr/bin/env python3
"""
DIAGNOSTIK STREAM VIDEO - Menguji seluruh rantai komunikasi
dari kamera -> vps_bridge -> VPS Redis -> Frontend
"""

import asyncio
import json
import os
import subprocess
import sys

# Tambah root path agar bisa import Hardware modules
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

VPS_WS_URL = os.getenv("VPS_WS_URL", "ws://127.0.0.1:8000/api/hardware/ws")

print("=" * 60)
print("    DIAGNOSTIK PENUH RANTAI STREAMING VIDEO")
print("=" * 60)

# ============================================================
# TEST 1: Cek apakah nvargus-daemon bisa diakses
# ============================================================
print("\n[TEST 1] Mengecek nvargus-daemon...")
result = subprocess.run(["pgrep", "-x", "nvargus-daemon"], capture_output=True, text=True)
if result.returncode == 0:
    print(f"  ✅ nvargus-daemon BERJALAN (PID: {result.stdout.strip()})")
else:
    print("  ❌ nvargus-daemon TIDAK BERJALAN!")
    print("     -> Jalankan: sudo systemctl restart nvargus-daemon")

# ============================================================
# TEST 2: Cek apakah ada proses lain yang memakai kamera (fd /dev/video0)
# ============================================================
print("\n[TEST 2] Mengecek proses yang memegang /dev/video0...")
result = subprocess.run(["fuser", "/dev/video0"], capture_output=True, text=True)
if result.stdout.strip():
    print(f"  ⚠️  /dev/video0 dipegang oleh PID: {result.stdout.strip()}")
    print("     -> Ada proses lain yang menggunakan kamera! Kill dulu:")
    print(f"        kill -9 {result.stdout.strip()}")
else:
    print("  ✅ /dev/video0 bebas, tidak ada proses yang memegang.")

# ============================================================
# TEST 3: Cek apakah ada proses main.py yang masih berjalan
# ============================================================
print("\n[TEST 3] Mengecek sisa proses main.py dari sesi sebelumnya...")
result = subprocess.run(["pgrep", "-a", "-f", "main.py"], capture_output=True, text=True)
if result.stdout.strip():
    print(f"  ⚠️  Ditemukan proses main.py yang masih berjalan:")
    for line in result.stdout.strip().split('\n'):
        print(f"     {line}")
    print("     -> Hentikan dulu sebelum menjalankan lagi!")
else:
    print("  ✅ Tidak ada proses main.py yang tersisa.")

# ============================================================
# TEST 4: Uji GStreamer - dapat satu frame?
# ============================================================
print("\n[TEST 4] Menguji GStreamer pipeline (ambil 1 frame)...")
gst_cmd = [
    "gst-launch-1.0", "-q",
    "nvarguscamerasrc", "sensor-id=0", "num-buffers=1", "!",
    "video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1", "!",
    "nvvidconv", "!",
    "video/x-raw,width=320,height=240,format=(string)BGRx", "!",
    "videoconvert", "!",
    "video/x-raw,format=(string)BGR", "!",
    "fdsink", "fd=1"
]

proc = subprocess.Popen(gst_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
try:
    raw, stderr = proc.communicate(timeout=8)
    expected = 320 * 240 * 3
    if len(raw) == expected:
        print(f"  ✅ GStreamer BERHASIL! Dapat frame berukuran {len(raw)} bytes")
    else:
        print(f"  ❌ GStreamer GAGAL! Dapat {len(raw)} bytes (harusnya {expected})")
        if stderr:
            err_lines = stderr.decode('utf-8', errors='ignore').strip().split('\n')
            for line in err_lines[:5]:
                print(f"     STDERR: {line}")
except subprocess.TimeoutExpired:
    proc.kill()
    print("  ❌ GStreamer TIMEOUT (>8 detik) - sensor kemungkinan hang/busy")

# ============================================================
# TEST 5: Cek koneksi ke VPS WebSocket
# ============================================================
print("\n[TEST 5] Menguji koneksi WebSocket ke VPS...")
print(f"  VPS URL: {VPS_WS_URL}")

async def test_ws():
    try:
        import websockets
        async with websockets.connect(VPS_WS_URL, open_timeout=5) as ws:
            print("  ✅ Berhasil terhubung ke VPS WebSocket!")
            await ws.send(json.dumps({"status": "DIAGNOSTIC_PING", "device": "TEST"}))
            print("  ✅ Berhasil mengirim ping ke VPS.")
            return True
    except Exception as e:
        print(f"  ❌ GAGAL terhubung ke VPS: {e}")
        return False

asyncio.run(test_ws())

print("\n" + "=" * 60)
print("    DIAGNOSTIK SELESAI")
print("=" * 60)
