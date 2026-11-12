"""Pembantu bersama skrip uji. Aman di-import di laptop (tanpa kamera, tanpa GRBL)."""
import asyncio
import csv
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_LOG_DIR = os.path.join(ROOT, "test_logs")
HEARTBEAT_INTERVAL_S = 20  # tanpa HEARTBEAT, server memutus klien setelah 90 detik
# Pong bisa tertahan di belakang frame video saat Wi-Fi penuh. Batas bawaan 20 detik
# memutus klien uji padahal koneksinya masih jalan.
KEEPALIVE_TIMEOUT_S = 120


def append_row(log_dir, name, **fields):
    """Tambah satu baris ke <log_dir>/<name>.csv. Tidak pernah menimpa data lama."""
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"{name}.csv")
    new_file = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields.keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(fields)
    return path


def percentile(values, q):
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def require_env(*names):
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        sys.exit(f"Env wajib belum diisi: {', '.join(missing)}")


def login():
    """Login lewat /api/auth/login dengan BASE_URL, TEST_USER, TEST_PASS. Mengembalikan (base, token)."""
    import requests

    require_env("BASE_URL", "TEST_USER", "TEST_PASS")
    base = os.environ["BASE_URL"].rstrip("/")
    res = requests.post(f"{base}/api/auth/login", timeout=15, json={
        "username": os.environ["TEST_USER"], "password": os.environ["TEST_PASS"]})
    if res.status_code != 200:
        sys.exit(f"Login gagal ({res.status_code}): {res.text[:200]}")
    return base, res.json()["access_token"]


def auth_headers(token):
    return {"Authorization": f"Bearer {token}"}


def client_ws_url(base, token):
    return base.replace("http", "ws", 1) + f"/api/hardware/client/ws?token={token}"


async def heartbeat_loop(ws):
    while True:
        await asyncio.sleep(HEARTBEAT_INTERVAL_S)
        await ws.send(json.dumps({"action": "HEARTBEAT"}))


def ensure_exclusive_hardware():
    """t01, t02, dan t05b memakai serial dan kamera langsung: main.py harus mati."""
    import psutil

    for proc in psutil.process_iter(["pid", "cmdline"]):
        cmd = " ".join(proc.info.get("cmdline") or [])
        if "main.py" in cmd and "Hardware" in cmd and proc.info["pid"] != os.getpid():
            sys.exit(f"main.py masih berjalan (PID {proc.info['pid']}). Matikan dulu: port serial "
                     "dan kamera tidak bisa dipakai dua proses sekaligus.")
    parent = os.path.dirname(ROOT)
    if parent not in sys.path:
        sys.path.insert(0, parent)


def open_motor():
    """Buka GRBL fisik. Mode simulasi ditolak: datanya tidak bermakna untuk Bab IV."""
    from Hardware.Devices.motor import motor_core

    if not motor_core.open() or motor_core.is_mock_mode:  # mock hanya kalau ALLOW_MOCK_HARDWARE=1
        sys.exit("GRBL tidak bisa dibuka. Periksa kabel, dan pastikan port tidak dipakai proses lain.")
    motor_core.unlock()
    return motor_core


def wait_idle(motor, timeout=60.0):
    """Tunggu GRBL benar-benar berhenti. Balasan 'ok' hanya berarti perintah masuk antrean."""
    time.sleep(0.1)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if motor.get_status()["status"] == "Idle":
            return True
        time.sleep(0.05)
    return False


def fresh_frame(camera, settle_s=0.0, discard=3, gap=0.05):
    """Frame baru setelah jeda settle. Beberapa frame dibuang karena buffer kamera bisa
    masih berisi citra dari sebelum motor berhenti."""
    if settle_s > 0:
        time.sleep(settle_s)
    frame = None
    for _ in range(discard + 1):
        time.sleep(gap)
        frame = camera.capture_frame()
    return frame


async def connect_as_pilot(url, timeout=10.0):
    """Buka WebSocket klien dengan peran PILOT. Akun ADMIN otomatis mengambil alih kursi."""
    import websockets

    ws = await websockets.connect(url, max_size=None, ping_interval=20, ping_timeout=KEEPALIVE_TIMEOUT_S)
    deadline = time.monotonic() + timeout
    role = None
    while time.monotonic() < deadline:
        try:
            message = json.loads(await asyncio.wait_for(ws.recv(), 1.0))
        except asyncio.TimeoutError:
            continue
        if message.get("event") == "ROLE_ASSIGNED":
            role = message.get("role")
            if role == "PILOT":
                return ws
            await ws.send(json.dumps({"action": "TAKEOVER"}))
    await ws.close()
    sys.exit(f"Tidak mendapat peran PILOT (terakhir: {role}). Tutup sesi lain atau pakai akun ADMIN.")


def open_camera():
    from Hardware.Devices.camera import camera_core

    camera_core.open()
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if camera_core.capture_frame() is not None:
            return camera_core
        time.sleep(0.2)
    sys.exit("Kamera tidak menghasilkan frame.")
