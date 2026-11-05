import asyncio
import json
import websockets
import base64
import os
import sys
import cv2
import time
import psutil
from urllib.parse import quote

# Impor driver inti dari folder Hardware
from Hardware.Devices.motor import motor_core
from Hardware.Devices.camera import camera_core

from Hardware.Utils.telemetry import get_jetson_temperatures, get_jetson_memory_stats
from Hardware.Utils import edge_state, test_logger
from Hardware.Handlers.motor_handler import handle_motor_action
from Hardware.Handlers.camera_handler import handle_camera_action
from Hardware.Handlers.ai_handler import handle_ai_action

# Konfigurasi Koneksi VPS & Direktori Edge
VPS_WS_URL = os.getenv("VPS_WS_URL", "ws://127.0.0.1:8000/api/hardware/ws")
EDGE_TOKEN_REJECTED_BACKOFF_S = 30

# Watchdog koneksi. VPS mengirim PING tiap 1 detik (lihat hardware.py /ws).
WATCHDOG_TIMEOUT = float(os.getenv("WATCHDOG_TIMEOUT", "3"))
WATCHDOG_TICK_S = 0.25
WATCHDOG_MAX_LOOP_LAG_S = 0.5
PING_MESSAGE = '{"action": "PING"}'
# Keepalive transport hanya untuk mendeteksi socket mati supaya bisa reconnect; motor
# dihentikan oleh watchdog di atas. Timeout dibuat longgar: pong tertahan selama VPS
# mengirim payload besar (mis. panorama base64 untuk CV) dan koneksi tidak boleh ikut putus.
WS_PING_INTERVAL = float(os.getenv("WS_PING_INTERVAL", "5"))
WS_PING_TIMEOUT = float(os.getenv("WS_PING_TIMEOUT", "20"))


def _edge_ws_url() -> str:
    token = os.getenv("EDGE_DEVICE_TOKEN", "")
    if not token:
        print("[EDGE WARNING] EDGE_DEVICE_TOKEN belum diisi di .env; VPS bisa menolak koneksi ini.")
        return VPS_WS_URL
    return f"{VPS_WS_URL}?device_token={quote(token, safe='')}"

# Global lock to prevent websocket send collisions between video stream and large image payloads
ws_lock = asyncio.Lock()

# State global disatukan dalam satu dictionary mutable
state = {
    "is_streaming": False,
    "is_recording": False,
    "recording_pending_init": False,
    "video_writer": None,
    "target_fps": 30,
    "recording_filepath": "",
    "recording_final_filepath": ""
}

class LinkState:
    """Status link VPS untuk watchdog. Watchdog baru aktif (armed) setelah PING pertama
    dari server, sehingga Jetson baru yang tersambung ke VPS lama tidak salah memicu hold."""
    def __init__(self):
        self.last_rx = time.monotonic()
        self.armed = False
        self.hold_sent = False

    def age_ms(self) -> float:
        return (time.monotonic() - self.last_rx) * 1000.0


_link_lost_since_connect = False


def _set_link_ok() -> None:
    global _link_lost_since_connect
    _link_lost_since_connect = False


def _log_watchdog(event: str, age_ms: float) -> None:
    test_logger.log("watchdog_log", always=True, t_ns=time.time_ns(), event=event,
                    last_rx_age_ms=round(age_ms, 1), motor_state=motor_core.last_status)


def _hold_motor(link: LinkState, event: str) -> None:
    """Motor berhenti saat link hilang. Tidak pernah resume otomatis: gerak berikutnya
    hanya lewat perintah baru, yang akan melepas Hold (lihat GRBLMotorCore.clear_hold)."""
    if link.hold_sent:
        return
    global _link_lost_since_connect
    _link_lost_since_connect = True
    link.hold_sent = True
    age = link.age_ms()
    _log_watchdog(event, age)
    if motor_core.feed_hold():
        _log_watchdog("FEED_HOLD_SENT", age)


async def receive_handler(websocket, link: LinkState, commands: asyncio.Queue):
    """TASK 1: Fokus membaca pesan masuk dari VPS. Tidak mengeksekusi apa pun di sini,
    supaya perintah lambat (capture, ganti setting kamera) tidak terbaca sebagai link putus."""
    async for message in websocket:
        link.last_rx = time.monotonic()
        t_rx_ns = time.time_ns()
        if link.hold_sent:
            link.hold_sent = False
            _log_watchdog("LINK_RESTORED", 0.0)
            _set_link_ok()
        if message == PING_MESSAGE:
            link.armed = True
            continue
        commands.put_nowait((message, t_rx_ns))


async def command_worker(websocket, commands: asyncio.Queue):
    """Mengeksekusi perintah dari VPS satu per satu, sesuai urutan datangnya."""
    while True:
        message, t_rx_ns = await commands.get()
        try:
            data = json.loads(message)
            data["_t_rx_ns"] = t_rx_ns
            action = data.get("action", "").upper()
            
            if action == "RESTART_EDGE":
                print("[BRIDGE] ♻️  Perintah ADMIN: restart layanan Edge (os.execv).")
                try:
                    async with ws_lock:
                        await websocket.send(json.dumps({
                            "event": "EDGE_RESTARTING", "status": "OK"
                        }))
                except Exception:
                    pass
                await asyncio.sleep(0.5)
                try:
                    camera_core.close()
                except Exception:
                    pass
                try:
                    motor_core.close()
                except Exception:
                    pass
                main_py = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "main.py"))
                print(f"[BRIDGE] Menjalankan ulang: {sys.executable} {main_py}")
                os.execv(sys.executable, [sys.executable, main_py])

            if action in ["MOVE_MOTOR", "HOMING", "UNLOCK", "SET_POSITION", "APPLY_CNC_SETTINGS", "APPLY_SOFT_LIMITS"]:
                await handle_motor_action(action, data, websocket, ws_lock)
            elif action in ["APPLY_CAMERA_SETTINGS", "START_STREAM", "STOP_STREAM", "SET_FPS", "START_RECORDING", "STOP_RECORDING", "CAPTURE_IMAGE", "CALIBRATE_FLATFIELD", "PURGE_CACHE", "PURGE_ALL_DATASETS", "PURGE_DATASET_FOLDER", "PURGE_DATASET_FILES"]:
                await handle_camera_action(action, data, websocket, ws_lock, state)
            elif action in ["START_STITCHING", "START_DL_COUNT", "APPLY_IMAGE_EDIT", "CV_COLONY_COUNT", "CV_THRESHOLD", "CV_CONTOUR", "CV_MORPHOLOGY", "CV_SOBEL", "CV_ROI", "CV_CALIBRATE", "CV_COLOR_SPLIT"]:
                await handle_ai_action(action, data, websocket, ws_lock, state)
            elif action == "PING":
                pass # Abaikan sinyal keep-alive
            else:
                print(f"[BRIDGE WARNING] Action tidak dikenali: {action}")

        except Exception as e:
            print(f"[BRIDGE ERROR] Gagal memproses data handler: {e}")


async def watchdog_task(link: LinkState):
    """Tahan motor kalau tidak ada pesan dari VPS selama WATCHDOG_TIMEOUT detik."""
    while True:
        t0 = time.monotonic()
        await asyncio.sleep(WATCHDOG_TICK_S)
        # Event loop sempat macet: PING mungkin sudah menunggu di socket, jadi beri
        # kesempatan reader membacanya dulu sebelum menyimpulkan link putus.
        if time.monotonic() - t0 - WATCHDOG_TICK_S > WATCHDOG_MAX_LOOP_LAG_S:
            continue
        if link.armed and link.age_ms() > WATCHDOG_TIMEOUT * 1000.0:
            _hold_motor(link, "WATCHDOG_TRIGGERED")


async def send_stream_task(websocket, state):
    """TASK 2: Membaca frame JPEG dari hardware Edge Camera secara asinkron lalu disiarkan ke VPS."""
    print("[BRIDGE] Task pemancar video stream aktif di background loop.")
    while True:
        start_t = time.time()
        try:
            # Selama tile stitching berat, kamera ditutup & RAM dihemat —
            # jangan sentuh kamera / encode / kirim frame sama sekali.
            if edge_state.stitching:
                await asyncio.sleep(0.5)
                continue

            if state["is_streaming"]:
                # Ambil JPEG bytes langsung dari buffer kamera (lebih efisien, skip re-encode)
                jpeg_bytes = camera_core.capture_jpeg_bytes()

                # Kamera offline, atau kamera belum menghasilkan frame baru: tidak ada
                # yang dikirim. Frame duplikat membuat FPS dan latensi terukur lebih baik
                # dari kenyataan, dan membuang bandwidth.
                frame_time = camera_core.last_frame_time_ns
                if jpeg_bytes is None or frame_time == state.get("last_frame_time_ns"):
                    await asyncio.sleep(0.005)
                    continue
                state["last_frame_time_ns"] = frame_time

                # Tulis ke video file jika sedang recording (perlu decode ke numpy)
                if state["is_recording"] and jpeg_bytes is not None:
                    frame = camera_core.capture_frame()
                    if frame is not None:
                        if state["recording_pending_init"]:
                            h, w = frame.shape[:2]
                            fps_w = max(1.0, float(state["target_fps"]))
                            path = state["recording_filepath"]
                            # Coba beberapa codec — 'vp09' sering TIDAK tersedia di
                            # OpenCV Jetson. mp4v (.mp4) & MJPG (.avi) hampir selalu ada.
                            attempts = [
                                (path.rsplit(".", 1)[0] + ".mp4", "mp4v"),
                                (path.rsplit(".", 1)[0] + ".avi", "MJPG"),
                                (path, "vp09"),
                            ]
                            vw = None
                            for cand_path, cc in attempts:
                                try:
                                    test = cv2.VideoWriter(cand_path, cv2.VideoWriter_fourcc(*cc), fps_w, (w, h))
                                    if test.isOpened():
                                        vw = test
                                        state["recording_filepath"] = cand_path
                                        state["recording_final_filepath"] = state["recording_final_filepath"].rsplit(".", 1)[0] + "." + cand_path.rsplit(".", 1)[1]
                                        print(f"[BRIDGE] VideoWriter OK ({cc}): {cand_path} ({w}x{h} @ {fps_w}fps)")
                                        break
                                    test.release()
                                except Exception as e:
                                    print(f"[BRIDGE] VideoWriter {cc} gagal: {e}")
                            state["video_writer"] = vw
                            state["recording_pending_init"] = False
                            if vw is None:
                                print("[BRIDGE ERROR] Semua codec VideoWriter gagal — rekaman tidak akan tersimpan.")
                        if state["video_writer"] is not None:
                            state["video_writer"].write(frame)

                # Kirim JPEG ke VPS via WebSocket (base64)
                if jpeg_bytes is not None:
                    base64_str = base64.b64encode(jpeg_bytes).decode('utf-8')
                    state["frame_id"] = state.get("frame_id", 0) + 1
                    async with ws_lock:
                        await websocket.send(json.dumps({
                            "event": "STREAM_DATA",
                            "image": f"data:image/jpeg;base64,{base64_str}",
                            "t_cap": frame_time,
                            "frame_id": state["frame_id"],
                        }))
            else:
                # Tidak streaming, tapi jika masih recording karena alasan apapun, jaga frame tetap masuk
                if state["is_recording"] and not state["recording_pending_init"] and state["video_writer"] is not None:
                    frame = camera_core.capture_frame()
                    if frame is not None:
                        state["video_writer"].write(frame)
                await asyncio.sleep(0.2)

        except Exception as e:
            print(f"[BRIDGE STREAM ERROR] Gagal mengirim stream: {e}")
            break

        # FPS Limiter Calculation
        elapsed = time.time() - start_t
        ideal_delay = 1.0 / state["target_fps"]
        sleep_time = max(0.01, ideal_delay - elapsed)
        await asyncio.sleep(sleep_time)

async def telemetry_sender(websocket):
    """TASK 3: Mengirim data telemetri posisi nyata XYZ ke VPS untuk disiarkan ke Redis Pub/Sub."""
    while True:
        try:
            status_data = await asyncio.to_thread(motor_core.get_status)
            mem_stats = get_jetson_memory_stats()
            temps = get_jetson_temperatures()
            cpu_usage = psutil.cpu_percent(interval=None)
            if test_logger.enabled():
                used, _, total = mem_stats["ram"].replace(" GB", "").partition("/")
                test_logger.log("telemetry", t_ns=time.time_ns(), status=status_data["status"],
                                x=status_data["X"], y=status_data["Y"], z=status_data["Z"], cpu=cpu_usage,
                                cpu_temp=temps.get("cpu"), gpu_temp=temps.get("gpu"),
                                ram_used_gb=used, ram_total_gb=total)
            async with ws_lock:
                await websocket.send(json.dumps({
                    "event": "TELEMETRY_DATA",
                    "status": status_data["status"],
                    "limit_switch": status_data.get("limit_switch", "N/A"),
                    "jetson_temperatures": temps,
                    "edge_cpu_usage": cpu_usage,
                    "ram_usage": mem_stats["ram"],
                    "rom_usage": mem_stats["rom"],
                    "position": {
                        "X": status_data["X"],
                        "Y": status_data["Y"],
                        "Z": status_data["Z"]
                    },
                    "soft_limits_enabled": motor_core.soft_limits_enabled,
                    "soft_limits": motor_core.soft_limits,
                    "at_limit": motor_core.limit_flags(),
                    "z_units_per_step": motor_core.z_units_per_step,
                }))
        except Exception:
            break
        # Saat stitching, kirim telemetri lebih jarang (hemat CPU/GIL untuk
        # subprocess); tetap dikirim supaya web tahu Jetson masih hidup.
        if edge_state.stitching:
            await asyncio.sleep(1.0)
        elif test_logger.enabled():
            # Saat uji, interval dirapatkan supaya waktu berhenti motor terukur halus.
            await asyncio.sleep(float(os.getenv("TEST_TELEMETRY_INTERVAL", "0.05")))
        else:
            await asyncio.sleep(0.2)

async def hardware_control_loop():
    print("[EDGE CONNECTOR] Menginisialisasi koneksi hardware...")
    motor_core.open()
    motor_core.unlock()
    camera_core.open()
    
    print(f"[EDGE CONNECTOR] Menghubungkan ke VPS: {VPS_WS_URL}")
    
    while True:
        try:
            link = LinkState()
            link_was_lost = _link_lost_since_connect
            async with websockets.connect(_edge_ws_url(), max_size=None,
                                          ping_interval=WS_PING_INTERVAL,
                                          ping_timeout=WS_PING_TIMEOUT) as websocket:
                print("[EDGE SUCCESS] Pipa jaringan asinkron ganda terhubung penuh!")
                if link_was_lost:
                    _log_watchdog("LINK_RESTORED", 0.0)
                    _set_link_ok()
                async with ws_lock:
                    await websocket.send(json.dumps({"status": "READY", "device": "JETSON_ORIN_NANO"}))

                commands = asyncio.Queue()
                helpers = [
                    asyncio.create_task(command_worker(websocket, commands)),
                    asyncio.create_task(send_stream_task(websocket, state)),
                    asyncio.create_task(telemetry_sender(websocket)),
                    asyncio.create_task(watchdog_task(link)),
                ]
                try:
                    await receive_handler(websocket, link, commands)
                finally:
                    # Link habis: hentikan motor dulu, lalu buang perintah yang belum jalan.
                    _hold_motor(link, "WATCHDOG_TRIGGERED")
                    for task in helpers:
                        task.cancel()
                    await asyncio.gather(*helpers, return_exceptions=True)
            # Penutupan normal dari server (1000/1001) tidak melempar exception.
            print("[EDGE ERROR] VPS menutup sambungan. Menghubungkan ulang dalam 5 detik...")
            await asyncio.sleep(5)
        except websockets.exceptions.ConnectionClosed as e:
            if e.rcvd is not None and e.rcvd.code == 4003:
                print(f"[EDGE ERROR] Device token ditolak VPS. Mencoba lagi dalam {EDGE_TOKEN_REJECTED_BACKOFF_S} detik.")
                await asyncio.sleep(EDGE_TOKEN_REJECTED_BACKOFF_S)
            else:
                print("[EDGE ERROR] Sambungan terputus. Menghubungkan ulang dalam 5 detik...")
                await asyncio.sleep(5)
        except ConnectionRefusedError:
            print("[EDGE ERROR] Sambungan terputus. Menghubungkan ulang dalam 5 detik...")
            await asyncio.sleep(5)
        except Exception as e:
            print(f"[EDGE CRITICAL] Kegagalan sistem: {e}")
            await asyncio.sleep(5)

if __name__ == "__main__":
    try:
        asyncio.run(hardware_control_loop())
    except KeyboardInterrupt:
        print("[EDGE] Mematikan jembatan data secara bersih...")
        camera_core.close()
        motor_core.close()