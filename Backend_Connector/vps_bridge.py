import asyncio
import json
import websockets
import base64
import os
import sys
import cv2
import time
import psutil

# Impor driver inti dari folder Hardware
from Hardware.Devices.motor import motor_core
from Hardware.Devices.camera import camera_core

from Hardware.Utils.telemetry import get_jetson_temperatures, get_jetson_memory_stats
from Hardware.Handlers.motor_handler import handle_motor_action
from Hardware.Handlers.camera_handler import handle_camera_action
from Hardware.Handlers.ai_handler import handle_ai_action

# Konfigurasi Koneksi VPS & Direktori Edge
VPS_WS_URL = os.getenv("VPS_WS_URL", "ws://127.0.0.1:8000/api/hardware/ws")

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

async def receive_handler(websocket):
    """TASK 1: Fokus mendengarkan instruksi masuk dari VPS secara asinkron."""
    async for message in websocket:
        try:
            data = json.loads(message)
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
            elif action in ["APPLY_CAMERA_SETTINGS", "START_STREAM", "STOP_STREAM", "SET_FPS", "START_RECORDING", "STOP_RECORDING", "CAPTURE_IMAGE", "PURGE_CACHE", "PURGE_ALL_DATASETS", "PURGE_DATASET_FOLDER", "PURGE_DATASET_FILES"]:
                await handle_camera_action(action, data, websocket, ws_lock, state)
            elif action in ["START_STITCHING", "START_DL_COUNT", "APPLY_IMAGE_EDIT", "CV_COLONY_COUNT", "CV_THRESHOLD", "CV_CONTOUR", "CV_MORPHOLOGY", "CV_SOBEL", "CV_ROI", "CV_CALIBRATE", "CV_COLOR_SPLIT"]:
                await handle_ai_action(action, data, websocket, ws_lock)
            elif action == "PING":
                pass # Abaikan sinyal keep-alive
            else:
                print(f"[BRIDGE WARNING] Action tidak dikenali: {action}")

        except Exception as e:
            print(f"[BRIDGE ERROR] Gagal memproses data handler: {e}")

async def send_stream_task(websocket, state):
    """TASK 2: Membaca frame JPEG dari hardware Edge Camera secara asinkron lalu disiarkan ke VPS."""
    import numpy as np
    print("[BRIDGE] Task pemancar video stream aktif di background loop.")
    while True:
        start_t = time.time()
        try:
            if state["is_streaming"]:
                # Ambil JPEG bytes langsung dari buffer kamera (lebih efisien, skip re-encode)
                jpeg_bytes = camera_core.capture_jpeg_bytes()

                # --- MOCK FALLBACK: Jika kamera offline, buat frame hitam ---
                if jpeg_bytes is None:
                    fallback_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(fallback_frame, "CAMERA OFFLINE", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (80, 80, 80), 3)
                    ret, buf = cv2.imencode('.jpg', fallback_frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                    jpeg_bytes = buf.tobytes() if ret else None

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
                    async with ws_lock:
                        await websocket.send(json.dumps({
                            "event": "STREAM_DATA",
                            "image": f"data:image/jpeg;base64,{base64_str}"
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
            async with ws_lock:
                await websocket.send(json.dumps({
                    "event": "TELEMETRY_DATA",
                    "status": status_data["status"],
                    "limit_switch": status_data.get("limit_switch", "N/A"),
                    "jetson_temperatures": get_jetson_temperatures(),
                    "edge_cpu_usage": psutil.cpu_percent(interval=None),
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
                }))
        except Exception:
            break
        await asyncio.sleep(0.2)

async def hardware_control_loop():
    print("[EDGE CONNECTOR] Menginisialisasi koneksi hardware...")
    motor_core.open()
    motor_core.unlock()
    camera_core.open()
    
    print(f"[EDGE CONNECTOR] Menghubungkan ke VPS: {VPS_WS_URL}")
    
    while True:
        try:
            async with websockets.connect(VPS_WS_URL, max_size=None, ping_interval=None) as websocket:
                print("[EDGE SUCCESS] Pipa jaringan asinkron ganda terhubung penuh!")
                async with ws_lock:
                    await websocket.send(json.dumps({"status": "READY", "device": "JETSON_ORIN_NANO"}))
                
                await asyncio.gather(
                    receive_handler(websocket),
                    send_stream_task(websocket, state),
                    telemetry_sender(websocket)
                )
        except (websockets.exceptions.ConnectionClosed, ConnectionRefusedError):
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