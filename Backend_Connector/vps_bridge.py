import asyncio
import json
import websockets
import base64
import os
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
            
            if action in ["MOVE_MOTOR", "HOMING", "UNLOCK", "APPLY_CNC_SETTINGS"]:
                await handle_motor_action(action, data, websocket, ws_lock)
            elif action in ["APPLY_CAMERA_SETTINGS", "START_STREAM", "STOP_STREAM", "SET_FPS", "START_RECORDING", "STOP_RECORDING", "CAPTURE_IMAGE", "PURGE_CACHE"]:
                await handle_camera_action(action, data, websocket, ws_lock, state)
            elif action in ["START_STITCHING", "START_DL_COUNT", "APPLY_IMAGE_EDIT", "CV_COLONY_COUNT", "CV_THRESHOLD", "CV_CONTOUR", "CV_MORPHOLOGY", "CV_SOBEL", "CV_ROI", "CV_CALIBRATE", "CV_COLOR_SPLIT"]:
                await handle_ai_action(action, data, websocket, ws_lock)
            else:
                print(f"[BRIDGE WARNING] Action tidak dikenali: {action}")

        except Exception as e:
            print(f"[BRIDGE ERROR] Gagal memproses data handler: {e}")

async def send_stream_task(websocket, state):
    """TASK 2: Membaca frame dari hardware Edge Camera secara asinkron lalu disiarkan ke VPS."""
    import numpy as np
    print("[BRIDGE] Task pemancar video stream aktif di background loop.")
    while True:
        start_t = time.time()
        try:
            if state["is_streaming"]:
                # Capture as raw numpy array for both streaming and recording
                frame = camera_core.capture_frame()
                
                # --- MOCK FALLBACK: Jika kamera offline, buat frame hitam agar stream & recording tetap berjalan ---
                if frame is None:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, "CAMERA OFFLINE", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (80, 80, 80), 3)

                # Tulis ke video file jika sedang recording
                if state["is_recording"]:
                    if state["recording_pending_init"]:
                        h, w = frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'vp09')  # VP9 - browser native (WebM)
                        state["video_writer"] = cv2.VideoWriter(state["recording_filepath"], fourcc, float(state["target_fps"]), (w, h))
                        if state["video_writer"].isOpened():
                            state["recording_pending_init"] = False
                            print(f"[BRIDGE] VideoWriter berhasil diinisialisasi: {state['recording_filepath']} ({w}x{h} @ {state['target_fps']}fps)")
                        else:
                            print(f"[BRIDGE ERROR] VideoWriter gagal dibuka! Path: {state['recording_filepath']}")
                            state["video_writer"] = None
                            
                    if state["video_writer"] is not None:
                        state["video_writer"].write(frame)
                    
                # Encode to JPEG for websocket stream
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ret:
                    frame_bytes = buffer.tobytes()
                    base64_str = base64.b64encode(frame_bytes).decode('utf-8')
                    await websocket.send(json.dumps({
                        "event": "STREAM_DATA",
                        "image": f"data:image/jpeg;base64,{base64_str}"
                    }))
            else:
                # Tidak streaming, tapi jika masih recording karena alasan apapun, jaga frame tetap masuk
                if state["is_recording"] and not state["recording_pending_init"] and state["video_writer"] is not None:
                    frame = camera_core.capture_frame()
                    if frame is None:
                        frame = np.zeros((480, 640, 3), dtype=np.uint8)
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
            status_data = motor_core.get_status()
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
                    }
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