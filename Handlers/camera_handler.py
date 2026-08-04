import os
import json
import base64
import cv2
from datetime import datetime
from Hardware.Devices.camera import camera_core

LOCAL_TMP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tmp_images"))
if not os.path.exists(LOCAL_TMP_DIR):
    os.makedirs(LOCAL_TMP_DIR)

async def handle_camera_action(action: str, data: dict, websocket, ws_lock, state: dict):
    """Router untuk instruksi dan konfigurasi Kamera"""
    if action == "APPLY_CAMERA_SETTINGS":
        print(
            f"[BRIDGE CAMERA] Settings diterima: shutter={data.get('shutter_speed')} iso={data.get('iso')}"
        )
        camera_core.apply_settings(
            shutter_speed=data.get("shutter_speed"),
            iso=data.get("iso"),
        )

    elif action == "START_STREAM":
        print("[BRIDGE] Perintah VPS: Aktifkan Live Stream.")
        state["is_streaming"] = True

    elif action == "STOP_STREAM":
        print("[BRIDGE] Perintah VPS: Matikan Live Stream.")
        state["is_streaming"] = False
        
    elif action == "SET_FPS":
        state["target_fps"] = int(data.get("fps", 30))
        print(f"[BRIDGE] Perintah VPS: Set target FPS stream menjadi {state['target_fps']}")

    elif action == "START_RECORDING":
        if not state["is_recording"]:
            state["is_recording"] = True
            state["is_streaming"] = True  # Pastikan stream aktif agar frame mengalir ke VideoWriter
            folder_id = data.get("folder_id", "unsorted")
            rec_dir = os.path.abspath(os.path.join(".", "local_datasets", folder_id))
            if not os.path.exists(rec_dir):
                os.makedirs(rec_dir)
            filename = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.webm"
            state["recording_final_filepath"] = os.path.join(rec_dir, filename)
            state["recording_filepath"] = os.path.join(rec_dir, f"_temp_{filename}")
            state["recording_pending_init"] = True
            print(f"[BRIDGE] Mulai merekam video ke {state['recording_filepath']}")

    elif action == "STOP_RECORDING":
        if state["is_recording"]:
            state["is_recording"] = False
            state["recording_pending_init"] = False
            if state["video_writer"] is not None:
                state["video_writer"].release()
                print(f"[BRIDGE] VideoWriter dirilis. File sementara: {state['recording_filepath']}")
                state["video_writer"] = None
                if os.path.exists(state["recording_filepath"]):
                    file_size = os.path.getsize(state["recording_filepath"])
                    print(f"[BRIDGE] Ukuran file sementara: {file_size} bytes")
                    if file_size > 0:
                        os.rename(state["recording_filepath"], state["recording_final_filepath"])
                        print(f"[BRIDGE] SUKSES! File final disimpan: {state['recording_final_filepath']}")
                    else:
                        os.remove(state["recording_filepath"])
                        print(f"[BRIDGE ERROR] File sementara 0-byte, dihapus. Recording gagal!")
                else:
                    print(f"[BRIDGE ERROR] File sementara tidak ditemukan: {state['recording_filepath']}")
            else:
                print(f"[BRIDGE ERROR] STOP_RECORDING dipanggil tapi video_writer=None. Tidak ada frame yang direkam!")
            print("[BRIDGE] Perekaman video dihentikan.")

    elif action == "CAPTURE_IMAGE":
        prefix = data.get("prefix", "IMG_MANUAL")
        cx = data.get("x", 0.0)
        cy = data.get("y", 0.0)
        requested_filename = data.get("filename")
        print(f"[BRIDGE] 📸 Mengambil foto manual dari frontend (prefix={prefix}, X={cx}, Y={cy})...")
        
        # Biarkan camera_core yang menentukan nama final dan menyimpannya (jika berhasil hardware)
        filename = camera_core.save_snapshot(LOCAL_TMP_DIR, prefix=prefix, coord_x=cx, coord_y=cy, requested_filename=requested_filename)
        
        if filename:
            file_path = os.path.join(LOCAL_TMP_DIR, filename)
            print(f"[BRIDGE CAMERA] Tersimpan: {file_path}")
        else:
            print(f"[BRIDGE CAMERA WARNING] Kamera fisik offline. Membangkitkan gambar mockup...")
            import numpy as np
            import random
            
            # Buat mockup
            if requested_filename:
                filename = requested_filename
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{prefix}_{timestamp}_X{str(cx).replace('.','_')}_Y{str(cy).replace('.','_')}.jpg"
                
            img_path = os.path.join(LOCAL_TMP_DIR, filename)
            mock_img = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_img[:] = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200)) # warna latar acak
            cv2.putText(mock_img, f"MOCK {filename}", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            cv2.imwrite(img_path, mock_img)
            print(f"[BRIDGE CAMERA] MOCK Tersimpan: {img_path}")
        
        # UPLOAD KE VPS SUPAYA BISA DILIHAT DI BROWSER
        try:
            with open(os.path.join(LOCAL_TMP_DIR, filename), "rb") as f:
                frame_bytes = f.read()
            encoded_image = base64.b64encode(frame_bytes).decode('utf-8')
            async with ws_lock:
                await websocket.send(json.dumps({
                    "event": "IMAGE_CAPTURED",
                    "filename": filename,
                    "image_data": f"data:image/jpeg;base64,{encoded_image}"
                }))
        except Exception as e:
            print(f"[BRIDGE ERROR] Gagal mengirim base64 image ke websocket: {e}")

    elif action == "PURGE_CACHE":
        print("[BRIDGE] 🧹 Menerima perintah PURGE_CACHE dari VPS.")
        if os.path.exists(LOCAL_TMP_DIR):
            cleared = 0
            for file in os.listdir(LOCAL_TMP_DIR):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(LOCAL_TMP_DIR, file)
                    try:
                        os.remove(file_path)
                        cleared += 1
                    except: pass
            print(f"[BRIDGE] ✅ Purge selesai. {cleared} file dihapus dari {LOCAL_TMP_DIR}.")
