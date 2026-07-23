import asyncio
import json
import websockets
import base64
import os
import glob

# Impor driver inti dan modul AI dari folder Hardware
from Hardware.Devices.motor import motor_core
from Hardware.Devices.camera import camera_core
from Hardware.Computer_Vision.tile_stitching import dl_stitcher
from Hardware.Computer_Vision.colony_counter import colony_counter
from Hardware.Computer_Vision.classic_cv_edit import classic_cv_editor

# Konfigurasi Koneksi VPS & Direktori Edge
VPS_WS_URL = os.getenv("VPS_WS_URL", "ws://127.0.0.1:8000/api/hardware/ws")
LOCAL_TMP_DIR = "./tmp_images"
if not os.path.exists(LOCAL_TMP_DIR):
    os.makedirs(LOCAL_TMP_DIR)

# State global untuk mengontrol live streaming video dari jarak jauh
is_streaming = False


def get_jetson_temperature():
    temp_paths = [
        "/sys/class/thermal/thermal_zone0/temp",
        "/sys/devices/virtual/thermal/thermal_zone0/temp",
    ]
    for path in temp_paths:
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    raw = f.read().strip()
                    if raw:
                        return round(int(raw) / 1000.0, 1)
        except Exception:
            continue
    return None

async def receive_handler(websocket):
    """TASK 1: Fokus mendengarkan instruksi masuk dari VPS secara asinkron."""
    global is_streaming
    async for message in websocket:
        try:
            data = json.loads(message)
            action = data.get("action", "").upper()
            
            if action == "MOVE_MOTOR":
                gcode = data.get("gcode", "")
                print(f"[BRIDGE] Eksekusi G-Code: {gcode}")
                grbl_resp = motor_core.jog_from_gcode(gcode)
                
                await websocket.send(json.dumps({
                    "event": "MOTOR_MOVED",
                    "status": "SUCCESS",
                    "grbl_response": grbl_resp
                }))

            elif action == "HOMING":
                print("[BRIDGE] Perintah VPS: Homing motor.")
                grbl_resp = motor_core.homing()
                await websocket.send(json.dumps({
                    "event": "MOTOR_MOVED",
                    "status": "SUCCESS",
                    "grbl_response": grbl_resp
                }))

            elif action == "UNLOCK":
                print("[BRIDGE] Perintah VPS: Unlock motor.")
                grbl_resp = motor_core.unlock()
                await websocket.send(json.dumps({
                    "event": "MOTOR_MOVED",
                    "status": "SUCCESS",
                    "grbl_response": grbl_resp
                }))

            elif action == "APPLY_CAMERA_SETTINGS":
                print(
                    f"[BRIDGE CAMERA] Settings diterima: shutter={data.get('shutter_speed')} iso={data.get('iso')}"
                )
                camera_core.apply_settings(
                    shutter_speed=data.get("shutter_speed"),
                    iso=data.get("iso"),
                )

            elif action == "APPLY_CNC_SETTINGS":
                print(
                    f"[BRIDGE CNC] Settings diterima: feed={data.get('feed_rate')} backlash={data.get('backlash')} accel={data.get('acceleration')} settle={data.get('settle_time')}"
                )
                motor_core.apply_motion_settings(
                    feed_rate=data.get("feed_rate"),
                    backlash=data.get("backlash"),
                    acceleration=data.get("acceleration"),
                    settle_time=data.get("settle_time"),
                )

            elif action == "START_STREAM":
                print("[BRIDGE] Perintah VPS: Aktifkan Live Stream.")
                is_streaming = True

            elif action == "STOP_STREAM":
                print("[BRIDGE] Perintah VPS: Matikan Live Stream.")
                is_streaming = False

            elif action == "CAPTURE_IMAGE":
                filename = data.get("filename", "IMG_0000.jpg")
                img_path = os.path.join(LOCAL_TMP_DIR, filename)
                print(f"[BRIDGE CAMERA] Menangkap gambar resolusi tinggi: {filename}")
                
                # Menangkap frame kualitas maksimum dari sensor Edge secara sinkron
                frame_bytes = camera_core.capture_to_bytes(quality=100)
                if not frame_bytes or len(frame_bytes) == 0:
                    print(f"[BRIDGE CAMERA WARNING] Kamera fisik offline. Membangkitkan gambar mockup {filename}...")
                    import cv2
                    import numpy as np
                    mock_img = np.zeros((480, 640, 3), dtype=np.uint8)
                    mock_img[:] = (50, 50, 150) # warna latar biru
                    cv2.putText(mock_img, f"MOCK {filename}", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                    cv2.imwrite(img_path, mock_img)
                    with open(img_path, "rb") as f:
                        frame_bytes = f.read()
                else:
                    with open(img_path, "wb") as f:
                        f.write(frame_bytes)
                    print(f"[BRIDGE CAMERA] Tersimpan: {img_path}")
                
                # UPLOAD KE VPS SUPAYA BISA DILIHAT DI BROWSER
                encoded_image = base64.b64encode(frame_bytes).decode('utf-8')
                await websocket.send(json.dumps({
                    "event": "IMAGE_CAPTURED",
                    "filename": filename,
                    "image_data": f"data:image/jpeg;base64,{encoded_image}"
                }))

            elif action == "START_STITCHING":
                print("[BRIDGE AI] Menjalankan Deep Learning Tile Stitching...")
                
                # FIX: Membaca gambar dari storage LOKAL Jetson, bukan folder VPS!
                # Prioritaskan daftar gambar spesifik dari VPS agar tidak terjadi kontaminasi sesi
                target_images = data.get("images", [])
                if target_images:
                    captured_tiles = [os.path.join(LOCAL_TMP_DIR, img) for img in target_images]
                else:
                    captured_tiles = sorted(glob.glob(os.path.join(LOCAL_TMP_DIR, "IMG_*.jpg")))
                
                # Jalankan penjahitan gambar berbasis Deep Learning
                # Output sementara disimpan di lokal sebelum dikirim ke server
                local_output = os.path.join(LOCAL_TMP_DIR, "stitched_ta_output.jpg")
                output_file = dl_stitcher.stitch_with_deep_learning(captured_tiles, local_output)
                
                if output_file and os.path.exists(output_file):
                    # Konversi hasil jahitan ke Base64 agar bisa diunggah langsung ke static folder VPS lewat WebSocket
                    with open(output_file, "rb") as img_file:
                        encoded_stitched = base64.b64encode(img_file.read()).decode('utf-8')

                    await websocket.send(json.dumps({
                        "event": "STITCHING_COMPLETE",
                        "status": "SUCCESS",
                        "image_data": f"data:image/jpeg;base64,{encoded_stitched}",
                        "filename": "stitched_ta_output.jpg"
                    }))
                else:
                    await websocket.send(json.dumps({"event": "STITCHING_FAILED", "status": "ERROR"}))

            elif action == "START_DL_COUNT":
                print("[BRIDGE AI] Menjalankan Deep Learning Colony Counter (YOLO)...")
                # FIX: Membaca gambar hasil stitching yang tersimpan di lokal Jetson
                target_img = os.path.join(LOCAL_TMP_DIR, "stitched_ta_output.jpg")
                
                result_img, count_result = colony_counter.analyze_image(target_img)
                
                # Baca gambar hasil deteksi bounding box YOLO untuk diumpan balik ke VPS
                local_predicted_path = os.path.join(LOCAL_TMP_DIR, result_img)
                with open(local_predicted_path, "rb") as img_file:
                    encoded_predicted = base64.b64encode(img_file.read()).decode('utf-8')

                await websocket.send(json.dumps({
                    "event": "COUNTING_COMPLETE",
                    "status": "SUCCESS",
                    "total_cells": count_result,
                    "image_data": f"data:image/jpeg;base64,{encoded_predicted}",
                    "filename": result_img
                }))

            elif action == "APPLY_IMAGE_EDIT":
                edit_type = data.get("type", "")
                target_img = os.path.join(LOCAL_TMP_DIR, "stitched_ta_output.jpg")
                print(f"[BRIDGE CV] Menerapkan filter edit klasik: {edit_type}")
                
                if edit_type == "THRESHOLD":
                    res_file = classic_cv_editor.apply_adaptive_threshold(target_img)
                elif edit_type == "BRIGHTNESS":
                    b_val = data.get("brightness", 0)
                    c_val = data.get("contrast", 0)
                    res_file = classic_cv_editor.apply_brightness_contrast(target_img, b_val, c_val)
                
                local_edited_path = os.path.join(LOCAL_TMP_DIR, res_file)
                with open(local_edited_path, "rb") as img_file:
                    encoded_edited = base64.b64encode(img_file.read()).decode('utf-8')

                await websocket.send(json.dumps({
                    "event": "EDIT_COMPLETE",
                    "status": "SUCCESS",
                    "image_data": f"data:image/jpeg;base64,{encoded_edited}",
                    "filename": res_file
                }))
            
            elif action == "CV_COLONY_COUNT":
                filename = data.get("filename")
                print(f"[BRIDGE AI] Analysis API: YOLO Colony Counter on {filename}")
                input_path = os.path.abspath(os.path.join("../Software/backend/static/uploads", filename))
                if os.path.exists(input_path):
                    result_img, count_result = colony_counter.analyze_image(input_path)
                    json_output = os.path.join("../Software/backend/static/uploads", result_img.replace('.jpg', '.json').replace('.png', '.json'))
                    with open(json_output, 'w') as f:
                        json.dump({"colony_count": count_result}, f)
                    print(f"[BRIDGE AI] Selesai. Hasil: {count_result} koloni.")

            elif action == "CV_THRESHOLD":
                filename = data.get("filename")
                print(f"[BRIDGE CV] Analysis API: Adaptive Threshold on {filename}")
                input_path = os.path.abspath(os.path.join("../Software/backend/static/uploads", filename))
                if os.path.exists(input_path):
                    res_file = classic_cv_editor.apply_adaptive_threshold(input_path)
                    print(f"[BRIDGE CV] Selesai. Output: {res_file}")
                else:
                    print(f"[BRIDGE CV ERROR] File tidak ditemukan di: {input_path}")

            elif action == "CV_CONTOUR":
                filename = data.get("filename")
                print(f"[BRIDGE CV] Analysis API: Ekstraksi Kontur on {filename}")
                input_path = os.path.abspath(os.path.join("../Software/backend/static/uploads", filename))
                if os.path.exists(input_path):
                    res_file = classic_cv_editor.extract_and_draw_contours(input_path)
                    print(f"[BRIDGE CV] Selesai. Output: {res_file}")
                else:
                    print(f"[BRIDGE CV ERROR] File tidak ditemukan di: {input_path}")

            elif action == "CV_MORPHOLOGY":
                filename = data.get("filename")
                print(f"[BRIDGE CV] Analysis API: Kalkulasi Morfologi on {filename}")
                input_path = os.path.abspath(os.path.join("../Software/backend/static/uploads", filename))
                if os.path.exists(input_path):
                    res_file, stats = classic_cv_editor.calculate_morphology(input_path)
                    json_output = os.path.join("../Software/backend/static/uploads", res_file.replace('.jpg', '.json').replace('.png', '.json'))
                    with open(json_output, 'w') as f:
                        json.dump(stats, f)
                    print(f"[BRIDGE CV] Selesai. Output: {res_file}")
                else:
                    print(f"[BRIDGE CV ERROR] File tidak ditemukan di: {input_path}")

            elif action == "CV_SOBEL":
                filename = data.get("filename")
                print(f"[BRIDGE CV] Analysis API: Sobel Edge Detection on {filename}")
                input_path = os.path.abspath(os.path.join("../Software/backend/static/uploads", filename))
                if os.path.exists(input_path):
                    res_file = classic_cv_editor.apply_sobel_edge(input_path)
                    print(f"[BRIDGE CV] Selesai. Output: {res_file}")
                else:
                    print(f"[BRIDGE CV ERROR] File tidak ditemukan di: {input_path}")

            elif action == "CV_ROI":
                filename = data.get("filename")
                print(f"[BRIDGE CV] Analysis API: ROI Selection on {filename}")
                input_path = os.path.abspath(os.path.join("../Software/backend/static/uploads", filename))
                if os.path.exists(input_path):
                    res_file = classic_cv_editor.auto_roi_crop(input_path)
                    print(f"[BRIDGE CV] Selesai. Output: {res_file}")
                else:
                    print(f"[BRIDGE CV ERROR] File tidak ditemukan di: {input_path}")

            elif action == "CV_CALIBRATE":
                filename = data.get("filename")
                print(f"[BRIDGE CV] Analysis API: Scale Calibration on {filename}")
                input_path = os.path.abspath(os.path.join("../Software/backend/static/uploads", filename))
                if os.path.exists(input_path):
                    res_file = classic_cv_editor.draw_scale_calibration(input_path)
                    print(f"[BRIDGE CV] Selesai. Output: {res_file}")
                else:
                    print(f"[BRIDGE CV ERROR] File tidak ditemukan di: {input_path}")

            elif action == "CV_COLOR_SPLIT":
                filename = data.get("filename")
                print(f"[BRIDGE CV] Analysis API: Color Split on {filename}")
                input_path = os.path.abspath(os.path.join("../Software/backend/static/uploads", filename))
                if os.path.exists(input_path):
                    res_file = classic_cv_editor.split_color_channels(input_path)
                    print(f"[BRIDGE CV] Selesai. Output: {res_file}")
                else:
                    print(f"[BRIDGE CV ERROR] File tidak ditemukan di: {input_path}")

        except Exception as e:
            print(f"[BRIDGE ERROR] Gagal memproses data handler: {e}")

async def stream_sender(websocket):
    """TASK 2: Mengirim data stream video JPEG secara paralel tanpa mengganggu task lain."""
    global is_streaming
    print("[BRIDGE] Task pemancar video stream aktif di background loop.")
    while True:
        try:
            if is_streaming:
                frame_bytes = camera_core.capture_to_bytes(quality=70)
                
                # Pastikan frame_bytes benar-benar ada dan valid
                if frame_bytes and len(frame_bytes) > 0:
                    base64_str = base64.b64encode(frame_bytes).decode('utf-8')
                    await websocket.send(json.dumps({
                        "event": "STREAM_DATA",
                        "image": f"data:image/jpeg;base64,{base64_str}"
                    }))
                else:
                    # 🟢 FIX: Jika sensor kamera sedang warm-up/kosong, beri jeda asinkron ringan 
                    # lalu biarkan loop selesai mengalir ke bawah agar tidak memblokir task lain
                    print("[BRIDGE WARNING] Frame kamera kosong, menunggu sensor siap...")
                    await asyncio.sleep(0.1)
            else:
                # Jika status transmisi mati, beri jeda tidur yang cukup agar CPU laptop tidak overload
                await asyncio.sleep(0.2)
                
        except Exception as e:
            print(f"[BRIDGE STREAM ERROR] Gagal mengirim stream: {e}")
            break
        
        # 🟢 JAMINAN UTAMA: Selalu beri jeda tidur asinkron di setiap akhir putaran loop
        await asyncio.sleep(0.066)

async def telemetry_sender(websocket):
    """TASK 3: Mengirim data telemetri posisi nyata XYZ ke VPS untuk disiarkan ke Redis Pub/Sub."""
    while True:
        try:
            status_data = motor_core.get_status()
            await websocket.send(json.dumps({
                "event": "TELEMETRY_DATA",
                "status": status_data["status"],
                "limit_switch": status_data.get("limit_switch", "N/A"),
                "jetson_temp_c": get_jetson_temperature(),
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
            async with websockets.connect(VPS_WS_URL) as websocket:
                print("[EDGE SUCCESS] Pipa jaringan asinkron ganda terhubung penuh!")
                await websocket.send(json.dumps({"status": "READY", "device": "JETSON_ORIN_NANO"}))
                
                await asyncio.gather(
                    receive_handler(websocket),
                    stream_sender(websocket),
                    telemetry_sender(websocket)
                )
        except (websockets.exceptions.ConnectionClosed, ConnectionRefusedError):
            print("[EDGE ERROR] Sambungan terput as. Menghubungkan ulang dalam 5 detik...")
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