import asyncio
import json
import websockets
import base64
import os
import glob
import cv2
import time
from datetime import datetime
import shutil

# Impor driver inti dan modul AI dari folder Hardware
from Hardware.Devices.motor import motor_core
from Hardware.Devices.camera import camera_core
from Hardware.Computer_Vision.tile_stitching import dl_stitcher
from Hardware.Computer_Vision.colony_counter import colony_counter
from Hardware.Computer_Vision.classic_cv_edit import classic_cv_editor

# Konfigurasi Koneksi VPS & Direktori Edge
VPS_WS_URL = os.getenv("VPS_WS_URL", "ws://127.0.0.1:8000/api/hardware/ws")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_TMP_DIR = os.path.join(BASE_DIR, "tmp_images")
if not os.path.exists(LOCAL_TMP_DIR):
    os.makedirs(LOCAL_TMP_DIR)

# Global lock to prevent websocket send collisions between video stream and large image payloads
ws_lock = asyncio.Lock()

# State global
is_streaming = False
is_recording = False
recording_pending_init = False
video_writer = None
target_fps = 30
recording_filepath = ""
recording_final_filepath = ""

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

async def receive_handler(websocket):
    """TASK 1: Fokus mendengarkan instruksi masuk dari VPS secara asinkron."""
    global is_streaming, target_fps, is_recording, video_writer, recording_pending_init
    global recording_filepath, recording_final_filepath
    async for message in websocket:
        try:
            data = json.loads(message)
            action = data.get("action", "").upper()
            
            if action == "MOVE_MOTOR":
                gcode = data.get("gcode", "")
                print(f"[BRIDGE] Eksekusi G-Code: {gcode}")
                grbl_resp = motor_core.jog_from_gcode(gcode)
                
                async with ws_lock:
                    await websocket.send(json.dumps({
                        "event": "MOTOR_MOVED",
                        "status": "SUCCESS",
                        "grbl_response": grbl_resp
                    }))

            elif action == "HOMING":
                print("[BRIDGE] Perintah VPS: Homing motor.")
                grbl_resp = motor_core.homing()
                async with ws_lock:
                    await websocket.send(json.dumps({
                        "event": "MOTOR_MOVED",
                        "status": "SUCCESS",
                        "grbl_response": grbl_resp
                    }))

            elif action == "UNLOCK":
                print("[BRIDGE] Perintah VPS: Unlock motor.")
                grbl_resp = motor_core.unlock()
                async with ws_lock:
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
                
            elif action == "SET_FPS":
                target_fps = int(data.get("fps", 30))
                print(f"[BRIDGE] Perintah VPS: Set target FPS stream menjadi {target_fps}")

            elif action == "START_RECORDING":
                if not is_recording:
                    is_recording = True
                    is_streaming = True  # Pastikan stream aktif agar frame mengalir ke VideoWriter
                    folder_id = data.get("folder_id", "unsorted")
                    rec_dir = os.path.abspath(os.path.join(".", "local_datasets", folder_id))
                    if not os.path.exists(rec_dir):
                        os.makedirs(rec_dir)
                    filename = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.webm"
                    recording_final_filepath = os.path.join(rec_dir, filename)
                    recording_filepath = os.path.join(rec_dir, f"_temp_{filename}")
                    # Use mp4v codec for MP4 so it plays natively in web browsers
                    recording_pending_init = True
                    print(f"[BRIDGE] Mulai merekam video ke {recording_filepath}")

            elif action == "STOP_RECORDING":
                if is_recording:
                    is_recording = False
                    recording_pending_init = False
                    if video_writer is not None:
                        video_writer.release()
                        print(f"[BRIDGE] VideoWriter dirilis. File sementara: {recording_filepath}")
                        video_writer = None
                        if os.path.exists(recording_filepath):
                            file_size = os.path.getsize(recording_filepath)
                            print(f"[BRIDGE] Ukuran file sementara: {file_size} bytes")
                            if file_size > 0:
                                os.rename(recording_filepath, recording_final_filepath)
                                print(f"[BRIDGE] SUKSES! File final disimpan: {recording_final_filepath}")
                            else:
                                os.remove(recording_filepath)
                                print(f"[BRIDGE ERROR] File sementara 0-byte, dihapus. Recording gagal!")
                        else:
                            print(f"[BRIDGE ERROR] File sementara tidak ditemukan: {recording_filepath}")
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
                    # Sync ditangani oleh block di bawah melalui websocket
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

                    async with ws_lock:
                        await websocket.send(json.dumps({
                            "event": "STITCHING_COMPLETE",
                            "status": "SUCCESS",
                            "image_data": f"data:image/jpeg;base64,{encoded_stitched}",
                            "filename": "stitched_ta_output.jpg"
                        }))
                else:
                    print("[BRIDGE ERROR] Stitching gagal.")
                    async with ws_lock:
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

                async with ws_lock:
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

                async with ws_lock:
                    await websocket.send(json.dumps({
                        "event": "EDIT_COMPLETE",
                        "status": "SUCCESS",
                        "image_data": f"data:image/jpeg;base64,{encoded_edited}",
                        "filename": res_file
                    }))
            
            elif action == "CV_COLONY_COUNT":
                filename = data.get("filename")
                print("\n" + "="*60)
                print(f"[BRIDGE AI] 🧬 FITUR AKTIF: YOLO Colony Counter")
                print(f"[BRIDGE AI] ⏳ Memuat bobot neural network YOLO...")
                print(f"[BRIDGE AI] 🔍 Menganalisis citra: {filename}")
                input_path = os.path.abspath(os.path.join("../Software/backend/static/uploads", filename))
                if os.path.exists(input_path):
                    result_img, count_result = colony_counter.analyze_image(input_path)
                    json_output = os.path.join("../Software/backend/static/uploads", result_img.replace('.jpg', '.json').replace('.png', '.json'))
                    with open(json_output, 'w') as f:
                        json.dump({"colony_count": count_result}, f)
                    print(f"[BRIDGE AI] ✅ Selesai. Hasil deteksi: {count_result} koloni.")
                    print("="*60 + "\n")

            elif action == "CV_THRESHOLD":
                filename = data.get("filename")
                print("\n" + "-"*50)
                print(f"[BRIDGE CV] 🌗 FITUR AKTIF: Adaptive Threshold")
                print(f"[BRIDGE CV] 🧮 Menghitung nilai biner pada {filename}...")
                input_path = os.path.abspath(os.path.join("../Software/backend/static/uploads", filename))
                if os.path.exists(input_path):
                    res_file = classic_cv_editor.apply_adaptive_threshold(input_path)
                    print(f"[BRIDGE CV] ✅ Binarisasi Selesai. Output: {res_file}")
                    print("-" * 50 + "\n")
                else:
                    print(f"[BRIDGE CV ERROR] File tidak ditemukan di: {input_path}")

            elif action == "CV_CONTOUR":
                filename = data.get("filename")
                print("\n" + "-"*50)
                print(f"[BRIDGE CV] 🦠 FITUR AKTIF: Ekstraksi Kontur")
                print(f"[BRIDGE CV] 📐 Mencari dinding sel geometri pada {filename}...")
                input_path = os.path.abspath(os.path.join("../Software/backend/static/uploads", filename))
                if os.path.exists(input_path):
                    res_file = classic_cv_editor.extract_and_draw_contours(input_path)
                    print(f"[BRIDGE CV] ✅ Penggambaran Kontur Selesai. Output: {res_file}")
                    print("-" * 50 + "\n")
                else:
                    print(f"[BRIDGE CV ERROR] File tidak ditemukan di: {input_path}")

            elif action == "CV_MORPHOLOGY":
                filename = data.get("filename")
                print("\n" + "-"*50)
                print(f"[BRIDGE CV] 📏 FITUR AKTIF: Kalkulasi Morfologi")
                print(f"[BRIDGE CV] 📊 Mengekstrak area dan keliling dari {filename}...")
                input_path = os.path.abspath(os.path.join("../Software/backend/static/uploads", filename))
                if os.path.exists(input_path):
                    res_file, stats = classic_cv_editor.calculate_morphology(input_path)
                    json_output = os.path.join("../Software/backend/static/uploads", res_file.replace('.jpg', '.json').replace('.png', '.json'))
                    with open(json_output, 'w') as f:
                        json.dump(stats, f)
                    print(f"[BRIDGE CV] ✅ Kalkulasi Morfologi Selesai. Output: {res_file}")
                    print("-" * 50 + "\n")
                else:
                    print(f"[BRIDGE CV ERROR] File tidak ditemukan di: {input_path}")

            elif action == "CV_SOBEL":
                filename = data.get("filename")
                print("\n" + "-"*50)
                print(f"[BRIDGE CV] 🔪 FITUR AKTIF: Sobel Edge Detection")
                print(f"[BRIDGE CV] 🧮 Mengekstrak garis tepi konvolusi pada {filename}...")
                input_path = os.path.abspath(os.path.join("../Software/backend/static/uploads", filename))
                if os.path.exists(input_path):
                    res_file = classic_cv_editor.apply_sobel_edge(input_path)
                    print(f"[BRIDGE CV] ✅ Deteksi Tepi Selesai. Output: {res_file}")
                    print("-" * 50 + "\n")
                else:
                    print(f"[BRIDGE CV ERROR] File tidak ditemukan di: {input_path}")

            elif action == "CV_ROI":
                filename = data.get("filename")
                print("\n" + "-"*50)
                print(f"[BRIDGE CV] ✂️ FITUR AKTIF: ROI Selection")
                print(f"[BRIDGE CV] 📍 Memotong area spesifik citra {filename}...")
                input_path = os.path.abspath(os.path.join("../Software/backend/static/uploads", filename))
                if os.path.exists(input_path):
                    res_file = classic_cv_editor.auto_roi_crop(input_path)
                    print(f"[BRIDGE CV] ✅ Pemotongan Selesai. Output: {res_file}")
                    print("-" * 50 + "\n")
                else:
                    print(f"[BRIDGE CV ERROR] File tidak ditemukan di: {input_path}")

            elif action == "CV_CALIBRATE":
                filename = data.get("filename")
                print("\n" + "-"*50)
                print(f"[BRIDGE CV] 🔬 FITUR AKTIF: Scale Calibration")
                print(f"[BRIDGE CV] 📐 Menerapkan matriks kalibrasi lensa objektif pada {filename}...")
                input_path = os.path.abspath(os.path.join("../Software/backend/static/uploads", filename))
                if os.path.exists(input_path):
                    res_file = classic_cv_editor.draw_scale_calibration(input_path)
                    print(f"[BRIDGE CV] ✅ Kalibrasi Selesai. Output: {res_file}")
                    print("-" * 50 + "\n")
                else:
                    print(f"[BRIDGE CV ERROR] File tidak ditemukan di: {input_path}")

            elif action == "CV_COLOR_SPLIT":
                filename = data.get("filename")
                print("\n" + "-"*50)
                print(f"[BRIDGE CV] 🎨 FITUR AKTIF: Color Channel Split")
                print(f"[BRIDGE CV] 🧪 Memisahkan warna stain RGB spesifik pada {filename}...")
                input_path = os.path.abspath(os.path.join("../Software/backend/static/uploads", filename))
                if os.path.exists(input_path):
                    res_file = classic_cv_editor.split_color_channels(input_path)
                    print(f"[BRIDGE CV] ✅ Pemisahan Warna Selesai. Output: {res_file}")
                    print("-" * 50 + "\n")
                else:
                    print(f"[BRIDGE CV ERROR] File tidak ditemukan di: {input_path}")

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

        except Exception as e:
            print(f"[BRIDGE ERROR] Gagal memproses data handler: {e}")

async def send_stream_task(websocket):
    """TASK 2: Membaca frame dari hardware Edge Camera secara asinkron lalu disiarkan ke VPS."""
    global is_streaming, is_recording, video_writer, target_fps, recording_pending_init, recording_filepath
    import numpy as np
    print("[BRIDGE] Task pemancar video stream aktif di background loop.")
    while True:
        start_t = time.time()
        try:
            if is_streaming:
                # Capture as raw numpy array for both streaming and recording
                frame = camera_core.capture_frame()
                
                # --- MOCK FALLBACK: Jika kamera offline, buat frame hitam agar stream & recording tetap berjalan ---
                if frame is None:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, "CAMERA OFFLINE", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (80, 80, 80), 3)

                # Tulis ke video file jika sedang recording
                if is_recording:
                    if recording_pending_init:
                        h, w = frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'vp09')  # VP9 - browser native (WebM)
                        video_writer = cv2.VideoWriter(recording_filepath, fourcc, float(target_fps), (w, h))
                        if video_writer.isOpened():
                            recording_pending_init = False
                            print(f"[BRIDGE] VideoWriter berhasil diinisialisasi: {recording_filepath} ({w}x{h} @ {target_fps}fps)")
                        else:
                            print(f"[BRIDGE ERROR] VideoWriter gagal dibuka! Path: {recording_filepath}")
                            video_writer = None
                            
                    if video_writer is not None:
                        video_writer.write(frame)
                    
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
                if is_recording and not recording_pending_init and video_writer is not None:
                    frame = camera_core.capture_frame()
                    if frame is None:
                        frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    video_writer.write(frame)
                await asyncio.sleep(0.2)
                
        except Exception as e:
            print(f"[BRIDGE STREAM ERROR] Gagal mengirim stream: {e}")
            break
        
        # FPS Limiter Calculation
        elapsed = time.time() - start_t
        ideal_delay = 1.0 / target_fps
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
                    send_stream_task(websocket),
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