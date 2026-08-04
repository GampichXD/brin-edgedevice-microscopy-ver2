import os
import json
import base64
import glob
from Hardware.Computer_Vision.tile_stitching import dl_stitcher
from Hardware.Computer_Vision.colony_counter import colony_counter
from Hardware.Computer_Vision.classic_cv_edit import classic_cv_editor

LOCAL_TMP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tmp_images"))
if not os.path.exists(LOCAL_TMP_DIR):
    os.makedirs(LOCAL_TMP_DIR)

async def handle_ai_action(action: str, data: dict, websocket, ws_lock):
    """Router untuk instruksi Computer Vision dan AI"""
    if action == "START_STITCHING":
        print("[BRIDGE AI] Menjalankan Deep Learning Tile Stitching...")
        
        target_images = data.get("images", [])
        if target_images:
            captured_tiles = [os.path.join(LOCAL_TMP_DIR, img) for img in target_images]
        else:
            captured_tiles = sorted(glob.glob(os.path.join(LOCAL_TMP_DIR, "IMG_*.jpg")))
        
        local_output = os.path.join(LOCAL_TMP_DIR, "stitched_ta_output.jpg")
        output_file = dl_stitcher.stitch_with_deep_learning(captured_tiles, local_output)
        
        if output_file and os.path.exists(output_file):
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
        target_img = os.path.join(LOCAL_TMP_DIR, "stitched_ta_output.jpg")
        
        result_img, count_result = colony_counter.analyze_image(target_img)
        
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
        input_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../Software/backend/static/uploads", filename))
        if os.path.exists(input_path):
            result_img, count_result = colony_counter.analyze_image(input_path)
            json_output = os.path.join(os.path.dirname(__file__), "../../../Software/backend/static/uploads", result_img.replace('.jpg', '.json').replace('.png', '.json'))
            with open(json_output, 'w') as f:
                json.dump({"colony_count": count_result}, f)
            print(f"[BRIDGE AI] ✅ Selesai. Hasil deteksi: {count_result} koloni.")
            print("="*60 + "\n")

    elif action == "CV_THRESHOLD":
        filename = data.get("filename")
        print("\n" + "-"*50)
        print(f"[BRIDGE CV] 🌗 FITUR AKTIF: Adaptive Threshold")
        print(f"[BRIDGE CV] 🧮 Menghitung nilai biner pada {filename}...")
        input_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../Software/backend/static/uploads", filename))
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
        input_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../Software/backend/static/uploads", filename))
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
        input_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../Software/backend/static/uploads", filename))
        if os.path.exists(input_path):
            res_file, stats = classic_cv_editor.calculate_morphology(input_path)
            json_output = os.path.join(os.path.dirname(__file__), "../../../Software/backend/static/uploads", res_file.replace('.jpg', '.json').replace('.png', '.json'))
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
        input_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../Software/backend/static/uploads", filename))
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
        input_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../Software/backend/static/uploads", filename))
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
        input_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../Software/backend/static/uploads", filename))
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
        input_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../Software/backend/static/uploads", filename))
        if os.path.exists(input_path):
            res_file = classic_cv_editor.split_color_channels(input_path)
            print(f"[BRIDGE CV] ✅ Pemisahan Warna Selesai. Output: {res_file}")
            print("-" * 50 + "\n")
        else:
            print(f"[BRIDGE CV ERROR] File tidak ditemukan di: {input_path}")
