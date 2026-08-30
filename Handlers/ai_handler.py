import os
import json
import base64
import glob
import re
import shutil
try:
    import requests
except ImportError:
    requests = None
from Hardware.Computer_Vision.Quantize_model.SP_LG import run_pipeline, CONFIG as SP_LG_CONFIG
from Hardware.Computer_Vision.Quantize_model.colony_counting import count_colonies
from Hardware.Computer_Vision.classic_cv_edit import classic_cv_editor

LOCAL_TMP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tmp_images"))
if not os.path.exists(LOCAL_TMP_DIR):
    os.makedirs(LOCAL_TMP_DIR)

# Host publik VPS untuk mengunduh tile yang tidak ada di memori lokal Edge
# (mode "Input Images"). Diturunkan dari VPS_BASE_URL (".../api/dataset").
_VPS_BASE = os.getenv("VPS_BASE_URL", "http://localhost:8000/api/dataset")
VPS_STATIC_HOST = _VPS_BASE.split("/api/")[0] if "/api/" in _VPS_BASE else _VPS_BASE.rstrip("/")

# Pilihan model stitching dari frontend -> backend SP_LG (cfg['backend']).
_STITCH_BACKEND = {
    "sp_lg_tensorrt": "tensorrt",
    "sp_lg_pytorch": "pytorch",
    "sp_lg_onnx": "onnx",
}

# Bobot YOLO segmentation TensorRT (INT8) untuk perhitungan koloni -- lihat
# Quantize_model/Weights/yolo_model/. .engine tidak menyimpan metadata task
# seperti .pt, jadi task='segment' selalu di-pass eksplisit di bawah.
YOLO_SEG_MODEL = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "Computer_Vision", "Quantize_model",
    "Weights", "yolo_model", "best_Seg_1280_int8.engine"
))

# Pola nama file hasil capture tile: "<prefix>_<timestamp>_X<x>_Y<y>.jpg"
# (lihat Devices/camera.py -> save_snapshot / Handlers/camera_handler.py
# mock fallback), dengan '.' pada koordinat diganti '_'.
_COORD_PAT = re.compile(r'_X(-?\d+(?:_\d+)?)_Y(-?\d+(?:_\d+)?)\.jpg$', re.IGNORECASE)


def _parse_coord_token(token):
    """'1_5' -> 1.5, '-2' -> -2.0 (kebalikan dari str(coord).replace('.','_'))."""
    if '_' in token:
        int_part, frac_part = token.split('_', 1)
        return float(f"{int_part}.{frac_part}")
    return float(token)


def _fmt_coord(v):
    """Angka -> string yang cocok regex SP_LG 'Focused_(-?\\d+(?:\\.\\d+)?)'
    (tanpa notasi ilmiah, tanpa trailing nol)."""
    s = f"{round(float(v), 3):.3f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-") else "0"


def _resolve_tile_source(t):
    """Kembalikan path lokal ke gambar tile. Pakai file di LOCAL_TMP_DIR kalau
    ada (kasus Auto/Manual Gather); kalau tidak, unduh dari URL publik VPS
    (kasus 'Input Images' dari database/upload)."""
    fname = (t or {}).get("filename")
    if fname:
        local = os.path.join(LOCAL_TMP_DIR, fname)
        if os.path.exists(local):
            return local
    url = (t or {}).get("url")
    if url and requests is not None:
        if url.startswith("/"):
            url = VPS_STATIC_HOST + url
        try:
            dl_dir = os.path.join(LOCAL_TMP_DIR, "_stitch_dl")
            os.makedirs(dl_dir, exist_ok=True)
            dst = os.path.join(dl_dir, fname or os.path.basename(url))
            with requests.get(url, stream=True, timeout=30) as rq:
                if rq.status_code == 200:
                    with open(dst, "wb") as f:
                        for chunk in rq.iter_content(8192):
                            if chunk:
                                f.write(chunk)
                    return dst
                print(f"[BRIDGE AI WARNING] Unduh tile gagal ({rq.status_code}): {url}")
        except Exception as e:
            print(f"[BRIDGE AI WARNING] Unduh tile error: {e}")
    return None


def _stage_tiles_from_coords(tiles, stage_dir):
    """Stage tile memakai info eksplisit dari backend. Prioritas nama:
    'tile_r<row>_c<col>.jpg' (indeks grid) kalau ada gridX/gridY, kalau tidak
    'Focused_<x>_<y>.jpg' (koordinat mm). Keduanya dikenali SP_LG.load_image()."""
    if os.path.exists(stage_dir):
        shutil.rmtree(stage_dir)
    os.makedirs(stage_dir)

    staged = 0
    for t in tiles:
        t = t or {}
        gx, gy = t.get("gridX"), t.get("gridY")
        x, y = t.get("coordX"), t.get("coordY")

        if gx is not None and gy is not None and int(gx) >= 0 and int(gy) >= 0:
            dst_name = f"tile_r{int(gy)}_c{int(gx)}.jpg"
        elif x is not None and y is not None:
            dst_name = f"Focused_{_fmt_coord(x)}_{_fmt_coord(y)}.jpg"
        else:
            print(f"[BRIDGE AI WARNING] Tile tanpa posisi grid/koordinat: {t}")
            continue

        src = _resolve_tile_source(t)
        if not src:
            print(f"[BRIDGE AI WARNING] Sumber tile tidak tersedia: {t.get('filename')}")
            continue
        shutil.copy2(src, os.path.join(stage_dir, dst_name))
        staged += 1
    return staged


def _stage_tiles_for_sp_lg(image_paths, stage_dir):
    """
    SP_LG.load_image() hanya mengenali nama file 'Focused_<x>_<y>.jpg' atau
    'tile_r<row>_c<col>.jpg' untuk merekonstruksi grid. Tile hasil capture
    disimpan dengan pola '..._X<x>_Y<y>.jpg', jadi di-copy (bukan di-move,
    agar file asli di LOCAL_TMP_DIR tetap ada) ke folder staging sementara
    dengan nama yang dikenali sebelum pipeline SP_LG dijalankan.
    """
    if os.path.exists(stage_dir):
        shutil.rmtree(stage_dir)
    os.makedirs(stage_dir)

    staged = 0
    for path in image_paths:
        fname = os.path.basename(path)
        m = _COORD_PAT.search(fname)
        if not m or not os.path.exists(path):
            print(f"[BRIDGE AI WARNING] Lewati tile tanpa koordinat X/Y valid: {fname}")
            continue
        x, y = _parse_coord_token(m.group(1)), _parse_coord_token(m.group(2))
        shutil.copy2(path, os.path.join(stage_dir, f"Focused_{x}_{y}.jpg"))
        staged += 1
    return staged


def _run_colony_count(input_path, output_dir):
    """
    Jalankan model YOLO segmentation nyata (best_Seg_1280_int8.engine) lewat
    colony_counting.count_colonies(), lalu salin hasil anotasinya ke nama
    lama 'yolo_<original>.jpg' supaya kontrak filename yang sudah dipakai
    downstream (event WebSocket, file JSON di static/uploads) tetap sama.
    Return (legacy_filename, count) -- meniru signature colony_counter lama.
    """
    result = count_colonies(input_path, YOLO_SEG_MODEL, output_dir=output_dir,
                            task="segment", save_annotated=True)

    legacy_filename = "yolo_" + os.path.basename(input_path)
    legacy_path = os.path.join(output_dir, legacy_filename)
    if result["annotated_path"] and os.path.abspath(result["annotated_path"]) != os.path.abspath(legacy_path):
        shutil.copy2(result["annotated_path"], legacy_path)

    return legacy_filename, result["count"]


async def handle_ai_action(action: str, data: dict, websocket, ws_lock):
    """Router untuk instruksi Computer Vision dan AI"""
    if action == "START_STITCHING":
        model_key = (data.get("model") or "sp_lg_tensorrt").lower()
        backend = _STITCH_BACKEND.get(model_key, "tensorrt")
        print(f"[BRIDGE AI] Menjalankan Tile Stitching SP_LG (model={model_key}, backend={backend})...")

        stage_dir = os.path.join(LOCAL_TMP_DIR, "_sp_lg_stage")

        tiles = data.get("tiles")
        if tiles:
            # Jalur BARU: posisi grid / koordinat dikirim eksplisit oleh backend.
            staged = _stage_tiles_from_coords(tiles, stage_dir)
        else:
            # Jalur LAMA: koordinat dibaca dari nama file '..._X<x>_Y<y>.jpg'.
            target_images = data.get("images", [])
            if target_images:
                captured_tiles = [os.path.join(LOCAL_TMP_DIR, img) for img in target_images]
            else:
                captured_tiles = sorted(glob.glob(os.path.join(LOCAL_TMP_DIR, "IMG_*.jpg")))
            staged = _stage_tiles_for_sp_lg(captured_tiles, stage_dir)

        output_file = None
        fail_detail = ""
        if staged >= 2:
            try:
                cfg = {**SP_LG_CONFIG, "backend": backend}
                result = run_pipeline(cfg=cfg, folder_path=stage_dir)
                output_file = result["save_path"]
            except Exception as e:
                fail_detail = f"Pipeline SP_LG gagal: {e}"
                print(f"[BRIDGE AI ERROR] {fail_detail}")
        else:
            fail_detail = f"Tile dengan posisi valid kurang dari 2 (staged={staged})."
            print(f"[BRIDGE AI ERROR] {fail_detail}")

        if output_file and os.path.exists(output_file):
            local_output = os.path.join(LOCAL_TMP_DIR, "stitched_ta_output.jpg")
            shutil.copy2(output_file, local_output)

            with open(local_output, "rb") as img_file:
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
                await websocket.send(json.dumps({
                    "event": "STITCHING_FAILED", "status": "ERROR",
                    "detail": fail_detail or "Stitching gagal di Edge Device.",
                }))

    elif action == "START_DL_COUNT":
        print("[BRIDGE AI] Menjalankan YOLO Segmentation Colony Counter (best_Seg_1280_int8)...")
        target_img = os.path.join(LOCAL_TMP_DIR, "stitched_ta_output.jpg")

        try:
            result_img, count_result = _run_colony_count(target_img, LOCAL_TMP_DIR)
        except Exception as e:
            print(f"[BRIDGE AI ERROR] Colony counting gagal: {e}")
            async with ws_lock:
                await websocket.send(json.dumps({"event": "COUNTING_FAILED", "status": "ERROR"}))
        else:
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
        print(f"[BRIDGE AI] 🧬 FITUR AKTIF: YOLO Segmentation Colony Counter")
        print(f"[BRIDGE AI] ⏳ Memuat model TensorRT best_Seg_1280_int8...")
        print(f"[BRIDGE AI] 🔍 Menganalisis citra: {filename}")
        input_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../Software/backend/static/uploads", filename))
        if os.path.exists(input_path):
            output_dir = os.path.dirname(input_path)
            try:
                result_img, count_result = _run_colony_count(input_path, output_dir)
            except Exception as e:
                print(f"[BRIDGE AI ERROR] Colony counting gagal: {e}")
            else:
                json_output = os.path.join(output_dir, result_img.replace('.jpg', '.json').replace('.png', '.json'))
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
