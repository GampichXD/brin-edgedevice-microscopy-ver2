import os
import sys
import json
import base64
import glob
import re
import shutil
import asyncio
try:
    import requests
except ImportError:
    requests = None
from Hardware.Computer_Vision.Quantize_model.colony_counting import count_colonies
from Hardware.Computer_Vision.classic_cv_edit import classic_cv_editor

LOCAL_TMP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tmp_images"))
if not os.path.exists(LOCAL_TMP_DIR):
    os.makedirs(LOCAL_TMP_DIR)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_QM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                      "Computer_Vision", "Quantize_model"))

# Host publik VPS untuk mengunduh tile yang tidak ada di memori lokal Edge
# (mode "Input Images"). Diturunkan dari VPS_BASE_URL (".../api/dataset").
_VPS_BASE = os.getenv("VPS_BASE_URL", "http://localhost:8000/api/dataset")
VPS_STATIC_HOST = _VPS_BASE.split("/api/")[0] if "/api/" in _VPS_BASE else _VPS_BASE.rstrip("/")

# Pilihan model stitching (dari frontend) -> (script standalone, nama file hasil
# yang ditulis script itu ke dalam folder --path).
_STITCH_SCRIPTS = {
    "sp_lg_tensorrt": ("SP_LG.py",      "result_sp_lg_tensorrt.jpg"),
    "sift_bfm":       ("SIFT_BFM.py",   "result_sift_bfm.jpg"),
    "sift_lg":        ("SIFT_LG.py",    "result_sift_lg.jpg"),
    "brute_force":    ("brute_force.py", "result_brute_force.jpg"),
}

# Penanda di stdout pipeline -> (persen, label) untuk event STITCH_PROGRESS.
_STITCH_MILESTONES = [
    ("gambar dimuat",   25, "Memuat tile & menyusun grid"),
    ("Dimensi Grid",    32, "Menghitung zona overlap"),
    ("Reference image", 62, "Menautkan tile (feature matching)"),
    ("Reachable",       74, "Menghitung transformasi global"),
    ("blending",        86, "Feather blending"),
    ("Saved:",          96, "Menyimpan hasil"),
]

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


def _safe_session(s):
    """ID sesi -> nama folder aman (samakan dengan Handlers/camera_handler.py)."""
    return re.sub(r"[^A-Za-z0-9_-]", "", str(s))[:40] or "adhoc"


def _resolve_tile_source(t, session=None):
    """Kembalikan path lokal ke gambar tile. Urutan:
    (a) tmp_images/<session>/tile_r<gy>_c<gx>.jpg  (grid scan -> satu-satunya salinan lokal)
    (b) tmp_images/<filename>                       (kompat file flat lama)
    (c) unduh dari URL publik VPS                   ('Input Images' database/upload)."""
    t = t or {}
    fname = t.get("filename")
    gx, gy = t.get("gridX"), t.get("gridY")

    if session and gx is not None and gy is not None:
        try:
            p = os.path.join(LOCAL_TMP_DIR, _safe_session(session),
                             f"tile_r{int(gy)}_c{int(gx)}.jpg")
            if os.path.exists(p):
                return p
        except (TypeError, ValueError):
            pass

    if fname:
        local = os.path.join(LOCAL_TMP_DIR, fname)
        if os.path.exists(local):
            return local
    url = t.get("url")
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


def _stage_tiles_from_coords(tiles, stage_dir, session=None):
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

        src = _resolve_tile_source(t, session=session)
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


def _run_colony_count(input_path, output_dir, conf=0.25):
    """
    Jalankan model YOLO segmentation nyata (best_Seg_1280_int8.engine) lewat
    colony_counting.count_colonies(), lalu salin hasil anotasinya ke nama
    lama 'yolo_<original>.jpg' supaya kontrak filename yang sudah dipakai
    downstream (event WebSocket, file JSON di static/uploads) tetap sama.
    Return (legacy_filename, count) -- meniru signature colony_counter lama.
    """
    result = count_colonies(input_path, YOLO_SEG_MODEL, output_dir=output_dir,
                            conf=conf, task="segment", save_annotated=True)

    legacy_filename = "yolo_" + os.path.basename(input_path)
    legacy_path = os.path.join(output_dir, legacy_filename)
    if result["annotated_path"] and os.path.abspath(result["annotated_path"]) != os.path.abspath(legacy_path):
        shutil.copy2(result["annotated_path"], legacy_path)

    return legacy_filename, result["count"]


# =====================================================================
# Image Analysis (CV_*) — SEMUA dieksekusi DI JETSON, hasil dikirim balik
# ke VPS sebagai base64 lewat WebSocket (event CV_RESULT / CV_FAILED).
# Tidak ada ketergantungan filesystem bersama dengan VPS.
# =====================================================================
_CV_IN_DIR = os.path.join(LOCAL_TMP_DIR, "_cv_in")


def _fetch_analysis_input(filename, image_b64=None):
    """Path lokal ke gambar input analisis di Jetson. Urutan:
    (a) base64 dari payload perintah, (b) sudah ada di tmp_images,
    (c) unduh dari VPS <host>/static/uploads/<filename>."""
    if not filename:
        return None
    base = os.path.basename(str(filename))

    if image_b64:
        try:
            os.makedirs(_CV_IN_DIR, exist_ok=True)
            payload = image_b64.split(",", 1)[1] if "," in image_b64 else image_b64
            dst = os.path.join(_CV_IN_DIR, base)
            with open(dst, "wb") as f:
                f.write(base64.b64decode(payload))
            return dst
        except Exception as e:
            print(f"[BRIDGE CV] Gagal decode base64 input: {e}")

    local = os.path.join(LOCAL_TMP_DIR, base)
    if os.path.exists(local):
        return local

    if requests is not None:
        url = f"{VPS_STATIC_HOST}/static/uploads/{filename}"
        try:
            os.makedirs(_CV_IN_DIR, exist_ok=True)
            dst = os.path.join(_CV_IN_DIR, base)
            with requests.get(url, stream=True, timeout=30) as rq:
                if rq.status_code == 200:
                    with open(dst, "wb") as f:
                        for chunk in rq.iter_content(8192):
                            if chunk:
                                f.write(chunk)
                    return dst
                print(f"[BRIDGE CV] Unduh input gagal ({rq.status_code}): {url}")
        except Exception as e:
            print(f"[BRIDGE CV] Unduh input error: {e}")
    return None


async def _send_cv_result(websocket, ws_lock, out_path, report_name, extra=None):
    with open(out_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    msg = {
        "event": "CV_RESULT", "status": "SUCCESS",
        "filename": report_name,
        "image_data": f"data:image/jpeg;base64,{b64}",
    }
    if extra:
        msg.update(extra)
    async with ws_lock:
        await websocket.send(json.dumps(msg))


async def _send_cv_failed(websocket, ws_lock, tool, detail, output_name=None):
    print(f"[BRIDGE CV ERROR] {tool}: {detail}")
    async with ws_lock:
        await websocket.send(json.dumps({
            "event": "CV_FAILED", "status": "ERROR", "tool": tool,
            "detail": detail, "output_name": output_name,
        }))


# tool CV_* -> (method callable pada classic_cv_editor, apakah mengembalikan (name, extra))
_CV_CLASSIC = {
    "CV_THRESHOLD":   ("apply_adaptive_threshold", None),
    "CV_CONTOUR":     ("extract_and_draw_contours", None),
    "CV_SOBEL":       ("apply_sobel_edge", None),
    "CV_ROI":         ("auto_roi_crop", None),
    "CV_CALIBRATE":   ("draw_scale_calibration", None),
    "CV_COLOR_SPLIT": ("split_color_channels", None),
    "CV_MORPHOLOGY":  ("calculate_morphology", "stats"),
}


async def handle_ai_action(action: str, data: dict, websocket, ws_lock):
    """Router untuk instruksi Computer Vision dan AI"""
    if action == "START_STITCHING":
        model_key = (data.get("model") or "sp_lg_tensorrt").lower()
        if model_key not in _STITCH_SCRIPTS:
            model_key = "sp_lg_tensorrt"
        session = data.get("session")
        print(f"[BRIDGE AI] Tile Stitching (model={model_key}, sesi={session})...")

        async def emit_progress(pct, label):
            try:
                async with ws_lock:
                    await websocket.send(json.dumps({
                        "event": "STITCH_PROGRESS", "pct": int(pct), "phase": label,
                    }))
            except Exception:
                pass

        await emit_progress(5, "Menyiapkan tile")

        tiles = data.get("tiles")
        stage_dir = None
        staged = 0

        # 1) Folder sesi terisolasi yang sudah berisi 'tile_r<row>_c<col>.jpg'.
        if session:
            sess_dir = os.path.join(LOCAL_TMP_DIR, _safe_session(session))
            grid_files = glob.glob(os.path.join(sess_dir, "tile_r*_c*.jpg"))
            if os.path.isdir(sess_dir) and len(grid_files) >= 2:
                stage_dir = sess_dir
                staged = len(grid_files)
                print(f"[BRIDGE AI] Memakai folder sesi ({staged} tile): {sess_dir}")

        # 2) Fallback: stage dari daftar 'tiles' ke folder staging PER-SESI.
        if stage_dir is None:
            stage_dir = os.path.join(LOCAL_TMP_DIR, "_stage_" + _safe_session(session or "adhoc"))
            if tiles:
                staged = _stage_tiles_from_coords(tiles, stage_dir, session=session)
            elif data.get("images"):
                captured_tiles = [os.path.join(LOCAL_TMP_DIR, img) for img in data["images"]]
                staged = _stage_tiles_for_sp_lg(captured_tiles, stage_dir)
            else:
                print("[BRIDGE AI ERROR] Tidak ada tile yang ditentukan (tiles/images kosong).")
                staged = 0

        output_file = None
        fail_detail = ""

        if staged >= 2:
            script, result_name = _STITCH_SCRIPTS[model_key]
            script_path = os.path.join(_QM_DIR, script)
            expected = os.path.join(stage_dir, result_name)
            if os.path.exists(expected):
                os.remove(expected)

            await emit_progress(12, f"Menjalankan {model_key}")
            try:
                env = dict(os.environ)
                env["PYTHONPATH"] = os.pathsep.join(
                    [_REPO_ROOT, _QM_DIR, env.get("PYTHONPATH", "")])
                proc = await asyncio.create_subprocess_exec(
                    sys.executable, script_path, "--path", stage_dir,
                    cwd=_QM_DIR, env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
                seen = set()
                while True:
                    raw = await proc.stdout.readline()
                    if not raw:
                        break
                    line = raw.decode("utf-8", "replace").rstrip()
                    if line:
                        print(f"[STITCH:{model_key}] {line}")
                    for key, pct, label in _STITCH_MILESTONES:
                        if key in line and key not in seen:
                            seen.add(key)
                            await emit_progress(pct, label)
                rc = await proc.wait()
                if rc == 0 and os.path.exists(expected):
                    output_file = expected
                else:
                    fail_detail = (f"Pipeline {model_key} berakhir (exit {rc}) tanpa "
                                   f"menghasilkan mosaik — kemungkinan overlap antar-tile "
                                   f"terlalu kecil.")
            except Exception as e:
                fail_detail = f"Gagal menjalankan {model_key}: {e}"
                print(f"[BRIDGE AI ERROR] {fail_detail}")
        else:
            fail_detail = f"Tile dengan posisi valid kurang dari 2 (staged={staged})."
            print(f"[BRIDGE AI ERROR] {fail_detail}")

        if output_file and os.path.exists(output_file):
            await emit_progress(98, "Mengunggah hasil")
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
            print(f"[BRIDGE ERROR] Stitching gagal: {fail_detail}")
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
        params = data.get("params") or {}

        res_file = None
        if edit_type == "THRESHOLD":
            res_file = classic_cv_editor.apply_adaptive_threshold(target_img, params)
        elif edit_type == "BRIGHTNESS":
            res_file = classic_cv_editor.apply_brightness_contrast(
                target_img, brightness=data.get("brightness", 0), contrast=data.get("contrast", 0))

        if not res_file:
            async with ws_lock:
                await websocket.send(json.dumps({
                    "event": "EDIT_FAILED", "status": "ERROR",
                    "detail": f"APPLY_IMAGE_EDIT: tipe '{edit_type}' tak dikenal atau gambar tidak ada.",
                }))
        else:
            local_edited_path = os.path.join(classic_cv_editor.output_dir, res_file)
            with open(local_edited_path, "rb") as img_file:
                encoded_edited = base64.b64encode(img_file.read()).decode('utf-8')
            async with ws_lock:
                await websocket.send(json.dumps({
                    "event": "EDIT_COMPLETE",
                    "status": "SUCCESS",
                    "image_data": f"data:image/jpeg;base64,{encoded_edited}",
                    "filename": res_file
                }))

    elif action in ("CV_COLONY_COUNT", "CV_THRESHOLD", "CV_CONTOUR", "CV_MORPHOLOGY",
                    "CV_SOBEL", "CV_ROI", "CV_CALIBRATE", "CV_COLOR_SPLIT"):
        tool = action
        filename = data.get("filename")
        params = data.get("params") or {}
        # Nama file hasil yang HARUS dilaporkan balik ke VPS (backend menunggu
        # file bernama persis ini di static/uploads). Disuplai backend.
        report_name = data.get("output_name")
        print(f"[BRIDGE CV] {tool} pada '{filename}' (params={params}) — DIEKSEKUSI DI JETSON")

        src = _fetch_analysis_input(filename, data.get("image_b64"))
        if not src or not os.path.exists(src):
            await _send_cv_failed(websocket, ws_lock, tool,
                                  f"Gambar input tidak tersedia di Edge: {filename}", report_name)
            return

        try:
            out_dir = classic_cv_editor.output_dir
            extra = None

            if tool == "CV_COLONY_COUNT":
                try:
                    conf = float(params.get("conf", 0.25))
                except (TypeError, ValueError):
                    conf = 0.25
                out_name, count = await asyncio.to_thread(_run_colony_count, src, out_dir, conf)
                extra = {"colonies": int(count)}
                print(f"[BRIDGE CV] YOLO colony count = {count} (conf={conf})")
            else:
                method_name, extra_key = _CV_CLASSIC[tool]
                method = getattr(classic_cv_editor, method_name)
                result = await asyncio.to_thread(method, src, params)
                if extra_key == "stats":
                    out_name, stats = result
                    extra = {"stats": stats}
                else:
                    out_name = result

            if not out_name:
                await _send_cv_failed(websocket, ws_lock, tool,
                                      "Proses CV tidak menghasilkan output (gambar tak terbaca?).", report_name)
                return
            out_path = os.path.join(out_dir, out_name)
            if not os.path.exists(out_path):
                await _send_cv_failed(websocket, ws_lock, tool, f"File output hilang: {out_name}", report_name)
                return

            await _send_cv_result(websocket, ws_lock, out_path, report_name or out_name, extra)
            print(f"[BRIDGE CV] ✅ {tool} selesai -> dilaporkan sbg {report_name or out_name}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            await _send_cv_failed(websocket, ws_lock, tool, f"{type(e).__name__}: {e}", report_name)
