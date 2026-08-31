import os
import re
import json
import base64
import shutil
import asyncio
import cv2
from datetime import datetime
from Hardware.Devices.camera import camera_core

LOCAL_TMP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tmp_images"))
if not os.path.exists(LOCAL_TMP_DIR):
    os.makedirs(LOCAL_TMP_DIR)


def _safe_session(s):
    return re.sub(r"[^A-Za-z0-9_-]", "", str(s))[:40] or "adhoc"

async def handle_camera_action(action: str, data: dict, websocket, ws_lock, state: dict):
    """Router untuk instruksi dan konfigurasi Kamera"""
    if action == "APPLY_CAMERA_SETTINGS":
        print(
            f"[BRIDGE CAMERA] Settings diterima: shutter={data.get('shutter_speed')} iso={data.get('iso')}"
        )
        # apply_settings me-restart pipeline GStreamer -> blocking beberapa detik.
        # Jalankan di thread pool agar tidak memblokir event loop / WebSocket.
        await asyncio.to_thread(
            camera_core.apply_settings,
            data.get("shutter_speed"),
            data.get("iso"),
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
            # ABSOLUT relatif ke Hardware/local_datasets (bukan CWD) — sama dengan
            # yang dipantau vps_syncer.py, jadi video pasti ke-sync ke VPS.
            rec_dir = os.path.abspath(os.path.join(
                os.path.dirname(__file__), "..", "local_datasets", folder_id))
            os.makedirs(rec_dir, exist_ok=True)
            # .mp4 (mp4v) — codec paling kompatibel di OpenCV Jetson & browser.
            filename = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
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
        session = data.get("session")
        gx, gy = data.get("grid_x"), data.get("grid_y")
        is_grid = bool(session) and gx is not None and gy is not None
        print(f"[BRIDGE] 📸 CAPTURE_IMAGE (grid={is_grid}, prefix={prefix}, X={cx}, Y={cy})...")

        # Grid scan  -> simpan LANGSUNG ke tmp_images/<session>/tile_r<row>_c<col>.jpg
        #               (satu-satunya salinan lokal; TIDAK ada duplikat flat).
        # Manual/dsb -> tetap flat di tmp_images/ seperti biasa.
        if is_grid:
            save_dir = os.path.join(LOCAL_TMP_DIR, _safe_session(session))
            local_name = f"tile_r{int(gy)}_c{int(gx)}.jpg"
        else:
            save_dir = LOCAL_TMP_DIR
            local_name = requested_filename  # boleh None -> camera_core buat nama sendiri

        saved = camera_core.save_snapshot(save_dir, prefix=prefix, coord_x=cx, coord_y=cy,
                                          requested_filename=local_name)
        if saved:
            local_path = os.path.join(save_dir, saved)
            print(f"[BRIDGE CAMERA] Tersimpan: {local_path}")
        else:
            print("[BRIDGE CAMERA WARNING] Kamera fisik offline. Membangkitkan gambar mockup...")
            import numpy as np
            import random
            saved = local_name or (
                f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                f"_X{str(cx).replace('.','_')}_Y{str(cy).replace('.','_')}.jpg"
            )
            os.makedirs(save_dir, exist_ok=True)
            local_path = os.path.join(save_dir, saved)
            mock_img = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_img[:] = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
            cv2.putText(mock_img, f"MOCK {saved}", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            cv2.imwrite(local_path, mock_img)
            print(f"[BRIDGE CAMERA] MOCK Tersimpan: {local_path}")

        # Nama yang DILAPORKAN ke VPS (kontrak static/uploads + capturedImages di
        # frontend). Grid -> nama dari backend 'IMG_<session>_r_c.jpg' (unik lintas
        # sesi). Manual -> nama file apa adanya.
        report_name = requested_filename if (is_grid and requested_filename) else saved
        if is_grid:
            print(f"[BRIDGE CAMERA] Sesi: {local_path} (dilaporkan sbg {report_name})")

        # UPLOAD KE VPS SUPAYA BISA DILIHAT DI BROWSER.
        # Kirim versi PREVIEW yang dikecilkan (bukan full-res) supaya paket
        # WebSocket kecil -> tidak menyumbat stream/telemetri saat grid besar
        # (mis. 5x5+). Full-res tetap ada di folder sesi untuk stitching.
        try:
            src_path = local_path
            frame_bytes = None
            try:
                img = cv2.imread(src_path)
                if img is not None:
                    h, w = img.shape[:2]
                    if w > 900:
                        img = cv2.resize(img, (900, max(1, int(h * 900 / w))))
                    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if ok:
                        frame_bytes = buf.tobytes()
            except Exception:
                frame_bytes = None
            if frame_bytes is None:
                with open(src_path, "rb") as f:
                    frame_bytes = f.read()
            encoded_image = base64.b64encode(frame_bytes).decode('utf-8')
            async with ws_lock:
                await websocket.send(json.dumps({
                    "event": "IMAGE_CAPTURED",
                    "filename": report_name,
                    "image_data": f"data:image/jpeg;base64,{encoded_image}"
                }))
        except Exception as e:
            print(f"[BRIDGE ERROR] Gagal mengirim base64 image ke websocket: {e}")

    elif action == "PURGE_CACHE":
        print("[BRIDGE] 🧹 Menerima perintah PURGE_CACHE dari VPS.")
        if os.path.exists(LOCAL_TMP_DIR):
            cleared = 0
            for entry in os.listdir(LOCAL_TMP_DIR):
                path = os.path.join(LOCAL_TMP_DIR, entry)
                try:
                    if os.path.isdir(path):
                        # folder sesi scan / staging stitching
                        shutil.rmtree(path, ignore_errors=True)
                        cleared += 1
                    elif entry.lower().endswith(('.png', '.jpg', '.jpeg')):
                        os.remove(path)
                        cleared += 1
                except Exception:
                    pass
            print(f"[BRIDGE] ✅ Purge selesai. {cleared} item dihapus dari {LOCAL_TMP_DIR}.")
            
    elif action == "PURGE_DATASET_FOLDER":
        # VPS menghapus 1 folder dataset -> hapus salinan lokal di local_datasets/<id>
        folder_id = _safe_session(data.get("folder_id", ""))
        if folder_id:
            target = os.path.abspath(os.path.join(
                os.path.dirname(__file__), "..", "local_datasets", folder_id))
            if os.path.isdir(target):
                shutil.rmtree(target, ignore_errors=True)
                print(f"[BRIDGE] 🧹 PURGE_DATASET_FOLDER: {target} dihapus.")
            else:
                print(f"[BRIDGE] PURGE_DATASET_FOLDER: {target} tidak ada, lewati.")

    elif action == "PURGE_DATASET_FILES":
        # VPS menghapus beberapa gambar -> hapus file yang cocok di local_datasets/<id>
        folder_id = _safe_session(data.get("folder_id", ""))
        filenames = data.get("filenames", []) or []
        target_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "local_datasets", folder_id))
        removed = 0
        if folder_id and os.path.isdir(target_dir):
            existing = set(os.listdir(target_dir))
            for fn in filenames:
                base = os.path.basename(str(fn))
                # cocokkan nama persis ATAU prefix (vps sering menambah _timestamp)
                stem, ext = os.path.splitext(base)
                for cand in list(existing):
                    if cand == base or cand.startswith(stem + "_") or cand == stem + ext:
                        try:
                            os.remove(os.path.join(target_dir, cand))
                            existing.discard(cand)
                            removed += 1
                        except Exception:
                            pass
        print(f"[BRIDGE] 🧹 PURGE_DATASET_FILES: {removed} file dihapus dari {target_dir}.")

    elif action == "PURGE_ALL_DATASETS":
        print("[BRIDGE] 🚨 Menerima perintah PURGE_ALL_DATASETS dari VPS!")
        import shutil
        # Hapus LOCAL_TMP_DIR isinya
        if os.path.exists(LOCAL_TMP_DIR):
            shutil.rmtree(LOCAL_TMP_DIR, ignore_errors=True)
            os.makedirs(LOCAL_TMP_DIR, exist_ok=True)
        # Hapus seluruh folder local_datasets
        local_datasets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "local_datasets"))
        if os.path.exists(local_datasets_dir):
            shutil.rmtree(local_datasets_dir, ignore_errors=True)
            os.makedirs(local_datasets_dir, exist_ok=True)
        print("[BRIDGE] ✅ Seluruh data di Edge berhasil dimusnahkan.")
