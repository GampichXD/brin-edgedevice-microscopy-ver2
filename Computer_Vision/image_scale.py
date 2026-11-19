"""
Skala fisik citra (µm per piksel) untuk perhitungan luas koloni
===============================================================
Asumsi kalibrasi: piksel PERSEGI 7.0 µm/px pada tile kamera 900 x 506 px,
jadi satu tile mencakup 6.3 x 3.542 mm (900*7 µm x 506*7 µm) dan
luas_µm² = luas_px * 7.0². Frame kamera dengan resolusi lain (rasio aspek
sama, mis. 1800x1012) diskalakan dari lebarnya: µm/px = 7.0 * 900 / lebar.
Field x/y tetap disimpan terpisah (keduanya sama) supaya format data tidak
berubah bila suatu saat kalibrasi per-sumbu dibutuhkan.

Skala yang sama tidak berlaku untuk panorama hasil stitching: pipeline
mengecilkan tile dulu (CONFIG['resize_factor'] = 0.5, dan untuk grid besar
juga STITCH_BIG_SCALE), sehingga µm/px panorama = µm/px tile / faktor total.
Karena itu skala DITANAM di dalam file JPEG (segmen komentar COM, diabaikan
browser/viewer) saat stitching selesai, dan ikut ke mana pun file itu disalin
(VPS static/uploads, dataset, base64 balik ke Jetson). Re-encode lewat OpenCV
menghapus segmen ini -- classic_cv_edit._save() menyalinnya kembali.

Urutan penentuan skala (resolve_scale):
  1. override eksplisit (param umPerPxX/umPerPxY dari pemanggil)
  2. skala yang tertanam di JPEG (panorama stitching, atau hasil olahan)
  3. frame kamera mentah: rasio aspek sama dengan TILE_REF_PX (900x506, toleransi
     2%) pada resolusi berapa pun -> FOV / dimensi
  4. tidak diketahui -> luas hanya dalam px²

Konfigurasi (env, opsional):
  TILE_UM_PER_PX=7.0  TILE_REF_PX=900x506
"""

import json
import os

_COM_TAG = b"E-EYE-SCALE "


def _env_float(key, default):
    try:
        return float(os.getenv(key, default))
    except (TypeError, ValueError):
        return float(default)


def _ref_px():
    try:
        w, h = os.getenv("TILE_REF_PX", "900x506").lower().split("x")
        return int(w), int(h)
    except ValueError:
        return 900, 506


TILE_UM_PER_PX = _env_float("TILE_UM_PER_PX", 7.0)       # pada ukuran TILE_REF_PX
TILE_REF_W, TILE_REF_H = _ref_px()
TILE_FOV_MM_X = TILE_UM_PER_PX * TILE_REF_W / 1000.0     # 6.3 mm   (turunan, untuk info)
TILE_FOV_MM_Y = TILE_UM_PER_PX * TILE_REF_H / 1000.0     # 3.542 mm
ASPECT_TOL = 0.02


def tile_scale_for_dims(w, h):
    """µm/px untuk frame kamera mentah berukuran w x h, atau None bila rasio
    aspeknya bukan frame kamera (mis. gambar unggahan dari sumber lain)."""
    if not w or not h:
        return None
    ref = TILE_REF_W / TILE_REF_H
    if abs((w / h) / ref - 1.0) > ASPECT_TOL:
        return None
    s = TILE_UM_PER_PX * TILE_REF_W / w                  # piksel persegi
    return s, s


# ── Segmen COM di JPEG ─────────────────────────────────────────────────────
def _segments(data):
    """Yield (marker, start, end) untuk tiap segmen sebelum SOS. end = indeks
    byte setelah segmen. Berhenti di SOS (0xDA) / EOI."""
    if data[:2] != b"\xff\xd8":
        return
    i = 2
    while i + 4 <= len(data) and data[i] == 0xFF:
        marker = data[i + 1]
        if marker in (0xD9, 0xDA):
            return
        length = int.from_bytes(data[i + 2:i + 4], "big")
        yield marker, i, i + 2 + length
        i += 2 + length


def read_scale(path):
    """(sx, sy, source) dari segmen COM, atau None."""
    try:
        with open(path, "rb") as f:
            data = f.read(1 << 20)          # header JPEG selalu di awal file
    except OSError:
        return None
    for marker, start, end in _segments(data):
        if marker == 0xFE and data[start + 4:start + 4 + len(_COM_TAG)] == _COM_TAG:
            try:
                d = json.loads(data[start + 4 + len(_COM_TAG):end].decode("utf-8"))
                sx, sy = float(d["um_per_px_x"]), float(d["um_per_px_y"])
                if sx > 0 and sy > 0:
                    return sx, sy, str(d.get("source", "embedded"))
            except (ValueError, KeyError, UnicodeDecodeError):
                return None
    return None


def write_scale(path, sx, sy, source):
    """Tanam/ganti skala di JPEG (in place). Return True bila berhasil."""
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return False
    if data[:2] != b"\xff\xd8":
        return False
    payload = _COM_TAG + json.dumps({"um_per_px_x": round(sx, 6), "um_per_px_y": round(sy, 6),
                                     "source": source}).encode("utf-8")
    seg = b"\xff\xfe" + (len(payload) + 2).to_bytes(2, "big") + payload

    insert_at, drop = 2, []
    for marker, start, end in _segments(data):
        if 0xE0 <= marker <= 0xEF:             # APPn (JFIF/EXIF) harus tetap di depan
            insert_at = end
        if marker == 0xFE and data[start + 4:start + 4 + len(_COM_TAG)] == _COM_TAG:
            drop.append((start, end))
    for start, end in reversed(drop):          # buang tag lama
        data = data[:start] + data[end:]
        if start < insert_at:
            insert_at -= end - start
    data = data[:insert_at] + seg + data[insert_at:]
    tmp = path + ".scale.tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)
    return True


def copy_scale(src_path, dst_path):
    """Salin tag skala src -> dst (untuk olahan yang tidak mengubah geometri)."""
    s = read_scale(src_path)
    if s:
        write_scale(dst_path, s[0], s[1], s[2])
    return s


def resolve_scale(path, w, h, override=None):
    """Return dict {um_per_px_x, um_per_px_y, source} atau None (tak terkalibrasi)."""
    if override and override[0] and override[0] > 0:
        sx = float(override[0])
        sy = float(override[1]) if len(override) > 1 and override[1] and override[1] > 0 else sx
        return {"um_per_px_x": sx, "um_per_px_y": sy, "source": "manual"}
    emb = read_scale(path)
    if emb:
        return {"um_per_px_x": emb[0], "um_per_px_y": emb[1], "source": emb[2]}
    t = tile_scale_for_dims(w, h)
    if t:
        return {"um_per_px_x": t[0], "um_per_px_y": t[1],
                "source": f"camera tile {TILE_UM_PER_PX:g} um/px @ {TILE_REF_W}x{TILE_REF_H}px"
                          + ("" if w == TILE_REF_W else f" (rescaled to {w}x{h}px)")}
    return None
