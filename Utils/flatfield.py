"""Flat-field correction (B9). Default mati: mengubah warna input YOLO bisa menurunkan
akurasi colony count, jadi aktifkan FLATFIELD_ENABLED=1 hanya setelah disepakati dengan
subsistem pengolahan citra.

Rumus: correct(img) = clip((img - dark) / (flat - dark) * mean(flat - dark)), per kanal.
"""
import os

import numpy as np

CALIB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "local_calibration"))
FLAT_PATH = os.path.join(CALIB_DIR, "flat.npy")
DARK_PATH = os.path.join(CALIB_DIR, "dark.npy")
REFERENCE_FRAMES = 20


def enabled() -> bool:
    return os.getenv("FLATFIELD_ENABLED", "0") == "1"


def capture_reference(grab_frame, mode: str, frames: int = REFERENCE_FRAMES) -> str:
    """Rata-ratakan frame referensi. mode='flat': slide kosong, LED menyala, fokus dan
    exposure sama dengan saat scan. mode='dark': LED mati. `grab_frame` mengembalikan
    citra BGR numpy, atau None kalau kamera belum siap."""
    if mode not in ("flat", "dark"):
        raise ValueError("mode harus 'flat' atau 'dark'")
    total, count = None, 0
    for _ in range(frames):
        frame = grab_frame()
        if frame is None:
            continue
        total = frame.astype(np.float64) if total is None else total + frame
        count += 1
    if count == 0:
        raise RuntimeError("Kamera tidak menghasilkan frame untuk referensi.")
    os.makedirs(CALIB_DIR, exist_ok=True)
    path = FLAT_PATH if mode == "flat" else DARK_PATH
    np.save(path, (total / count).astype(np.float32))
    return path


def correct(img: np.ndarray, flat: np.ndarray, dark: np.ndarray) -> np.ndarray:
    gain = flat.astype(np.float64) - dark
    # Piksel referensi yang sama gelapnya dengan dark tidak punya informasi: biarkan apa adanya.
    usable = gain > 1.0
    scale = np.ones_like(gain)
    for ch in range(gain.shape[2]):
        channel, mask = gain[..., ch], usable[..., ch]
        if mask.any():
            scale[..., ch][mask] = channel[mask].mean() / channel[mask]
    out = (img.astype(np.float64) - dark) * scale
    return np.clip(out, 0, 255).astype(np.uint8)


def load_reference():
    """(flat, dark), atau None kalau kalibrasi belum dilakukan."""
    if not (os.path.exists(FLAT_PATH) and os.path.exists(DARK_PATH)):
        return None
    return np.load(FLAT_PATH), np.load(DARK_PATH)
