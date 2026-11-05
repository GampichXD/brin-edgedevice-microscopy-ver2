"""Pengukuran posisi dari citra stage micrometer, tanpa perangkat keras (lihat t01_positioning.py).

Garis micrometer berulang tiap periode P, jadi fase garis saja ambigu kelipatan P.
Karena itu: kasar = lokasi penanda unik (template), halus = fase garis periodik (sub-piksel),
gabungan = P*round((kasar - halus)/P) + halus.
"""
import cv2
import numpy as np


def preprocess(frame):
    """Kanal hijau (SNR tertinggi), flat-field semu untuk vignetting, lalu CLAHE."""
    green = frame[:, :, 1].astype(np.float32)
    background = cv2.GaussianBlur(green, (0, 0), 30)
    flat = green / np.maximum(background, 1.0)
    flat = cv2.normalize(flat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(flat)


def line_profile(gray, band, axis):
    """Profil intensitas tegak lurus garis, dirata-ratakan di dalam pita skala (x, y, w, h).
    axis 'X': garis vertikal, profil sepanjang kolom. axis 'Y': garis horizontal."""
    x, y, w, h = band
    roi = gray[y:y + h, x:x + w].astype(np.float64)
    profile = roi.mean(axis=0 if axis == "X" else 1)
    return profile - profile.mean()


def estimate_period_px(profile, min_px=3.0):
    """Periode dominan garis (piksel), dihitung dari seluruh pita agar error skala kecil."""
    n = len(profile)
    spectrum = np.abs(np.fft.rfft(profile * np.hanning(n), n * 8))
    freqs = np.fft.rfftfreq(n * 8)
    valid = (freqs > 1.0 / (n / 2)) & (freqs < 1.0 / min_px)
    return 1.0 / freqs[valid][np.argmax(spectrum[valid])]


def fine_phase_px(profile, period_px):
    """Posisi garis modulo periode, dengan presisi sub-piksel, dari fase pada frekuensi 1/P."""
    x = np.arange(len(profile))
    component = np.sum(profile * np.exp(-2j * np.pi * x / period_px))
    return (-np.angle(component) / (2 * np.pi) * period_px) % period_px


def _subpixel_peak(values, index):
    """Puncak parabola dari tiga titik di sekitar indeks."""
    if index <= 0 or index >= len(values) - 1:
        return float(index)
    a, b, c = values[index - 1], values[index], values[index + 1]
    denominator = a - 2 * b + c
    return float(index) if denominator == 0 else index + 0.5 * (a - c) / denominator


def locate_marker(gray, template):
    """Lokasi unik penanda. Mengembalikan (x, y, skor TM_CCOEFF_NORMED)."""
    result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, (px, py) = cv2.minMaxLoc(result)
    return _subpixel_peak(result[py, :], px), _subpixel_peak(result[:, px], py), float(score)


def combine(coarse_px, fine_px, period_px):
    """Gabungkan posisi kasar dan halus. valid=False kalau keduanya berselisih mendekati P/2,
    karena kelipatan periodenya bisa salah pilih."""
    multiple = round((coarse_px - fine_px) / period_px)
    position = period_px * multiple + fine_px
    return position, abs(position - coarse_px) < 0.4 * period_px


def motion_direction(dx_px, dy_px):
    """Vektor satuan arah sumbu meja di citra, dari gerak kalibrasi. Citra bisa tercermin,
    jadi arahnya harus diukur, tidak boleh diasumsikan."""
    norm = float(np.hypot(dx_px, dy_px))
    if norm < 1.0:
        raise ValueError("Penanda tidak bergeser saat gerak kalibrasi. Periksa motor dan ROI.")
    return dx_px / norm, dy_px / norm


def target_stays_in_frame(marker_xy, template_size, frame_size, direction, um_per_px, targets_mm, margin_mm=0.1):
    """Daftar target (mm) yang membuat penanda keluar bidang pandang, termasuk posisi awal -margin."""
    mx, my = marker_xy
    tw, th = template_size
    fw, fh = frame_size
    bad = []
    for target in list(targets_mm) + [min(targets_mm) - margin_mm]:
        shift_px = target * 1000.0 / um_per_px
        x, y = mx + direction[0] * shift_px, my + direction[1] * shift_px
        if x < 0 or y < 0 or x + tw > fw or y + th > fh:
            bad.append(target)
    return bad
