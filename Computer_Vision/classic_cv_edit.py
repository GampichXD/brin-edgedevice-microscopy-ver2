"""
Classic (non-AI) Computer Vision editor — runs ON THE JETSON EDGE DEVICE.
=======================================================================
Every method here is pure OpenCV/NumPy and is invoked by
Hardware/Handlers/ai_handler.py in response to CV_* commands coming from the
VPS. The VPS never runs these itself when the Edge is online — it only ships
the input image (base64 / URL) and reads back the annotated result.

Contract for each method:
    method(image_path: str, params: dict | None) -> str            # writes 1 file, returns its basename
    calculate_morphology(image_path, params) -> (str, dict)         # + stats dict

Output goes to `self.output_dir` (a Jetson-local scratch folder). The caller
resolves the full path as os.path.join(editor.output_dir, returned_name).
`params` is always optional; sensible defaults keep the old call sites working.
"""

import os
import json
import cv2
import numpy as np


def _p(params, key, default, cast=float):
    """Ambil params[key] dengan aman -> cast; kalau kosong/aneh pakai default."""
    if not params:
        return default
    v = params.get(key, default)
    if v is None or v == "":
        return default
    try:
        return cast(v)
    except (TypeError, ValueError):
        return default


class ClassicCVImageEditor:
    def __init__(self, output_dir=None):
        # Default: folder scratch LOKAL Jetson (bukan path bersama VPS).
        if output_dir is None:
            output_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..", "tmp_images", "cv_out")
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    def _save(self, prefix, image_path, img):
        name = f"{prefix}{os.path.basename(image_path)}"
        # Normalisasi ekstensi -> .jpg supaya kontrak filename downstream stabil.
        stem, _ext = os.path.splitext(name)
        name = stem + ".jpg"
        cv2.imwrite(os.path.join(self.output_dir, name), img)
        return name

    # ── Image enhancement ────────────────────────────────────────────────
    def apply_brightness_contrast(self, image_path, params=None, brightness=None, contrast=None):
        img = cv2.imread(image_path)
        if img is None:
            return None
        brightness = int(brightness if brightness is not None else _p(params, "brightness", 0, int))
        contrast = int(contrast if contrast is not None else _p(params, "contrast", 0, int))

        buf = img.copy()
        if brightness != 0:
            shadow = brightness if brightness > 0 else 0
            highlight = 255 if brightness > 0 else 255 + brightness
            alpha_b = (highlight - shadow) / 255
            buf = cv2.addWeighted(img, alpha_b, img, 0, shadow)
        if contrast != 0:
            f = 131 * (contrast + 127) / (127 * (131 - contrast))
            buf = cv2.addWeighted(buf, f, buf, 0, 127 * (1 - f))
        return self._save("edited_bc_", image_path, buf)

    def apply_adaptive_threshold(self, image_path, params=None, block_size=None, c_value=None):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        block_size = int(block_size if block_size is not None
                         else _p(params, "blockSize", _p(params, "block_size", 11, int), int))
        c_value = int(c_value if c_value is not None
                      else _p(params, "C", _p(params, "c_value", 2, int), int))
        block_size = max(3, block_size)
        if block_size % 2 == 0:
            block_size += 1
        thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, block_size, c_value)
        return self._save("edited_thresh_", image_path, thresh)

    def extract_and_draw_contours(self, image_path, params=None):
        img = cv2.imread(image_path)
        if img is None:
            return None
        lo = int(_p(params, "cannyLo", 30, int))
        hi = int(_p(params, "cannyHi", 150, int))
        blur = int(_p(params, "blur", 5, int))
        if blur % 2 == 0:
            blur += 1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (blur, blur), 0)
        edged = cv2.Canny(blurred, lo, hi)
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = img.copy()
        cv2.drawContours(result, contours, -1, (0, 0, 255), 2)
        return self._save("edited_contours_", image_path, result)

    def apply_sobel_edge(self, image_path, params=None):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        ksize = int(_p(params, "ksize", 3, int))
        if ksize % 2 == 0:
            ksize += 1
        ksize = max(1, min(31, ksize))
        sx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
        sy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
        mag = cv2.magnitude(sx, sy)
        if mag.max() > 0:
            mag = mag / mag.max() * 255.0
        return self._save("edited_sobel_", image_path, np.uint8(np.clip(mag, 0, 255)))

    # ── Analysis ────────────────────────────────────────────────────────
    def calculate_morphology(self, image_path, params=None):
        img = cv2.imread(image_path)
        if img is None:
            return None, None
        min_area = float(_p(params, "minArea", 15.0))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        total_area = total_perimeter = 0.0
        valid = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                total_area += area
                total_perimeter += cv2.arcLength(cnt, True)
                valid += 1
                cv2.drawContours(img, [cnt], -1, (255, 0, 0), 2)
        name = self._save("edited_morphology_", image_path, img)
        stats = {
            "total_cells": valid,
            "avg_area": round(total_area / valid if valid else 0, 2),
            "avg_perimeter": round(total_perimeter / valid if valid else 0, 2),
            "total_area": round(total_area, 2),
        }
        return name, stats

    # ── Editor utilities ───────────────────────────────────────────────
    def auto_roi_crop(self, image_path, params=None):
        img = cv2.imread(image_path)
        if img is None:
            return None
        h, w = img.shape[:2]

        def _dim(key, frac_default, span):
            v = _p(params, key, None)
            if v is None:
                return None
            # <=1 -> perlakukan sebagai fraksi; >1 -> piksel absolut
            return int(v * span) if 0 <= v <= 1 else int(v)

        x = _dim("x", None, w)
        y = _dim("y", None, h)
        bw = _dim("w", None, w)
        bh = _dim("h", None, h)
        if None in (x, y, bw, bh):
            # Default lama: crop tengah 1/2 x 1/2
            bw, bh = w // 2, h // 2
            x, y = (w - bw) // 2, (h - bh) // 2
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        bw = max(1, min(bw, w - x))
        bh = max(1, min(bh, h - y))
        roi = img[y:y + bh, x:x + bw]
        return self._save("edited_roi_", image_path, roi if roi.size else img)

    def draw_scale_calibration(self, image_path, params=None):
        img = cv2.imread(image_path)
        if img is None:
            return None
        h, w = img.shape[:2]
        um_per_px = _p(params, "umPerPx", None)
        bar_um = _p(params, "barUm", 100.0)
        if um_per_px and um_per_px > 0:
            bar_len = int(round(bar_um / um_per_px))
            label = f"{bar_um:g} um"
        else:
            bar_len = w // 5
            label = _p(params, "label", f"{bar_um:g} um", str) if params else f"{bar_um:g} um"
        bar_len = max(10, min(bar_len, w - 40))
        x1, y1 = 20, h - 40
        x2, y2 = x1 + bar_len, h - 20
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 2)
        cv2.putText(img, label, (x1 + 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 4)
        cv2.putText(img, label, (x1 + 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        return self._save("edited_calibrate_", image_path, img)

    def split_color_channels(self, image_path, params=None):
        img = cv2.imread(image_path)
        if img is None:
            return None
        b, g, r = cv2.split(img)
        zeros = np.zeros_like(b)
        channel = (_p(params, "channel", "all", str) or "all").lower()
        single = {
            "red": cv2.merge([zeros, zeros, r]),
            "green": cv2.merge([zeros, g, zeros]),
            "blue": cv2.merge([b, zeros, zeros]),
        }
        if channel in single:
            return self._save("edited_colorsplit_", image_path, single[channel])
        top = np.hstack((img, single["red"]))
        bottom = np.hstack((single["green"], single["blue"]))
        grid = np.vstack((top, bottom))
        mx = max(grid.shape[:2])
        if mx > 2000:
            s = 2000 / mx
            grid = cv2.resize(grid, (0, 0), fx=s, fy=s)
        return self._save("edited_colorsplit_", image_path, grid)


# Singleton editor (dipakai Hardware/Handlers/ai_handler.py).
classic_cv_editor = ClassicCVImageEditor()
