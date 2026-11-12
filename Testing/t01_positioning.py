#!/usr/bin/env python3
"""
t01_positioning.py — Uji akurasi & repeatability posisi X/Y memakai stage micrometer
(adaptasi ISO 230-2). Dijalankan di JETSON saat main.py MATI.

Metode "kasar + halus":
  - KASAR : posisi satu penanda unik (angka skala) dilacak dengan template matching.
            Unik -> tidak ambigu, tapi kurang presisi.
  - HALUS : fase garis-garis kecil micrometer (periodik) dari profil intensitas.
            Presisi sub-piksel, tapi ambigu kelipatan periode P.
  - GABUNG: d = P*round((d_kasar - d_halus)/P) + d_halus   (syarat: |error kasar| < P/2)

Subperintah:
  calibrate  : hitung um/px, sudut garis, arah gerak sumbu di citra, simpan calib_<axis>.json
  run        : jalankan uji dua arah, tulis positioning.csv
  selftest   : uji algoritma pada satu foto dengan pergeseran sintetis (tanpa motor)

Contoh:
  python3 t01_positioning.py calibrate --axis X --minor-um 10 --roi 560,120,90,70
  python3 t01_positioning.py run --axis X --targets 0,0.13,0.27,0.38,0.51 --runs 5
  python3 t01_positioning.py selftest --image foto_slide.jpg --roi 560,120,90,70
"""
import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DEFAULT_LOG_DIR as LOG_DIR, append_row, fresh_frame, open_camera, open_motor, ensure_exclusive_hardware, wait_idle  # noqa: E402


# ======================================================================= algoritma (murni)
def preprocess(img_bgr):
    """Kanal hijau (SNR tertinggi pada kamera ini) + flat-field sederhana."""
    g = img_bgr[:, :, 1].astype(np.float32) if img_bgr.ndim == 3 else img_bgr.astype(np.float32)
    bg = cv2.GaussianBlur(g, (0, 0), 30) + 1e-3
    n = g / bg
    n = (n - n.mean()) / (n.std() + 1e-6)
    return n


def ruler_geometry(pre, min_period_px=6, max_period_px=80, max_angle_deg=20):
    """Cari frekuensi spasial dominan garis kecil dengan FFT 2D.
    Kembalikan (period_px, u) dengan u = vektor satuan sepanjang penggaris (tegak lurus garis)."""
    h, w = pre.shape
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    F = np.abs(np.fft.fftshift(np.fft.fft2(pre * win)))
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    fy, fx = (yy - cy) / h, (xx - cx) / w
    fr = np.hypot(fx, fy) + 1e-9
    # Hanya arah dekat horizontal/vertikal (±max_angle_deg). Goresan diagonal di
    # slide/lensa (±45°) punya frekuensi kuat dan harus diabaikan.
    ang = np.degrees(np.arctan2(np.abs(fy), np.abs(fx)))          # 0..90
    near_axis = (ang < max_angle_deg) | (ang > 90 - max_angle_deg)
    mask = (fr > 1.0 / max_period_px) & (fr < 1.0 / min_period_px) & near_axis
    F2 = np.where(mask, F, 0)
    iy, ix = np.unravel_index(np.argmax(F2), F2.shape)
    vx, vy = fx[iy, ix], fy[iy, ix]
    f = math.hypot(vx, vy)
    u = np.array([vx / f, vy / f])
    # arah kanonik: komponen terbesar bernilai positif
    if (abs(u[0]) >= abs(u[1]) and u[0] < 0) or (abs(u[1]) > abs(u[0]) and u[1] < 0):
        u = -u
    return 1.0 / f, u


def ruler_geometry_near(pre, roi, half=128, min_period_px=6, max_period_px=80):
    """Arah & periode GARIS KECIL di sekitar penanda (ROI).
    1) arah dari FFT 2D potongan di sekitar ROI (hanya dekat horizontal/vertikal);
    2) periode = puncak autokorelasi pertama (tahan terhadap harmonik garis tipis);
    3) diperhalus pada pita penuh selebar citra."""
    x, y, w, h = roi
    cx, cy = x + w // 2, y + h // 2
    H, W = pre.shape
    crop = pre[max(0, cy - half):min(H, cy + half), max(0, cx - half):min(W, cx + half)]
    _, u = ruler_geometry(crop, min_period_px, max_period_px)
    rc = _rotate_to_u(crop, u)
    prof = rc.mean(axis=0)
    prof = prof - prof.mean()
    # Autokorelasi: puncak PERTAMA setelah lag 0 = periode dasar garis kecil.
    # (Spektrum FFT tidak dipakai di sini karena garis tipis menghasilkan
    #  harmonik kuat di P/2, P/3 yang bisa salah dipilih.)
    ac = np.correlate(prof, prof, mode="full")[len(prof) - 1:]
    ac = ac / (ac[0] + 1e-9)
    lo, hi = int(min_period_px), min(int(max_period_px), len(ac) - 2)
    peaks = [k for k in range(lo, hi) if ac[k] >= ac[k - 1] and ac[k] >= ac[k + 1] and ac[k] > 0.2]
    if not peaks:
        raise RuntimeError("Garis periodik tidak terdeteksi di sekitar ROI. Pilih ROI yang menempel pada penggaris.")
    k = peaks[0]
    d = ac[k - 1] - 2 * ac[k] + ac[k + 1]
    p0 = k + (0.5 * (ac[k - 1] - ac[k + 1]) / d if abs(d) > 1e-9 else 0.0)
    return refine_period(pre, u, p0), u


def refine_period(pre, u, p0, span=0.06, steps=301):
    """Cari periode presisi di sekitar p0 pada pita garis selebar citra."""
    rot = _rotate_to_u(pre, u)
    rows = band_rows(rot, p0)
    prof = rot[rows[0]:rows[1], :].mean(axis=0)
    prof = (prof - prof.mean()) * np.hanning(len(prof))
    n = np.arange(len(prof))
    best, bp = -1, p0
    for P in np.linspace(p0 * (1 - span), p0 * (1 + span), steps):
        m = abs(np.sum(prof * np.exp(-2j * np.pi * n / P)))
        if m > best:
            best, bp = m, P
    return float(bp)


def _rotate_to_u(pre, u):
    """Putar citra supaya arah u menjadi sumbu +x (garis penggaris jadi vertikal)."""
    h, w = pre.shape
    ang = math.degrees(math.atan2(u[1], u[0]))
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    return cv2.warpAffine(pre, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def band_rows(rot, period_px, frac=0.5):
    """Baris-baris yang berisi garis periodik (energi tinggi pada frekuensi 1/P)."""
    n = rot.shape[1]
    k = np.exp(-2j * np.pi * np.arange(n) / period_px) * np.hanning(n)
    mag = np.abs(rot @ k)
    sm = np.convolve(mag, np.ones(9) / 9, mode="same")
    rows = np.where(sm > frac * sm.max())[0]
    return int(rows.min()), int(rows.max()) + 1


def fine_phase(pre, u, period_px, rows):
    """Fase (radian) pola periodik sepanjang u di pita baris 'rows'."""
    rot = _rotate_to_u(pre, u)
    prof = rot[rows[0]:rows[1], :].mean(axis=0)
    prof = prof - prof.mean()
    n = len(prof)
    z = np.sum(prof * np.hanning(n) * np.exp(-2j * np.pi * np.arange(n) / period_px))
    return float(np.angle(z))


def coarse_position(pre, templ):
    """Posisi (x, y) piksel template di citra, dengan penyempurnaan parabola sub-piksel."""
    res = cv2.matchTemplate(pre.astype(np.float32), templ.astype(np.float32), cv2.TM_CCOEFF_NORMED)
    _, score, _, (x, y) = cv2.minMaxLoc(res)

    def sub(a, b, c):
        d = a - 2 * b + c
        return 0.0 if abs(d) < 1e-9 else 0.5 * (a - c) / d
    dx = sub(res[y, x - 1], res[y, x], res[y, x + 1]) if 0 < x < res.shape[1] - 1 else 0.0
    dy = sub(res[y - 1, x], res[y, x], res[y + 1, x]) if 0 < y < res.shape[0] - 1 else 0.0
    return np.array([x + dx, y + dy]), float(score)


class Tracker:
    """Mengukur pergeseran (piksel, sepanjang u) relatif terhadap citra referensi."""
    def __init__(self, ref_bgr, roi, period_px, u):
        self.period, self.u = period_px, np.asarray(u, float)
        pre = preprocess(ref_bgr)
        x, y, w, h = roi
        self.templ = pre[y:y + h, x:x + w].copy()
        self.rows = band_rows(_rotate_to_u(pre, self.u), period_px)
        self.p0, _ = coarse_position(pre, self.templ)
        self.phi0 = fine_phase(pre, self.u, period_px, self.rows)

    def displacement(self, img_bgr):
        pre = preprocess(img_bgr)
        p, score = coarse_position(pre, self.templ)
        d_coarse = float(np.dot(p - self.p0, self.u))
        dphi = fine_phase(pre, self.u, self.period, self.rows) - self.phi0
        d_fine = -dphi / (2 * np.pi) * self.period          # modulo P
        k = round((d_coarse - d_fine) / self.period)
        d = k * self.period + d_fine
        resid = abs(d - d_coarse)
        valid = int(score > 0.6 and resid < 0.4 * self.period)
        return d, d_coarse, score, valid


# ======================================================================= selftest
def selftest(a):
    img = cv2.imread(a.image)
    if img is None:
        sys.exit(f"Tidak bisa membaca {a.image}")
    roi = tuple(int(v) for v in a.roi.split(","))
    period, u = ruler_geometry_near(preprocess(img), roi)
    tr = Tracker(img, roi, period, u)
    print(f"Periode garis kecil = {period:.3f} px | arah penggaris u = ({u[0]:.3f}, {u[1]:.3f}) | "
          f"pita baris = {tr.rows}")
    errs = []
    for s in [-31.7, -12.3, -3.1, 0.0, 2.4, 7.77, 15.2, 26.9, 41.3]:
        M = np.float32([[1, 0, s * u[0]], [0, 1, s * u[1]]])
        sh = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_REFLECT)
        d, dc, sc, ok = tr.displacement(sh)
        errs.append(d - s)
        print(f"  geser {s:7.2f} px -> ukur {d:7.3f} (kasar {dc:7.2f}, skor {sc:.2f}, valid {ok}) "
              f"error {d - s:+.3f} px")
    e = np.abs(errs)
    print(f"Error rata-rata {e.mean():.3f} px, maks {e.max():.3f} px")
    if a.minor_um:
        print(f"  ≈ {e.mean() * a.minor_um / period:.2f} µm rata-rata (dengan {a.minor_um} µm/garis)")
    sys.exit(0 if e.max() < 0.5 else 1)


# ======================================================================= calibrate & run
def calib_path(axis):
    return os.path.join(LOG_DIR, f"calib_{axis}.json")


def calibrate(a):
    ensure_exclusive_hardware()
    motor = open_motor()   # menolak mode simulasi: datanya tidak sah untuk laporan
    cam = open_camera()
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        img0 = fresh_frame(cam, settle_s=1.0)
        prev = os.path.join(LOG_DIR, f"calib_{a.axis}_preview.png")
        _save_preview(img0, prev)
        print(f"Preview dengan grid koordinat: {prev}  (pakai untuk memilih --roi angka skala)")
        if not a.roi:
            print("Buka preview, pilih kotak di sekitar SATU angka skala yang menempel pada penggaris,")
            print("lalu ulangi perintah ini dengan --roi x,y,w,h.")
            return
        roi = tuple(int(v) for v in a.roi.split(","))
        period, u = ruler_geometry_near(preprocess(img0), roi)
        um_per_px = a.minor_um / period
        h, w = img0.shape[:2]
        print(f"Periode garis kecil {period:.3f} px  ->  {um_per_px:.4f} µm/px | "
              f"FOV ≈ {w * um_per_px / 1000:.3f} × {h * um_per_px / 1000:.3f} mm | "
              f"arah penggaris u = ({u[0]:.3f}, {u[1]:.3f})")
        tr = Tracker(img0, roi, period, u)
        # arah gerak sumbu meja di citra: gerak +step lalu ukur
        pos = motor.get_status()
        start = float(pos.get(a.axis, 0.0))
        motor.move_xyz(**{a.axis.lower(): round(start + a.step, 4)}, feedrate=a.feed)
        wait_idle(motor)
        d, dc, sc, ok = tr.displacement(fresh_frame(cam, settle_s=a.settle_ms / 1000))
        motor.move_xyz(**{a.axis.lower(): round(start, 4)}, feedrate=a.feed)
        wait_idle(motor)
        sign = 1 if d > 0 else -1
        motion_um_per_px = a.step * 1000 / abs(d) if abs(d) > 1e-6 else float("nan")
        cal = {"axis": a.axis, "period_px": period, "u": u.tolist(), "um_per_px": um_per_px,
               "minor_um": a.minor_um, "roi": roi, "sign": sign, "step_mm": a.step,
               "motion_check_um_per_px": motion_um_per_px, "score": sc, "valid": ok,
               "image_size": [w, h], "t_ns": time.time_ns()}
        cv2.imwrite(os.path.join(LOG_DIR, f"calib_{a.axis}_ref.png"), img0)
        with open(calib_path(a.axis), "w") as f:
            json.dump(cal, f, indent=2)
        print(json.dumps(cal, indent=2))
        print(f"Cek silang: skala dari gerak motor = {motion_um_per_px:.4f} µm/px "
              f"(dari penggaris {um_per_px:.4f}). Selisih besar = cek langkah/steps-per-mm.")
    finally:
        cam.close()
        motor.close()


def _save_preview(img, path, step=50):
    p = img.copy()
    for x in range(0, p.shape[1], step):
        cv2.line(p, (x, 0), (x, p.shape[0]), (0, 0, 255) if x % 250 == 0 else (0, 255, 255), 1)
        if x % 250 == 0:
            cv2.putText(p, str(x), (x + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    for y in range(0, p.shape[0], step):
        cv2.line(p, (0, y), (p.shape[1], y), (0, 0, 255) if y % 250 == 0 else (0, 255, 255), 1)
        if y % 250 == 0:
            cv2.putText(p, str(y), (2, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    cv2.imwrite(path, p)


def run(a):
    if not os.path.exists(calib_path(a.axis)):
        sys.exit(f"Belum ada {calib_path(a.axis)}. Jalankan 'calibrate' dulu.")
    cal = json.load(open(calib_path(a.axis)))
    targets = [float(t) for t in a.targets.split(",")]
    span_mm = max(targets) - min(targets)
    w = cal["image_size"][0]
    fov_mm = w * cal["um_per_px"] / 1000
    if span_mm > 0.7 * fov_mm:
        sys.exit(f"Rentang target {span_mm:.3f} mm terlalu besar untuk FOV {fov_mm:.3f} mm "
                 f"(maks ±{0.7 * fov_mm:.3f} mm). Perkecil --targets.")
    ensure_exclusive_hardware()
    motor = open_motor()   # menolak mode simulasi: datanya tidak sah untuk laporan
    cam = open_camera()
    ax = a.axis.lower()
    try:
        origin = float(motor.get_status().get(a.axis, 0.0))
        ref = cv2.imread(os.path.join(LOG_DIR, f"calib_{a.axis}_ref.png"))
        tr = Tracker(ref, tuple(cal["roi"]), cal["period_px"], cal["u"])
        img_dir = os.path.join(LOG_DIR, "positioning_imgs", a.axis)
        for run_i in range(1, a.runs + 1):
            os.makedirs(os.path.join(img_dir, str(run_i)), exist_ok=True)
            # pendekatan konsisten: mulai dari sebelum target pertama
            motor.move_xyz(**{ax: round(origin + min(targets) - a.approach, 4)}, feedrate=a.feed)
            wait_idle(motor)
            base = None
            for direction, seq in (("fwd", targets), ("bwd", list(reversed(targets)))):
                if direction == "bwd":   # lewati target terjauh sedikit lalu kembali
                    motor.move_xyz(**{ax: round(origin + max(targets) + a.approach, 4)}, feedrate=a.feed)
                    wait_idle(motor)
                for t in seq:
                    motor.move_xyz(**{ax: round(origin + t, 4)}, feedrate=a.feed)
                    wait_idle(motor)
                    img = fresh_frame(cam, settle_s=a.settle_ms / 1000)
                    d_px, d_c, score, valid = tr.displacement(img)
                    mm = cal["sign"] * d_px * cal["um_per_px"] / 1000
                    if base is None:
                        base = mm - t            # posisi terukur relatif target pertama run ini
                    measured = mm - base
                    cv2.imwrite(os.path.join(img_dir, str(run_i), f"{direction}_{t:.3f}.png"), img)
                    append_row(LOG_DIR, "positioning", axis=a.axis, target_mm=t, run=run_i, direction=direction,
                               measured_mm=round(measured, 5), valid=valid, score=round(score, 3))
                    print(f"run {run_i} {direction} target {t:.3f} mm -> terukur {measured:.4f} mm "
                          f"(dev {1000 * (measured - t):+.1f} µm, valid {valid})")
        motor.move_xyz(**{ax: round(origin, 4)}, feedrate=a.feed)
        wait_idle(motor)
    finally:
        cam.close()
        motor.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate")
    c.add_argument("--axis", choices=["X", "Y"], required=True)
    c.add_argument("--minor-um", type=float, default=10.0, help="jarak garis kecil (µm), cek label slide")
    c.add_argument("--roi", help="x,y,w,h kotak penanda unik (angka skala) di citra")
    c.add_argument("--step", type=float, default=0.1, help="gerak uji arah sumbu (mm)")
    c.add_argument("--feed", type=float, default=100)
    c.add_argument("--settle-ms", type=int, default=800)
    r = sub.add_parser("run")
    r.add_argument("--axis", choices=["X", "Y"], required=True)
    r.add_argument("--targets", default="0,0.13,0.27,0.38,0.51")
    r.add_argument("--runs", type=int, default=5)
    r.add_argument("--approach", type=float, default=0.1, help="jarak pendekatan sebelum target (mm)")
    r.add_argument("--feed", type=float, default=100)
    r.add_argument("--settle-ms", type=int, default=800)
    s = sub.add_parser("selftest")
    s.add_argument("--image", required=True)
    s.add_argument("--roi", required=True)
    s.add_argument("--minor-um", type=float, default=10.0)
    a = ap.parse_args()
    {"calibrate": calibrate, "run": run, "selftest": selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
