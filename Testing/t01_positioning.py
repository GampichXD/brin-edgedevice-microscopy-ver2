#!/usr/bin/env python3
"""Akurasi dan repeatability XY dengan stage micrometer (adaptasi ISO 230-2).
Jalankan di Jetson saat main.py mati. Metode ada di positioning_math.py.

1. Simpan citra acuan untuk memilih ROI:
     python3 Testing/t01_positioning.py --axis X --snapshot
2. Kalibrasi (sekali per sumbu). --marker = kotak penanda unik, misalnya satu angka skala.
   --band = pita garis skala, tanpa goresan yang tidak ikut bergerak saat meja bergerak:
     python3 Testing/t01_positioning.py --axis X --calibrate --marker x,y,w,h --band x,y,w,h \
         --period-um 100 --minor-um 10
3. Uji:
     python3 Testing/t01_positioning.py --axis X --targets 0,0.13,0.27,0.38,0.51 --runs 5
     python3 Testing/t01_positioning.py --axis Y --targets 0,0.07,0.15,0.22,0.29 --runs 5

Pastikan nilai --period-um dan --minor-um dari tulisan di label slide. Target sengaja tidak
kelipatan periode. Keluaran: positioning.csv (axis,target_mm,run,direction,measured_mm,valid,score)
dan citra di test_logs/positioning_imgs/<axis>/<run>/.
"""
import argparse
import json
import math
import os
import sys
import time

import cv2

import positioning_math as pm
from common import DEFAULT_LOG_DIR, append_row, ensure_exclusive_hardware, open_camera, open_motor, wait_idle

CALIB_MOVE_MM = 0.1
FEED = 250.0
START_MARGIN_MM = 0.1


def parse_box(text):
    values = [int(v) for v in text.split(",")]
    if len(values) != 4:
        sys.exit("Format kotak: x,y,w,h")
    return values


def move_abs(motor, axis, value, settle_ms):
    # Sengaja lewat send_command absolut, bukan jog: kompensasi backlash tidak ikut,
    # supaya yang terukur adalah perilaku mekanik apa adanya.
    motor.send_command(f"G1 {axis}{value:.4f} F{FEED}")
    if not wait_idle(motor):
        sys.exit("GRBL tidak kembali Idle.")
    time.sleep(settle_ms / 1000.0)


def capture_gray(camera):
    frame = camera.capture_frame()
    if frame is None:
        sys.exit("Frame kamera kosong.")
    return frame, pm.preprocess(frame)


def measure(gray, template, calib):
    """Posisi sepanjang sumbu gerak (mm), plus status valid dan skor template."""
    axis = calib["axis"]
    mx, my, score = pm.locate_marker(gray, template)
    profile = pm.line_profile(gray, calib["band"], axis)
    fine = pm.fine_phase_px(profile, calib["minor_px"])
    coarse_along = mx if axis == "X" else my
    position_px, valid = pm.combine(coarse_along, fine, calib["minor_px"])
    # Koreksi sudut antara garis skala dan arah gerak meja, dan arah tercermin.
    direction = calib["direction"][0] if axis == "X" else calib["direction"][1]
    position_mm = position_px * calib["um_per_px"] / 1000.0 / direction
    return position_mm, valid and score >= 0.5, score


def calib_path(log_dir, axis):
    return os.path.join(log_dir, f"calib_{axis}.json")


def do_calibrate(args, motor, camera):
    if not args.marker or not args.band:
        sys.exit("--calibrate butuh --marker dan --band. Pakai --snapshot untuk melihat citra.")
    frame, gray = capture_gray(camera)
    x, y, w, h = parse_box(args.marker)
    template = gray[y:y + h, x:x + w].copy()
    band = parse_box(args.band)

    profile = pm.line_profile(gray, band, args.axis)
    minor_px = pm.estimate_period_px(profile)
    um_per_px = args.minor_um / minor_px

    start = motor.get_status()[args.axis]
    x0, y0, _ = pm.locate_marker(gray, template)
    move_abs(motor, args.axis, start + CALIB_MOVE_MM, args.settle_ms)
    _, moved = capture_gray(camera)
    x1, y1, score = pm.locate_marker(moved, template)
    move_abs(motor, args.axis, start, args.settle_ms)
    if score < 0.5:
        sys.exit(f"Penanda tidak ketemu setelah bergerak (skor {score:.2f}). Pilih penanda lain.")
    direction = pm.motion_direction(x1 - x0, y1 - y0)

    calib = {
        "axis": args.axis, "minor_um": args.minor_um, "period_um": args.period_um,
        "minor_px": minor_px, "um_per_px": um_per_px, "band": band, "marker": [x, y, w, h],
        "direction": direction, "camera_angle_deg": math.degrees(math.atan2(direction[1], direction[0])),
        "frame_size": [gray.shape[1], gray.shape[0]],
    }
    os.makedirs(args.log_dir, exist_ok=True)
    with open(calib_path(args.log_dir, args.axis), "w") as f:
        json.dump(calib, f, indent=2)
    cv2.imwrite(os.path.join(args.log_dir, f"marker_{args.axis}.png"), template)

    expected = CALIB_MOVE_MM * 1000.0 / um_per_px
    actual = math.hypot(x1 - x0, y1 - y0)
    print(f"um_per_px={um_per_px:.4f}, garis kecil={minor_px:.2f} px, arah={direction}")
    print(f"Pemeriksaan skala: gerak {CALIB_MOVE_MM} mm mengharapkan {expected:.1f} px, terukur {actual:.1f} px")
    if abs(actual - expected) > 0.2 * expected:
        print("PERINGATAN: selisih lebih dari 20%. Periksa --minor-um dan --band.")


def do_test(args, motor, camera):
    path = calib_path(args.log_dir, args.axis)
    if not os.path.exists(path):
        sys.exit(f"Belum ada {path}. Jalankan --calibrate dulu.")
    with open(path) as f:
        calib = json.load(f)
    template = cv2.imread(os.path.join(args.log_dir, f"marker_{args.axis}.png"), cv2.IMREAD_GRAYSCALE)

    targets = sorted(float(t) for t in args.targets.split(","))
    _, gray = capture_gray(camera)
    mx, my, score = pm.locate_marker(gray, template)
    bad = pm.target_stays_in_frame((mx, my), template.shape[::-1], calib["frame_size"], calib["direction"],
                                   calib["um_per_px"], targets, START_MARGIN_MM)
    if bad:
        sys.exit(f"Target {bad} membuat penanda keluar bidang pandang. Pakai rentang lebih kecil.")

    base = motor.get_status()[args.axis]
    img_root = os.path.join(args.log_dir, "positioning_imgs", args.axis)
    rejected = 0
    try:
        for run in range(1, args.runs + 1):
            os.makedirs(os.path.join(img_root, str(run)), exist_ok=True)
            origin = None
            # Dua arah dengan pendekatan konsisten: maju dari bawah target terkecil, lalu mundur.
            move_abs(motor, args.axis, base + targets[0] - START_MARGIN_MM, args.settle_ms)
            for direction, sequence in (("up", targets), ("down", list(reversed(targets)))):
                if direction == "down":
                    move_abs(motor, args.axis, base + targets[-1] + START_MARGIN_MM, args.settle_ms)
                for target in sequence:
                    move_abs(motor, args.axis, base + target, args.settle_ms)
                    frame, gray = capture_gray(camera)
                    cv2.imwrite(os.path.join(img_root, str(run), f"{direction}_{target:.3f}.png"), frame)
                    position, valid, score = measure(gray, template, calib)
                    if origin is None:
                        origin = position - target
                    rejected += not valid
                    append_row(args.log_dir, "positioning", axis=args.axis, target_mm=target, run=run,
                               direction=direction, measured_mm=round(position - origin, 5),
                               valid=int(valid), score=round(score, 3))
            print(f"Run {run}/{args.runs} selesai")
    finally:
        move_abs(motor, args.axis, base, args.settle_ms)
    print(f"Selesai. Baris valid=0: {rejected}. Baris itu sebaiknya diperiksa, bukan langsung dibuang.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--axis", required=True, choices=["X", "Y"])
    parser.add_argument("--snapshot", action="store_true", help="simpan citra acuan lalu keluar")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--marker", help="x,y,w,h kotak penanda unik")
    parser.add_argument("--band", help="x,y,w,h pita garis skala")
    parser.add_argument("--period-um", type=float, default=100.0, help="jarak antar-angka")
    parser.add_argument("--minor-um", type=float, default=10.0, help="jarak garis kecil")
    parser.add_argument("--targets", default="0,0.13,0.27,0.38,0.51")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--settle-ms", type=int, default=800)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    args = parser.parse_args()

    ensure_exclusive_hardware()
    camera = open_camera()
    if args.snapshot:
        os.makedirs(args.log_dir, exist_ok=True)
        frame, gray = capture_gray(camera)
        cv2.imwrite(os.path.join(args.log_dir, f"snapshot_{args.axis}.png"), frame)
        cv2.imwrite(os.path.join(args.log_dir, f"snapshot_{args.axis}_pre.png"), gray)
        print(f"Tersimpan di {args.log_dir}. Pilih kotak --marker dan --band dari citra ini.")
        camera.close()
        return

    motor = open_motor()
    motor.send_command("G90")
    try:
        if args.calibrate:
            do_calibrate(args, motor, camera)
        else:
            do_test(args, motor, camera)
    finally:
        camera.close()
        motor.close()


if __name__ == "__main__":
    main()
