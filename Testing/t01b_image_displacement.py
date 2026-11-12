#!/usr/bin/env python3
"""Akurasi dan repeatability gerak CNC dari pergeseran citra. Jalankan di Jetson saat main.py mati.

Pengganti t01 bila stage micrometer tidak tersedia saat uji. Sampel bertekstur (mis. koloni
pada agar) dipakai sebagai penanda; pergeseran citra diukur dengan phase correlation, lalu
diubah ke mm memakai skala kamera yang diukur dari foto slide kalibrasi (--px-per-mm, untuk
lebar frame --scale-width piksel).

Tiap ulangan: ke titik A (selalu didekati dari arah yang sama), potret referensi, gerak +d,
potret, kembali ke A, potret. Dari situ diperoleh:
  - galat posisi      = pergeseran terukur - d
  - galat kembali     = pergeseran citra A_kembali terhadap A_awal (repeatability)
Keluaran: positioning_img.csv (axis,distance_mm,rep,measured_mm,error_mm,return_error_mm,response).
"""
import argparse
import time

import cv2
import numpy as np

from common import DEFAULT_LOG_DIR, append_row, ensure_exclusive_hardware, fresh_frame, open_camera, open_motor, wait_idle


def gray(frame):
    return frame[:, :, 1].astype(np.float32)


def averaged(camera, n=3):
    return np.mean([gray(fresh_frame(camera)) for _ in range(n)], axis=0)


def shift_px(ref, img, window):
    (dx, dy), response = cv2.phaseCorrelate(ref, img, window)
    return dx, dy, response


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--px-per-mm", type=float, default=128.8, help="skala dari foto slide kalibrasi")
    parser.add_argument("--scale-width", type=int, default=900, help="lebar foto saat skala diukur")
    parser.add_argument("--x-distances", default="0.5,1,2,3")
    parser.add_argument("--y-distances", default="0.5,1,1.5")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--feed", type=float, default=250.0)
    parser.add_argument("--settle", type=float, default=0.5)
    parser.add_argument("--approach", type=float, default=0.3, help="mm, mendekati A dari arah yang sama")
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    args = parser.parse_args()

    ensure_exclusive_hardware()
    motor = open_motor()
    camera = open_camera()
    motor.send_command("G90")
    start = motor.get_status()
    x0, y0 = start["X"], start["Y"]
    probe = averaged(camera, 1)
    px_per_mm = args.px_per_mm * probe.shape[1] / args.scale_width
    window = cv2.createHanningWindow(probe.shape[::-1], cv2.CV_32F)
    print(f"Titik A: X{x0:.3f} Y{y0:.3f}; frame {probe.shape[1]}x{probe.shape[0]}, {px_per_mm:.2f} px/mm")

    def goto(x, y):
        motor.send_command(f"G1 X{x:.3f} Y{y:.3f} F{args.feed}")
        wait_idle(motor)
        time.sleep(args.settle)

    def goto_a():
        # Selalu dekati A dari sisi negatif supaya backlash tidak bercampur ke galat.
        goto(x0 - args.approach, y0 - args.approach)
        goto(x0, y0)

    direction = {}
    try:
        for axis, distances in (("X", args.x_distances), ("Y", args.y_distances)):
            for d in sorted(float(v) for v in distances.split(",")):
                errs = []
                for rep in range(args.reps):
                    goto_a()
                    ref = averaged(camera)
                    goto(x0 + d, y0) if axis == "X" else goto(x0, y0 + d)
                    moved = averaged(camera)
                    goto_a()
                    back = averaged(camera)
                    dx, dy, resp = shift_px(ref, moved, window)
                    bx, by, _ = shift_px(ref, back, window)
                    # Kamera bisa terpasang miring terhadap sumbu meja: pakai panjang vektor pergeseran.
                    # phaseCorrelate hanya mengukur sampai setengah frame; geser yang lebih jauh
                    # "terlipat" dan dibuka memakai arah dari gerak terkecil sumbu yang sama.
                    if axis in direction:
                        ux, uy = direction[axis]
                        dx = min((dx, dx - ref.shape[1], dx + ref.shape[1]), key=lambda c: abs(c - ux * d))
                        dy = min((dy, dy - ref.shape[0], dy + ref.shape[0]), key=lambda c: abs(c - uy * d))
                    else:
                        direction[axis] = (dx / d, dy / d)
                    measured = float(np.hypot(dx, dy)) / px_per_mm
                    ret = float(np.hypot(bx, by)) / px_per_mm
                    err = measured - d
                    errs.append(err)
                    append_row(args.log_dir, "positioning_img", axis=axis, distance_mm=d, rep=rep,
                               shift_x_px=round(dx, 2), shift_y_px=round(dy, 2),
                               measured_mm=round(measured, 4), error_mm=round(err, 4),
                               return_error_mm=round(ret, 4), response=round(resp, 3))
                    print(f"{axis} {d:.2f} mm ulang {rep}: terukur {measured:.3f} mm "
                          f"(dx {dx:.1f}, dy {dy:.1f} px, respons {resp:.2f}), kembali {ret * 1000:.0f} um", flush=True)
                print(f"== {axis} {d:.2f} mm: galat rata-rata {np.mean(errs) * 1000:.0f} um, "
                      f"sd {np.std(errs) * 1000:.0f} um", flush=True)
    finally:
        goto(x0, y0)
        camera.close()
        motor.close()


if __name__ == "__main__":
    main()
