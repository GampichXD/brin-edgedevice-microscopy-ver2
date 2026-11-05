#!/usr/bin/env python3
"""Pembuktian jeda stabilisasi (A14). Jalankan di Jetson saat main.py mati.

Untuk tiap jeda, ambil citra setelah gerak dalam baris dan setelah pindah baris, lalu hitung
ketajaman variance of Laplacian (Pech-Pacheco dkk., ICPR 2000). Pilih jeda saat ketajaman
sudah mendatar. Keluaran: settle.csv (move_type,delay_s,rep,sharpness).
"""
import argparse
import time

import cv2

from common import DEFAULT_LOG_DIR, append_row, ensure_exclusive_hardware, open_camera, open_motor, wait_idle


def sharpness(frame):
    # Kanal hijau: SNR tertinggi pada kamera ini (lihat B9).
    return float(cv2.Laplacian(frame[:, :, 1], cv2.CV_64F).var())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--delays", default="0,0.25,0.5,1.0,1.5")
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--step", type=float, default=0.5, help="langkah dalam mm")
    parser.add_argument("--feed", type=float, default=250.0)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    args = parser.parse_args()

    ensure_exclusive_hardware()
    motor = open_motor()
    camera = open_camera()
    motor.send_command("G90")
    pos = motor.get_status()
    x0, y0 = pos["X"], pos["Y"]
    try:
        for delay in (float(d) for d in args.delays.split(",")):
            for move_type in ("in_row", "row_change"):
                values = []
                for rep in range(args.reps):
                    motor.send_command(f"G1 X{x0:.3f} Y{y0:.3f} F{args.feed}")
                    wait_idle(motor)
                    time.sleep(1.5)
                    if move_type == "in_row":
                        target = f"G1 X{x0 + args.step:.3f} F{args.feed}"
                    else:  # pindah baris: Y bergerak dan arah X berbalik
                        target = f"G1 X{x0 - args.step:.3f} Y{y0 + args.step:.3f} F{args.feed}"
                    motor.send_command(target)
                    wait_idle(motor)
                    time.sleep(delay)
                    value = sharpness(camera.capture_frame())
                    append_row(args.log_dir, "settle", move_type=move_type, delay_s=delay, rep=rep,
                               sharpness=round(value, 3))
                    values.append(value)
                print(f"jeda {delay:.2f} s, {move_type}: rata-rata {sum(values) / len(values):.1f}")
    finally:
        motor.send_command(f"G1 X{x0:.3f} Y{y0:.3f} F{args.feed}")
        camera.close()
        motor.close()


if __name__ == "__main__":
    main()
