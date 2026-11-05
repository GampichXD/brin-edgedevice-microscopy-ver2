#!/usr/bin/env python3
"""Latensi perintah lokal Jetson ke GRBL. Jalankan di Jetson saat main.py mati.

Keluaran: cmd_local.csv (cmd_id,t_sent_ns,t_ok_ns).
"""
import argparse
import time

from common import DEFAULT_LOG_DIR, append_row, ensure_exclusive_hardware, open_motor, percentile, wait_idle


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--axis", default="X", choices=["X", "Y"])
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    args = parser.parse_args()

    ensure_exclusive_hardware()
    motor = open_motor()
    latencies = []
    # G91 dikirim sebagai perintah tunggal supaya pelacakan mode di motor.py tetap benar.
    motor.send_command("G91")
    try:
        for i in range(args.count):
            value = args.step if i % 2 == 0 else -args.step
            t_sent = time.time_ns()
            motor.send_command(f"G1 {args.axis}{value:.3f} F250")
            t_ok = time.time_ns()
            append_row(args.log_dir, "cmd_local", cmd_id=i, t_sent_ns=t_sent, t_ok_ns=t_ok)
            latencies.append((t_ok - t_sent) / 1e6)
            wait_idle(motor, timeout=5)  # jeda sampai gerak selesai, agar antrean GRBL tidak ikut terukur
    finally:
        motor.send_command("G90")
        motor.close()
    print(f"{len(latencies)} perintah: p50={percentile(latencies, .5):.2f} ms, p95={percentile(latencies, .95):.2f} ms")


if __name__ == "__main__":
    main()
