#!/usr/bin/env python3
"""Validasi ETA grid scan, persamaan (3)-(4). Jalankan di LAPTOP dengan akun Pilot atau Admin.

Env: BASE_URL, TEST_USER, TEST_PASS. Keluaran: eta.csv (grid,run,n_tiles,predicted_s,actual_s).
Uji termal dijalankan bersamaan dengan grid 10x10: telemetry.csv terekam otomatis kalau
main.py jalan dengan TEST_LOGGING=1.
"""
import argparse
import math
import sys
import time

import requests

from common import DEFAULT_LOG_DIR, append_row, auth_headers, login

# ── Salinan dari Software/backend/app/services/scan_planner.py. Ubah keduanya bersamaan. ──
MIN_MOVE_DELAY_S = 0.5
ROW_CHANGE_SETTLE_S = 1.0


def plan_serpentine(rows, cols, x0, y0, dx, dy):
    tiles = []
    for r in range(rows):
        for c in range(cols):
            actual_c = c if r % 2 == 0 else (cols - 1 - c)
            tiles.append({"x": x0 + actual_c * dx, "y": y0 + r * dy, "row_change": c == 0 and r > 0})
    return tiles


def compute_delay(distance_mm, feed_mm_per_s, delay_ms, row_change=False, row_settle_s=ROW_CHANGE_SETTLE_S):
    travel = distance_mm / feed_mm_per_s if distance_mm > 0 else 0.0
    return max(MIN_MOVE_DELAY_S, travel) + delay_ms / 1000.0 + (row_settle_s if row_change else 0.0)
# ─────────────────────────────────────────────────────────────────────────────────────


def predict_seconds(rows, cols, step, feed_mm_min, delay_ms):
    total, last_x, last_y = 0.0, 0.0, 0.0
    for tile in plan_serpentine(rows, cols, 0.0, 0.0, step, step):
        distance = math.hypot(tile["x"] - last_x, tile["y"] - last_y)
        total += compute_delay(distance, feed_mm_min / 60.0, delay_ms, tile["row_change"])
        last_x, last_y = tile["x"], tile["y"]
    return total


def run_scan(base, headers, rows, cols, args):
    res = requests.post(f"{base}/api/hardware/scan/grid", headers=headers, timeout=30, json={
        "columns": cols, "rows": rows, "step_x": args.step, "step_y": args.step,
        "delay_ms": args.delay_ms, "unit": "mm", "start_x": 0.0, "start_y": 0.0})
    if res.status_code != 200:
        sys.exit(f"Scan ditolak ({res.status_code}): {res.text[:200]}")
    t0 = time.monotonic()
    while True:
        time.sleep(1)
        status = requests.get(f"{base}/api/hardware/scan/status", headers=headers, timeout=15).json()
        if not status["running"]:
            if status.get("error") or status.get("cancelled"):
                sys.exit(f"Scan tidak selesai: {status.get('error') or 'dibatalkan'}")
            return time.monotonic() - t0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grids", default="3x3,5x5,10x10")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--step", type=float, default=0.5)
    parser.add_argument("--delay-ms", type=int, default=500)
    parser.add_argument("--feed-rate", type=float, default=250.0, help="mm/menit, samakan dengan setting feedRate")
    parser.add_argument("--home-wait", type=float, default=15.0, help="detik menunggu homing selesai antar-run")
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    args = parser.parse_args()

    base, token = login()
    headers = auth_headers(token)
    for grid in args.grids.split(","):
        rows, cols = (int(v) for v in grid.lower().split("x"))
        predicted = predict_seconds(rows, cols, args.step, args.feed_rate, args.delay_ms)
        for run in range(1, args.runs + 1):
            actual = run_scan(base, headers, rows, cols, args)
            append_row(args.log_dir, "eta", grid=grid, run=run, n_tiles=rows * cols,
                       predicted_s=round(predicted, 2), actual_s=round(actual, 2))
            print(f"{grid} run {run}: prediksi {predicted:.1f} s, aktual {actual:.1f} s")
            requests.post(f"{base}/api/hardware/motor/home", headers=headers, timeout=15)
            time.sleep(args.home_wait)


if __name__ == "__main__":
    main()
