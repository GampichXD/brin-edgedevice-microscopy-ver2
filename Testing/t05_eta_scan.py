#!/usr/bin/env python3
"""Validasi ETA grid scan, persamaan (3)-(4). Jalankan di LAPTOP dengan akun Pilot atau Admin.

Env: BASE_URL, TEST_USER, TEST_PASS. Keluaran: eta.csv (grid,run,n_tiles,predicted_s,actual_s).
Uji termal dijalankan bersamaan dengan grid 10x10: telemetry.csv terekam otomatis kalau
main.py jalan dengan TEST_LOGGING=1.
"""
import argparse
import asyncio
import json
import math
import sys
import time

import requests

from common import DEFAULT_LOG_DIR, append_row, auth_headers, client_ws_url, connect_as_pilot, heartbeat_loop, login

# ── Salinan dari Software/backend/app/services/scan_planner.py. Ubah keduanya bersamaan. ──
MIN_MOVE_DELAY_S = 0.5
ROW_CHANGE_SETTLE_S = 1.0
CAPTURE_OVERHEAD_S = 0.43


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


def predict_seconds(rows, cols, step, feed_mm_min, delay_ms, capture_s=CAPTURE_OVERHEAD_S):
    total, last_x, last_y = 0.0, 0.0, 0.0
    for tile in plan_serpentine(rows, cols, 0.0, 0.0, step, step):
        distance = math.hypot(tile["x"] - last_x, tile["y"] - last_y)
        total += compute_delay(distance, feed_mm_min / 60.0, delay_ms, tile["row_change"]) + capture_s
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


class PilotLink:
    """Satu koneksi Pilot selama seluruh uji: scan/grid dan gerak kembali mewajibkan kursi Pilot."""

    def __init__(self, ws):
        self.ws = ws
        self.telemetry = None
        self.rejected = None

    async def reader(self):
        while True:
            m = json.loads(await self.ws.recv())
            if m.get("event") == "TELEMETRY_DATA" and m.get("position"):
                self.telemetry = m
            elif m.get("event") == "COMMAND_REJECTED":
                self.rejected = m.get("reason")
            elif m.get("event") == "ROLE_ASSIGNED" and m.get("role") != "PILOT":
                print(f"Peringatan: peran berubah menjadi {m.get('role')}")

    async def position(self, timeout=15.0):
        deadline = time.monotonic() + timeout
        while self.telemetry is None and time.monotonic() < deadline:
            await asyncio.sleep(0.2)
        if self.telemetry is None:
            sys.exit("Posisi motor tidak terbaca dari telemetri.")
        p = self.telemetry["position"]
        return {"X": float(p["X"]), "Y": float(p["Y"])}

    async def return_to(self, start, timeout=90.0):
        """Kembali ke titik awal dengan gerak absolut biasa (mesin tanpa limit switch homing)."""
        self.rejected = None
        await self.ws.send(json.dumps({"action": "MOVE_MOTOR", "gcode": "G90"}))
        await asyncio.sleep(0.4)
        await self.ws.send(json.dumps({"action": "MOVE_MOTOR",
                                       "gcode": f"G1 X{start['X']:.3f} Y{start['Y']:.3f} F250"}))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            await asyncio.sleep(0.3)
            if self.rejected:
                sys.exit(f"Gerak kembali ditolak: {self.rejected}")
            t = self.telemetry
            if (t and t.get("status") == "Idle" and abs(float(t["position"]["X"]) - start["X"]) < 0.01
                    and abs(float(t["position"]["Y"]) - start["Y"]) < 0.01):
                return
        sys.exit("Motor tidak kembali ke titik awal dalam batas waktu.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grids", default="3x3,5x5,10x10")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--step", type=float, default=0.5)
    parser.add_argument("--delay-ms", type=int, default=500)
    parser.add_argument("--feed-rate", type=float, default=250.0, help="mm/menit, samakan dengan setting feedRate")
    parser.add_argument("--home-wait", type=float, default=15.0, help="detik menunggu homing selesai antar-run")
    parser.add_argument("--no-home", action="store_true",
                        help="mesin tanpa limit switch: kembali ke titik awal dengan G90/G1, bukan $H")
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    args = parser.parse_args()
    asyncio.run(run_all(args))


async def run_all(args):
    base, token = login()
    headers = auth_headers(token)
    loop = asyncio.get_running_loop()
    ws = await connect_as_pilot(client_ws_url(base, token))
    async with ws:
        link = PilotLink(ws)
        tasks = [asyncio.create_task(link.reader()), asyncio.create_task(heartbeat_loop(ws))]
        start = await link.position()
        print(f"Titik awal: X{start['X']:.3f} Y{start['Y']:.3f}")
        for grid in args.grids.split(","):
            rows, cols = (int(v) for v in grid.lower().split("x"))
            predicted = predict_seconds(rows, cols, args.step, args.feed_rate, args.delay_ms)
            for run in range(1, args.runs + 1):
                actual = await loop.run_in_executor(None, run_scan, base, headers, rows, cols, args)
                append_row(args.log_dir, "eta", grid=grid, run=run, n_tiles=rows * cols,
                           predicted_s=round(predicted, 2), actual_s=round(actual, 2))
                print(f"{grid} run {run}: prediksi {predicted:.1f} s, aktual {actual:.1f} s", flush=True)
                if args.no_home:
                    await link.return_to(start)
                    await asyncio.sleep(1.0)
                else:
                    await loop.run_in_executor(None, lambda: requests.post(
                        f"{base}/api/hardware/motor/home", headers=headers, timeout=15))
                    await asyncio.sleep(args.home_wait)
        for t in tasks:
            t.cancel()


if __name__ == "__main__":
    main()
