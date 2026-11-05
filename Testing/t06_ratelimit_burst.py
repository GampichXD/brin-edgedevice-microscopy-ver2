#!/usr/bin/env python3
"""Validasi rate-limit, persamaan (6). Jalankan di LAPTOP dengan akun yang sedang jadi Pilot.

Kirim 20 perintah per detik selama 10 detik lewat REST, lalu lewat WebSocket.
Env: BASE_URL, TEST_USER, TEST_PASS. Keluaran: ratelimit.csv (t_send_ns,channel,result).
"""
import argparse
import asyncio
import json
import time

import requests

from common import DEFAULT_LOG_DIR, append_row, auth_headers, client_ws_url, connect_as_pilot, login


def burst_rest(base, headers, args):
    accepted = rejected = 0
    for i in range(int(args.rate * args.duration)):
        t_send = time.time_ns()
        value = args.step if i % 2 == 0 else -args.step
        res = requests.post(f"{base}/api/hardware/motor/move", headers=headers, timeout=15, json={
            "axis": "X", "value": value, "feed_rate": 250.0, "unit": "mm"})
        # Yang diuji hanya guard rate-limit: 409 berarti lolos guard, tetapi CNC belum tersambung.
        limited = res.status_code == 429
        accepted += not limited
        rejected += limited
        append_row(args.log_dir, "ratelimit", t_send_ns=t_send, channel="rest",
                   result="REJECTED" if limited else "ACCEPTED")
        time.sleep(max(0.0, 1.0 / args.rate - (time.time_ns() - t_send) / 1e9))
    print(f"REST: {accepted} diterima, {rejected} ditolak")


async def burst_ws(base, token, args):
    sent, rejected_ids = [], set()
    ws = await connect_as_pilot(client_ws_url(base, token))
    async with ws:
        async def reader():
            while True:
                message = json.loads(await ws.recv())
                if message.get("event") == "COMMAND_REJECTED" and message.get("reason") == "RATE_LIMIT":
                    rejected_ids.add(message.get("cmd_id"))

        task = asyncio.create_task(reader())
        for i in range(int(args.rate * args.duration)):
            value = args.step if i % 2 == 0 else -args.step
            sent.append((i, time.time_ns()))
            await ws.send(json.dumps({"action": "MOVE_MOTOR", "gcode": f"G1 X{value:.3f} F250", "cmd_id": i}))
            await asyncio.sleep(1.0 / args.rate)
        await asyncio.sleep(2)  # tunggu balasan terakhir
        task.cancel()
    for cmd_id, t_send in sent:
        append_row(args.log_dir, "ratelimit", t_send_ns=t_send, channel="ws",
                   result="REJECTED" if cmd_id in rejected_ids else "ACCEPTED")
    print(f"WS: {len(sent) - len(rejected_ids)} diterima, {len(rejected_ids)} ditolak")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rate", type=float, default=20, help="perintah per detik")
    parser.add_argument("--duration", type=float, default=10, help="detik")
    parser.add_argument("--step", type=float, default=0.01, help="mm, bolak-balik")
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    args = parser.parse_args()

    base, token = login()
    burst_rest(base, auth_headers(token), args)
    time.sleep(1)
    asyncio.run(burst_ws(base, token, args))


if __name__ == "__main__":
    main()
