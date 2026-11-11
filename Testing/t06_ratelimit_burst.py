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
    """Kirim tepat `rate` request per detik secara paralel. Menunggu balasan satu per satu
    membuat laju turun ke ~2-3/detik (waktu bolak-balik lewat Cloudflare), sehingga bukan burst."""
    from concurrent.futures import ThreadPoolExecutor

    session = requests.Session()

    def one(i, t_send):
        value = args.step if i % 2 == 0 else -args.step
        try:
            res = session.post(f"{base}/api/hardware/motor/move", headers=headers, timeout=15, json={
                "axis": "X", "value": value, "feed_rate": 250.0, "unit": "mm"})
            return t_send, res.status_code
        except requests.RequestException:
            return t_send, 0

    futures = []
    with ThreadPoolExecutor(max_workers=32) as pool:
        t0 = time.monotonic()
        for i in range(int(args.rate * args.duration)):
            delay = t0 + i / args.rate - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            futures.append(pool.submit(one, i, time.time_ns()))
        results = [f.result() for f in futures]

    accepted = rejected = 0
    other = {}
    for t_send, code in results:
        # 429 = ditolak rate-limit, 200 = diterima. Kode lain (403 bukan Pilot, 409 CNC offline,
        # 0 gagal jaringan) bukan hasil guard rate-limit, dicatat apa adanya dan tidak dihitung.
        if code == 429:
            rejected += 1
            result = "REJECTED"
        elif code == 200:
            accepted += 1
            result = "ACCEPTED"
        else:
            result = f"HTTP_{code}"
            other[code] = other.get(code, 0) + 1
        append_row(args.log_dir, "ratelimit", t_send_ns=t_send, channel="rest", result=result)
    print(f"REST: {accepted} diterima, {rejected} ditolak" + (f", kode lain {other}" if other else ""))


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

    async def rest_as_pilot():
        # REST motor/move mewajibkan kursi Pilot: pegang WebSocket Pilot selama burst REST.
        ws = await connect_as_pilot(client_ws_url(base, token))
        async with ws:
            await asyncio.get_running_loop().run_in_executor(None, burst_rest, base, auth_headers(token), args)

    asyncio.run(rest_as_pilot())
    time.sleep(1)
    asyncio.run(burst_ws(base, token, args))


if __name__ == "__main__":
    main()
