#!/usr/bin/env python3
"""Latensi perintah dari klien. Jalankan di LAPTOP, main.py di Jetson jalan dengan TEST_LOGGING=1.

Env: BASE_URL, TEST_USER, TEST_PASS. Keluaran: cmd_client.csv (cmd_id,path,t_sent_ns).
Pasangannya cmd.csv ditulis Jetson, lalu keduanya digabung lewat cmd_id oleh analyze_hw.py.

Gerak memakai "G1 X0.05 F250" (jog relatif di Jetson). Gabungan "G91 G1 ..." ditolak server,
karena membuat GRBL diam-diam tetap di mode relatif.
"""
import argparse
import asyncio
import json
import time

from common import DEFAULT_LOG_DIR, append_row, client_ws_url, connect_as_pilot, login


async def run(args):
    base, token = login()
    ws = await connect_as_pilot(client_ws_url(base, token))
    async with ws:
        rejected = {}

        async def reader():
            # Catat penolakan server; tanpa ini perintah yang ditolak hilang tanpa jejak.
            while True:
                m = json.loads(await ws.recv())
                if m.get("event") == "COMMAND_REJECTED":
                    rejected[m.get("cmd_id")] = m.get("reason")
                elif m.get("event") == "ROLE_ASSIGNED" and m.get("role") != "PILOT":
                    print(f"Peringatan: peran berubah menjadi {m.get('role')}")

        reader_task = asyncio.create_task(reader())
        start_id = int(time.time()) * 1000  # unik antar-sesi supaya penggabungan cmd_id tidak bentrok
        for i in range(args.count):
            value = args.step if i % 2 == 0 else -args.step
            t_sent = time.time_ns()
            cmd_id = start_id + i
            await ws.send(json.dumps({
                "action": "MOVE_MOTOR", "gcode": f"G1 X{value:.3f} F250", "cmd_id": cmd_id, "t_sent": t_sent}))
            append_row(args.log_dir, "cmd_client", cmd_id=cmd_id, path=args.path, t_sent_ns=t_sent)
            await asyncio.sleep(args.interval)
        await asyncio.sleep(2)
        reader_task.cancel()
    if rejected:
        reasons = {}
        for r in rejected.values():
            reasons[r] = reasons.get(r, 0) + 1
        print(f"{len(rejected)} perintah ditolak server: {reasons}")
    print(f"{args.count} perintah terkirim lewat jalur {args.path}. Salin cmd.csv dari Jetson ke {args.log_dir}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--path", required=True, choices=["cloud", "lan"])
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--interval", type=float, default=0.6, help="detik, harus di atas rate-limit 0,5")
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
