#!/usr/bin/env python3
"""Latensi video dan FPS. Jalankan di LAPTOP.

Env: BASE_URL, TEST_USER, TEST_PASS. Keluaran: video.csv (frame_id,t_cap_ns,t_recv_ns,bytes).
Jalankan untuk cloud dan LAN dengan --log-dir berbeda (test_logs/cloud, test_logs/lan).
"""
import argparse
import asyncio
import json
import time

import websockets

from common import DEFAULT_LOG_DIR, append_row, client_ws_url, heartbeat_loop, login, percentile


async def run(args):
    base, token = login()
    latencies, frames = [], 0
    deadline = time.monotonic() + args.duration
    async with websockets.connect(client_ws_url(base, token), max_size=None) as ws:
        beat = asyncio.create_task(heartbeat_loop(ws))
        await ws.send(json.dumps({"action": "START_STREAM"}))
        try:
            while time.monotonic() < deadline:
                try:
                    message = await asyncio.wait_for(ws.recv(), deadline - time.monotonic())
                except asyncio.TimeoutError:
                    break
                t_recv = time.time_ns()
                if '"STREAM_DATA"' not in message[:40]:
                    continue
                data = json.loads(message)
                frames += 1
                t_cap = data.get("t_cap")
                append_row(args.log_dir, "video", frame_id=data.get("frame_id", ""), t_cap_ns=t_cap or "",
                           t_recv_ns=t_recv, bytes=len(message))
                if t_cap:
                    latencies.append((t_recv - t_cap) / 1e6)
        finally:
            beat.cancel()
            await ws.send(json.dumps({"action": "STOP_STREAM"}))
    print(f"{frames} frame, {frames / args.duration:.2f} FPS")
    print(f"Latensi: p50={percentile(latencies, .5):.1f} ms, p95={percentile(latencies, .95):.1f} ms")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--duration", type=float, default=600)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
