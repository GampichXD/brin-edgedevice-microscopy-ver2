#!/usr/bin/env python3
"""Cek selisih jam terhadap server NTP, prasyarat uji latensi satu arah (D3, D4).

Jalankan di Jetson DAN di laptop dengan server NTP yang sama: selisih jam keduanya paling
besar sama dengan jumlah offset yang dilaporkan. Memakai NTP karena VPS tidak punya endpoint
yang mengembalikan waktu presisi. Exit 1 kalau offset melebihi batas.
"""
import argparse
import socket
import statistics
import struct
import sys
import time

from common import DEFAULT_LOG_DIR, append_row

NTP_EPOCH_OFFSET = 2208988800


def _to_seconds(sec, frac):
    return sec - NTP_EPOCH_OFFSET + frac / 2**32


def ntp_query(server, timeout=2.0):
    """Satu kueri SNTP. Mengembalikan (offset_ms, rtt_ms)."""
    packet = b"\x23" + 47 * b"\0"
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.settimeout(timeout)
        t1 = time.time()
        sock.sendto(packet, (server, 123))
        data, _ = sock.recvfrom(48)
        t4 = time.time()
    t2 = _to_seconds(*struct.unpack("!II", data[32:40]))
    t3 = _to_seconds(*struct.unpack("!II", data[40:48]))
    return ((t2 - t1) + (t3 - t4)) / 2 * 1000, ((t4 - t1) - (t3 - t2)) * 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", default="pool.ntp.org")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--max-offset-ms", type=float, default=10.0)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    args = parser.parse_args()

    host = socket.gethostname()
    offsets, rtts = [], []
    for _ in range(args.samples):
        try:
            offset, rtt = ntp_query(args.server)
        except OSError as e:
            print(f"Kueri gagal: {e}")
            continue
        offsets.append(offset)
        rtts.append(rtt)
        append_row(args.log_dir, "clock", t_ns=time.time_ns(), host=host,
                   offset_ms=round(offset, 3), rtt_ms=round(rtt, 3))
        time.sleep(0.2)
    if not offsets:
        sys.exit("Tidak ada balasan NTP. Periksa koneksi atau pakai --server lain.")

    offset = statistics.median(offsets)
    print(f"{host}: offset median {offset:.2f} ms, RTT median {statistics.median(rtts):.2f} ms ({len(offsets)} sampel)")
    if abs(offset) > args.max_offset_ms:
        print(f"GAGAL: offset melebihi {args.max_offset_ms} ms. Sinkronkan jam (chrony) lalu ulangi.")
        sys.exit(1)
    print("LULUS")


if __name__ == "__main__":
    main()
