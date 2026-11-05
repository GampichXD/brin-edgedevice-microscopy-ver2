"""
Bendera status ringan yang dibagi antar-modul di perangkat tepi (Jetson).

Dipakai untuk 'menenangkan' subsistem lain selama tile stitching gambar
banyak — proses berat CPU/RAM yang sebelumnya membuat main.py 'bridge error'.
Saat `stitching` True:
  - vps_bridge.send_stream_task  -> lewati capture/encode/kirim frame video
  - vps_bridge.telemetry_sender  -> kirim telemetri lebih jarang
  - vps_syncer.sync_datasets     -> lewati satu siklus sinkronisasi
Kamera fisik ditutup sementara oleh ai_handler untuk memberi ruang RAM, lalu
dibuka lagi setelah stitching selesai (sukses atau gagal).
"""

stitching = False       # True selama subprocess tile stitching berjalan
stream_was_on = False    # nilai is_streaming sebelum stitching, untuk dipulihkan
