# Uji watchdog koneksi (20 percobaan)

Prosedur dilakukan manual, datanya terekam otomatis ke `test_logs/watchdog_log.csv` dan `test_logs/telemetry.csv`.

1. Jalankan di Jetson: `TEST_LOGGING=1 TEST_TELEMETRY_INTERVAL=0.05 python3 main.py`
2. Matikan live stream (frame video memenuhi buffer socket yang putus dan bisa menahan telemetri). Dari web, jog sumbu X dengan perintah panjang (misalnya 20 mm, F100) supaya motor masih bergerak saat koneksi dicabut.
3. Cabut kabel LAN atau matikan Wi-Fi Jetson. Catat nomor percobaan dan jamnya di lembar uji.
4. Tunggu motor berhenti, lalu sambungkan kembali dan tunggu status tersambung.
5. Kirim satu perintah gerak kecil dari web. Motor harus bergerak sejauh perintah itu saja, tanpa melanjutkan sisa jog sebelumnya. Bandingkan posisi di web dengan posisi fisik.
6. Ulangi dari langkah 2.

## Yang dinilai

`analyze_hw.py` menghitung (a) putus sampai watchdog terpicu dan (b) putus sampai status GRBL `Hold`.

**Kriteria lulus:** semua percobaan berhenti, p95 (b) ≤ `WATCHDOG_TIMEOUT` + 500 ms, dan langkah 5 tidak menjalankan sisa gerakan lama.

## Catatan

- Watchdog baru aktif setelah PING pertama dari VPS, jadi VPS harus sudah memakai kode baru.
- Status `Hold` hanya terekam lewat telemetri lokal Jetson. Baris `telemetry.csv` tetap bertambah selama koneksi putus, kecuali kalau task telemetri sudah berhenti karena sambungan ditutup. Kalau itu terjadi, pakai `FEED_HOLD_SENT` di `watchdog_log.csv` sebagai acuan waktu berhenti, dan tulis batasan ini di laporan.
- Grid scan yang sedang berjalan dibatalkan oleh VPS saat Jetson putus. Tile yang sudah tersimpan tetap bisa dilihat di modal review.
