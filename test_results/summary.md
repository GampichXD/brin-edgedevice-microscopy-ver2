# Ringkasan Hasil Pengujian Hardware

Sumber data: `/home/brin-microscope/Documents/Tugas-Akhir/Hardware/test_logs/_analisis_20261111`

## D2. Latensi Perintah Motor

| Jalur | n | Rata-rata (ms) | SD | p50 | p95 | Maks |
|---|---|---|---|---|---|---|
| Lokal (Jetson→GRBL) | 100 | 6.9 | 0.3 | 6.8 | 7.4 | 7.7 |
| Cloud end-to-end | 99 | 110.8 | 90.7 | 76.6 | 304.0 | 588.6 |
| Cloud satu-arah (kirim→Jetson) | 99 | 90.7 | 85.2 | 62.0 | 234.2 | 571.2 |

Perintah tidak sampai ke Jetson: **1**.

## D3. Latensi Video & FPS

| Metrik | n | Rata-rata | SD | p50 | p95 | Maks |
|---|---|---|---|---|---|---|
| Latensi (ms) | 17159 | 658.9 | 472.1 | 526.9 | 1732.2 | 2519.3 |
| FPS | 599 | 28.6 | 5.8 | 30.0 | 35.0 | 64.0 |
| Jarak antar-frame (ms) | 17158 | 34.9 | 39.5 | 21.3 | 81.1 | 1810.6 |
| Bitrate (kbps) | 599 | 18232.1 | 3684.7 | 18822.8 | 22186.6 | 40591.2 |

## D4. Validasi Estimasi Waktu Grid Scan (persamaan 3–4)

| Grid | Tile | Prediksi (s) | Aktual rata-rata (s) | SD | MAPE (%) | Overhead/tile (s) |
|---|---|---|---|---|---|---|
| 3x3 | 9 | 14.9 | 16.0 | 0.4 | 6.8 | 0.12 |
| 5x5 | 25 | 39.8 | 40.0 | 1.3 | 2.7 | 0.01 |
| 10x10 | 100 | 152.0 | 178.9 | 15.0 | 14.6 | 0.27 |

MAPE persamaan (3)–(4) saja: **8.05%**. Overhead capture+unggah rata-rata **0.13 s/tile**; model diperluas `T = T_pred + n·t_capture` memberi MAPE **6.02%** (t_capture diestimasi dari data yang sama — untuk validasi yang adil, estimasi dari grid kecil lalu uji di grid besar).

## D5. Uji Rate-Limit (persamaan 6)

| Kanal | Dikirim/detik | Diterima/detik (rata-rata) | Diterima/detik (maks) | Total ditolak |
|---|---|---|---|---|
| rest | 20.0 | 1.90 | 2 | 181 |
| ws | 18.2 | 3.27 | 4 | 164 |

## D6. Termal & Sumber Daya Jetson

| Suhu CPU maks (°C) | Suhu GPU maks (°C) | RAM maks (%) | RAM rata-rata (%) | CPU rata-rata (%) | Durasi RAM > 75% (s) |
|---|---|---|---|---|---|
| 54.3 | 54.3 | 58.3 | 56.0 | 23.2 | 0 |

## D7. Uji Watchdog Keep-Alive (fail-safe)

13 percobaan (12 saat motor bergerak). 5 pemicu ulang saat Wi-Fi tersambung-putus kembali tidak dihitung. Waktu berhenti diukur dari `FEED_HOLD_SENT` karena telemetri tidak tercatat selama koneksi putus.

| Metrik | n | Rata-rata (ms) | SD | p95 | Maks |
|---|---|---|---|---|---|
| Pesan terakhir → watchdog terpicu | 13 | 3130 | 83 | 3236 | 3238 |
| Pesan terakhir → feed hold terkirim | 13 | 3131 | 83 | 3236 | 3239 |

## D8. Jeda Settle vs Ketajaman Citra

| Jenis gerak | Jeda minimum agar ketajaman ≥ 95% maks (s) |
|---|---|
| Dalam baris | 0.00 |
| Pindah baris | 0.00 |
