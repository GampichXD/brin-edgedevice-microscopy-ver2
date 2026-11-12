# Pengujian perangkat keras

Semua skrip menambah baris ke `test_logs/*.csv` (tidak menimpa) dan punya `--help`. `analyze_hw.py` mengubah CSV menjadi grafik dan tabel untuk Bab IV. **Jangan ubah nama kolom CSV tanpa mengubah skrip analisis.**

Skrip di Jetson menolak jalan tanpa GRBL sungguhan, supaya angka dari motor simulasi tidak pernah masuk laporan. `t01 selftest` tetap bisa jalan tanpa motor.

## Alat

- Stage micrometer (slide kalibrasi), untuk t01.
- Laptop di jaringan yang sama (LAN) dan di internet (cloud), untuk t03 sampai t06.
- `pip install -r Testing/requirements-test.txt`

## Urutan

| Langkah | Skrip | Di mana | `main.py` |
|---|---|---|---|
| 1 | `t00_check_clock.py` | Jetson dan laptop | bebas |
| 2 | `t01_positioning.py` | Jetson | mati |
| 3 | `t02_cmd_latency_local.py` | Jetson | mati |
| 4 | `t05b_settle_sharpness.py` | Jetson | mati |
| 5 | `t03_cmd_latency_client.py --path cloud`, lalu `lan` | laptop | hidup, `TEST_LOGGING=1` |
| 6 | `t04_video_client.py` | laptop | hidup |
| 7 | `t05_eta_scan.py` (termal terekam bersamaan) | laptop | hidup, `TEST_LOGGING=1` |
| 8 | `t06_ratelimit_burst.py` | laptop | hidup |
| 9 | `t07_watchdog_protocol.md` (manual) | Jetson | hidup, `TEST_LOGGING=1` |

Skrip di laptop memakai env `BASE_URL`, `TEST_USER`, dan `TEST_PASS`. Setelah tiap sesi, salin `test_logs/` dari Jetson ke laptop (`scp`), supaya semua CSV ada di satu folder.

## Uji posisi (t01)

1. Letakkan stage micrometer di bawah lensa, fokuskan, pastikan garis skala dan satu angka terlihat. `--minor-um` adalah jarak garis kecil pada label slide.
2. `python3 Testing/t01_positioning.py calibrate --axis X --minor-um 10`. Buka `test_logs/calib_X_preview.png`, yang diberi grid koordinat tiap 50 piksel.
3. Pilih kotak yang hanya berisi satu angka skala, lalu ulangi dengan `--roi x,y,w,h`.
4. `python3 Testing/t01_positioning.py run --axis X --runs 5`, lalu ulangi untuk Y.

Uji algoritma tanpa motor: `python3 Testing/t01_positioning.py selftest --image foto.jpg --roi x,y,w,h`.

## Mencoba Jetson dengan backend lokal

Set `SYNC_DISABLED=1`. `vps_syncer` menghapus dataset lokal berumur lebih dari 7 hari yang tidak ada di server.

## Analisis

```bash
python3 Testing/analyze_hw.py --data test_logs --out test_results
```

## Keterbatasan yang perlu ditulis di laporan

- Rentang uji t01 terbatas pada lebar bidang pandang (ratusan µm sampai 1 mm). Hasilnya mewakili akurasi **langkah pendek** yang relevan untuk tile scanning, bukan akurasi sepanjang langkah penuh mesin (yang di ISO 230-2 diukur dengan laser interferometer).
- t00 memakai NTP, karena VPS tidak punya endpoint waktu presisi. Selisih jam Jetson dan laptop dihitung dari offset masing-masing terhadap server NTP yang sama.

## Tes unit

`Testing/unit/` berjalan tanpa perangkat keras:

```bash
python3 -m pytest Testing/unit --rootdir=Testing/unit
```
