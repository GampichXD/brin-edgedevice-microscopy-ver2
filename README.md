# Edge Device Mikroskop Digital (Jetson Orin Nano)

Program kontrol di Jetson: motor CNC lewat GRBL, kamera IMX477 lewat GStreamer, stitching dan analisis citra, serta jembatan WebSocket ke VPS. Backend dan antarmuka web ada di repo `brin-digital-microscopy-ver2`.

## Syarat

Folder repo **harus bernama `Hardware`**, karena semua modul di-import sebagai `Hardware.xxx`.

## Menjalankan

```bash
cp .env.example .env    # lalu isi nilainya
python3 main.py
```

`EDGE_DEVICE_TOKEN` harus sama dengan nilai di VPS. Kalau ditolak (kode 4003), Jetson mencoba lagi setiap 30 detik.

## Keselamatan motor

- **Watchdog koneksi.** VPS mengirim PING tiap 1 detik. Kalau tidak ada pesan selama `WATCHDOG_TIMEOUT` detik (default 3), atau koneksi putus, Jetson mengirim feed hold (`!`) ke GRBL.
- **Tidak ada resume otomatis.** Setelah hold, motor diam sampai ada perintah baru. Perintah itu melepas hold lewat soft reset, sehingga sisa gerakan lama dibuang, lalu offset G92 dipulihkan.
- **Grid scan yang sedang berjalan dibatalkan** oleh VPS saat koneksi Jetson putus.
- **Soft limit** aktif secara default, tetapi batasnya harus diisi di `.env` dari hasil ukur rentang mekanis mesin. Sumbu tanpa nilai tidak dibatasi.

## Satuan sumbu Z

Sumbu Z dikonfigurasi GRBL `$102 = 1`, sehingga 1 unit G-code = 1 step. Nilainya dibaca saat start dan dicek terhadap `EXPECTED_Z_STEPS_PER_UNIT`; kalau berbeda, muncul peringatan di log. Label `stp` di web hanya benar selama setelan firmware ini tidak berubah.

## Bobot model

Bobot `*.engine`, `*.onnx`, dan sejenisnya tidak disimpan di Git (lihat `.gitignore`), tetapi `best_yolo26obj.pt`, `superpoint_lightglue_v0-1.pth`, dan `best_Seg_1280.pt` masih dilacak.

## Pengujian

`Testing/unit/` berisi tes unit yang tidak butuh perangkat keras:

```bash
python3 -m pytest Testing/unit --rootdir=Testing/unit
```

Prosedur uji perangkat keras untuk Bab IV ada di [Testing/README.md](Testing/README.md).

## Skrip diagnostik

`tools/` berisi skrip bantu yang tidak dipakai saat program berjalan: `coba.py`, `ws_test.py`, `diagnose_stream.py`, `test_camera.py`.
