# main.py otomatis di Jetson (systemd)

## Pasang (sekali)
```bash
sudo cp ~/Documents/Tugas-Akhir/Hardware/deploy/smart-microscope-edge.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now smart-microscope-edge
```

## Sehari-hari
| Perintah | Fungsi |
|---|---|
| `systemctl status smart-microscope-edge` | Cek jalan atau tidak |
| `journalctl -u smart-microscope-edge -f` | Lihat log langsung (Ctrl+C untuk keluar) |
| `sudo systemctl restart smart-microscope-edge` | Jalankan ulang, mis. setelah `git pull` |
| `sudo systemctl stop smart-microscope-edge` | Hentikan sementara (sampai reboot) |
| `sudo systemctl disable --now smart-microscope-edge` | Matikan dan tidak jalan otomatis lagi |

Jangan menjalankan `python main.py` di terminal selama layanan aktif: dua proses akan
berebut kamera dan GRBL. Hentikan layanan dulu kalau ingin menjalankan manual (mis. uji t01/t02/t05b).
