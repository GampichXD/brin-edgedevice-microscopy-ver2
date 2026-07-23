import os
import time
import requests
from glob import glob

# Konfigurasi: 
# Ambil lokasi absolut dari file script ini (Backend_Connector)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Mundur 2 folder ke root 'Tugas Akhir', lalu masuk ke Software/backend/static/datasets
DATASET_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "Software", "backend", "static", "datasets"))

# URL Mock VPS (Dalam skenario asli, ini diganti dengan URL Server VPS Publik)
VPS_SYNC_URL = os.getenv("VPS_SYNC_URL", "http://localhost:8000/api/dataset/vps-mock/sync")

# Interval sinkronisasi dalam detik
SYNC_INTERVAL = 10

def sync_datasets():
    print(f"[SYNCER] Memantau direktori lokal Jetson: {DATASET_DIR}")
    
    if not os.path.exists(DATASET_DIR):
        print("[SYNCER] Direktori dataset belum terbentuk. Menunggu...")
        return
        
    for folder_name in os.listdir(DATASET_DIR):
        folder_path = os.path.join(DATASET_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue
            
        # Temukan semua file gambar di dalam folder ini
        images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img in images:
            img_path = os.path.join(folder_path, img)
            synced_marker = img_path + ".synced"
            
            # Jika belum ada file .synced, berarti ini file baru yang perlu diupload ke VPS
            if not os.path.exists(synced_marker):
                print(f"[SYNCER] Ditemukan file baru: {img} di folder {folder_name}")
                print(f"[SYNCER] Mengunggah {img} ke Cloud VPS ({VPS_SYNC_URL})...")
                
                try:
                    with open(img_path, 'rb') as f:
                        files = {'files': (img, f, 'image/jpeg')}
                        response = requests.post(VPS_SYNC_URL, files=files, timeout=10)
                        
                    if response.status_code == 200:
                        # Jika berhasil, buat file marker kosong agar tidak diupload berulang kali
                        with open(synced_marker, 'w') as marker:
                            marker.write("SYNCED_OK")
                        print(f"[SYNCER] SUKSES! File {img} telah di-backup ke VPS.")
                    else:
                        print(f"[SYNCER] GAGAL mengunggah {img}. Server VPS merespon: {response.status_code}")
                except requests.exceptions.RequestException as e:
                    print(f"[SYNCER] ERROR Koneksi ke VPS: {e}. Akan dicoba lagi nanti.")

def run_daemon():
    print("="*50)
    print("HYBRID SYNC DAEMON AKTIF")
    print("="*50)
    while True:
        try:
            sync_datasets()
        except Exception as e:
            print(f"[SYNCER FATAL ERROR] {e}")
        time.sleep(SYNC_INTERVAL)

if __name__ == "__main__":
    try:
        run_daemon()
    except KeyboardInterrupt:
        print("[SYNCER] Daemon dihentikan oleh pengguna.")
