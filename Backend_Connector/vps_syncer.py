import os
import time
import requests
from glob import glob

# Konfigurasi: 
# Ambil lokasi absolut dari file script ini (Backend_Connector)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Direktori penyimpanan mandiri khusus Edge Device (Hardware)
DATASET_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "local_datasets"))

# URL Mock VPS (Dalam skenario asli, ini diganti dengan URL Server VPS Publik)
VPS_BASE_URL = os.getenv("VPS_BASE_URL", "http://localhost:8000/api/dataset")
VPS_SYNC_URL = os.getenv("VPS_SYNC_URL", f"{VPS_BASE_URL}/vps-mock/sync")

# Interval sinkronisasi dalam detik
SYNC_INTERVAL = 10

def sync_datasets():
    print(f"[SYNCER] Memantau direktori lokal Jetson: {DATASET_DIR}")
    
    if not os.path.exists(DATASET_DIR):
        print("[SYNCER] Direktori dataset belum terbentuk. Menunggu...")
        return
        
    # =========================================================================
    # STEP 1: UPSTREAM SYNC (NAIK: Hardware Jetson -> Server Hosting VPS)
    # =========================================================================
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
                print(f"[SYNCER UPSTREAM] Ditemukan file baru: {img} di folder {folder_name}")
                print(f"[SYNCER UPSTREAM] Mengunggah {img} ke Cloud VPS ({VPS_SYNC_URL})...")
                
                try:
                    with open(img_path, 'rb') as f:
                        files = {'files': (img, f, 'image/jpeg')}
                        response = requests.post(VPS_SYNC_URL, files=files, timeout=10)
                        
                    if response.status_code == 200:
                        # Jika berhasil, buat file marker kosong agar tidak diupload berulang kali
                        with open(synced_marker, 'w') as marker:
                            marker.write("SYNCED_OK")
                        print(f"[SYNCER UPSTREAM] SUKSES! File {img} telah di-backup ke VPS.")
                    else:
                        print(f"[SYNCER UPSTREAM] GAGAL mengunggah {img}. Server VPS merespon: {response.status_code}")
                except requests.exceptions.RequestException as e:
                    print(f"[SYNCER UPSTREAM] ERROR Koneksi ke VPS: {e}. Akan dicoba lagi nanti.")

    # =========================================================================
    # STEP 2: DOWNSTREAM MIRRORING (TURUN: Server Hosting -> Hardware Jetson)
    # =========================================================================
    try:
        index_url = f"{VPS_BASE_URL}/vps/sync-index"
        res = requests.get(index_url, timeout=10)
        if res.status_code == 200:
            server_folders = res.json().get("folders", {})
            for folder_id, data in server_folders.items():
                local_folder_path = os.path.join(DATASET_DIR, folder_id)
                os.makedirs(local_folder_path, exist_ok=True)
                
                for img_name in data.get("images", []):
                    local_img_path = os.path.join(local_folder_path, img_name)
                    # Jika file gambar dari server belum ada di hard disk fisik Jetson
                    if not os.path.exists(local_img_path):
                        print(f"[SYNCER DOWNSTREAM] Mengunduh cermin baru dari Server: {img_name} ke folder {folder_id}...")
                        download_url = f"{VPS_BASE_URL}/folders/{folder_id}/images/{img_name}/download"
                        with requests.get(download_url, stream=True, timeout=15) as r:
                            if r.status_code == 200:
                                with open(local_img_path, 'wb') as f:
                                    for chunk in r.iter_content(chunk_size=8192):
                                        if chunk:
                                            f.write(chunk)
                                # Beri tanda synced juga untuk file hasil download
                                with open(local_img_path + ".synced", 'w') as marker:
                                    marker.write("SYNCED_OK")
                                print(f"[SYNCER DOWNSTREAM] SUKSES! {img_name} selesai dicermin ke Hardware.")
    except Exception as e:
        print(f"[SYNCER DOWNSTREAM ERROR] Gagal mengambil indeks sinkronisasi dari Server VPS: {e}")

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
