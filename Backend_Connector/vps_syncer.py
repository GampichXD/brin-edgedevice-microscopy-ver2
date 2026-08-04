import os
import time
import requests
from glob import glob

# Konfigurasi: 
# Ambil lokasi absolut dari file script ini (Backend_Connector)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Direktori penyimpanan mandiri khusus Edge Device (Hardware)
DATASET_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "local_datasets"))

# URL VPS Base
VPS_BASE_URL = os.getenv("VPS_BASE_URL", "http://localhost:8000/api/dataset")

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
            
        # Temukan semua file media (gambar dan video) di dalam folder ini
        media_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.mp4', '.webm'))]
        
        for file_name in media_files:
            # Skip file sementara (sedang direkam) dan file 0-byte
            if file_name.startswith('_temp_'):
                continue
            file_path = os.path.join(folder_path, file_name)
            if os.path.getsize(file_path) == 0:
                print(f"[SYNCER] Melewati file 0-byte: {file_name}")
                continue
            synced_marker = file_path + ".synced"
            
            # Jika belum ada file .synced, berarti ini file baru yang perlu diupload ke VPS
            if not os.path.exists(synced_marker):
                print(f"[SYNCER UPSTREAM] Ditemukan file baru: {file_name} di folder {folder_name}")
                sync_url = f"{VPS_BASE_URL}/folders/{folder_name}/files"
                print(f"[SYNCER UPSTREAM] Mengunggah {file_name} ke Cloud VPS ({sync_url})...")
                
                try:
                    with open(file_path, 'rb') as f:
                        if file_name.lower().endswith('.mp4'):
                            mime_type = 'video/mp4'
                        elif file_name.lower().endswith('.webm'):
                            mime_type = 'video/webm'
                        else:
                            mime_type = 'image/jpeg'
                        files = {'files': (file_name, f, mime_type)}
                        response = requests.post(sync_url, files=files, timeout=30)
                        
                    if response.status_code == 200:
                        # Jika berhasil, buat file marker kosong agar tidak diupload berulang kali
                        with open(synced_marker, 'w') as marker:
                            marker.write("SYNCED_OK")
                        print(f"[SYNCER UPSTREAM] SUKSES! File {file_name} telah di-backup ke VPS.")
                    else:
                        print(f"[SYNCER UPSTREAM] GAGAL mengunggah {file_name}. Server VPS merespon: {response.status_code}")
                except requests.exceptions.RequestException as e:
                    print(f"[SYNCER UPSTREAM] ERROR Koneksi ke VPS: {e}. Akan dicoba lagi nanti.")

    # =========================================================================
    # STEP 2: DOWNSTREAM MIRRORING (TURUN: Server Hosting -> Hardware Jetson)
    # =========================================================================
    server_folders = None
    try:
        index_url = f"{VPS_BASE_URL}/vps/sync-index"
        res = requests.get(index_url, timeout=10)
        if res.status_code == 200:
            server_folders = res.json().get("folders", {})
            for folder_id, data in server_folders.items():
                local_folder_path = os.path.join(DATASET_DIR, folder_id)
                os.makedirs(local_folder_path, exist_ok=True)
                
                for file_name in data.get("files", []):
                    local_file_path = os.path.join(local_folder_path, file_name)
                    # Jika file dari server belum ada di hard disk fisik Jetson
                    if not os.path.exists(local_file_path):
                        print(f"[SYNCER DOWNSTREAM] Mengunduh cermin baru dari Server: {file_name} ke folder {folder_id}...")
                        download_url = f"{VPS_BASE_URL}/folders/{folder_id}/files/{file_name}/download"
                        with requests.get(download_url, stream=True, timeout=30) as r:
                            if r.status_code == 200:
                                with open(local_file_path, 'wb') as f:
                                    for chunk in r.iter_content(chunk_size=8192):
                                        if chunk:
                                            f.write(chunk)
                                # Beri tanda synced juga untuk file hasil download
                                with open(local_file_path + ".synced", 'w') as marker:
                                    marker.write("SYNCED_OK")
                                print(f"[SYNCER DOWNSTREAM] SUKSES! {file_name} selesai dicermin ke Hardware.")
    except Exception as e:
        print(f"[SYNCER DOWNSTREAM ERROR] Gagal mengambil indeks sinkronisasi dari Server VPS: {e}")

    # =========================================================================
    # STEP 3: AUTO-CLEANUP (Retensi 7 Hari untuk File Yatim)
    # =========================================================================
    if server_folders is not None:
        current_time = time.time()
        seven_days_sec = 7 * 24 * 3600
        
        for folder_name in os.listdir(DATASET_DIR):
            local_folder_path = os.path.join(DATASET_DIR, folder_name)
            if not os.path.isdir(local_folder_path):
                continue
                
            if folder_name not in server_folders:
                # Folder ini sudah dihapus di server
                folder_mtime = os.path.getmtime(local_folder_path)
                if (current_time - folder_mtime) > seven_days_sec:
                    import shutil
                    print(f"[SYNCER CLEANUP] Menghapus folder yatim (lebih dari 7 hari): {folder_name}")
                    shutil.rmtree(local_folder_path, ignore_errors=True)
            else:
                # Folder ada di server, periksa file individu
                server_files = set(server_folders[folder_name].get("files", []))
                for file_name in os.listdir(local_folder_path):
                    if file_name.endswith(".synced"):
                        continue
                    if file_name not in server_files:
                        file_path = os.path.join(local_folder_path, file_name)
                        file_mtime = os.path.getmtime(file_path)
                        synced_marker = file_path + ".synced"
                        # Hanya hapus jika sudah disinkronisasi (ada di server sebelumnya) dan sudah yatim selama 7 hari
                        if os.path.exists(synced_marker) and (current_time - file_mtime) > seven_days_sec:
                            print(f"[SYNCER CLEANUP] Menghapus file yatim (lebih dari 7 hari): {file_name}")
                            os.remove(file_path)
                            os.remove(synced_marker)

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
