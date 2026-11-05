"""Logger CSV untuk data uji Bab IV. Nama file dan kolom dibaca Testing/analyze_hw.py."""
import csv
import os
import threading

LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "test_logs"))
_lock = threading.Lock()


def enabled() -> bool:
    return os.getenv("TEST_LOGGING", "0") == "1"


def log(name: str, always: bool = False, **fields) -> None:
    """Tambah satu baris ke test_logs/<name>.csv. Aktif kalau TEST_LOGGING=1,
    atau always=True untuk kejadian yang wajib tercatat (watchdog).
    Logger tidak boleh mengganggu kontrol: semua kegagalan ditelan."""
    if not always and not enabled():
        return
    try:
        with _lock:
            os.makedirs(LOG_DIR, exist_ok=True)
            path = os.path.join(LOG_DIR, f"{name}.csv")
            new_file = not os.path.exists(path) or os.path.getsize(path) == 0
            with open(path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(fields.keys()))
                if new_file:
                    writer.writeheader()
                writer.writerow(fields)
                f.flush()
    except Exception as e:
        print(f"[TEST LOGGER] Gagal menulis {name}.csv: {e}")
