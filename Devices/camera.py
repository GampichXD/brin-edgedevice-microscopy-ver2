import cv2
import platform
import os
import threading
import time
import subprocess
import struct
import atexit
from datetime import datetime

class IMX477CameraCore:
    def __init__(self, sensor_id=0, width=1280, height=720, fps=30):
        self.sensor_id = sensor_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.shutter_speed = None
        self.iso = None
        self.proc = None

        # Threading untuk live streaming anti-blocking
        self.jpeg_frame = None   # Menyimpan JPEG bytes mentah (sudah terkompresi)
        self.started = False
        self.read_lock = threading.Lock()
        self.thread = None

        # Mendaftarkan pembersihan otomatis saat script mati
        atexit.register(self.close)

    def open(self):
        """Membuka lensa hardware kamera secara eksklusif dan menyalakan background thread."""
        if self.started:
            return True

        current_os = platform.system()
        try:
            if current_os == "Windows":
                print("[CORE CAMERA] Berjalan di Windows. Mengunci Webcam Laptop via DirectShow...")
                self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    print("[CORE CAMERA WARNING] DirectShow gagal, mencoba default backend...")
                    self.cap = cv2.VideoCapture(0)
                
                if not self.cap.isOpened():
                    print("[CORE CAMERA ERROR] Gagal membuka kamera di Windows.")
                    return False

                self.started = True
                self.thread = threading.Thread(target=self._update_loop_opencv, daemon=True)
                self.thread.start()
                print("[CORE CAMERA SUCCESS] Webcam berhasil dibuka (Windows mode).")
                return True
            
            else:
                # Jetson Linux: Gunakan GStreamer + jpegenc → pipe.
                # Coba maksimal 2x; di antara percobaan, reset nvargus-daemon
                # untuk melepas sensor yang mungkin 'bocor' dari sesi sebelumnya.
                for attempt in (1, 2):
                    first_frame = self._spawn_gst_pipeline()
                    if first_frame is not None:
                        with self.read_lock:
                            self.jpeg_frame = first_frame
                        self.started = True
                        self.thread = threading.Thread(target=self._update_loop_gst, daemon=True)
                        self.thread.start()
                        print("[CORE CAMERA SUCCESS] IMX477 terhubung, frame JPEG mengalir ke buffer.")
                        return True

                    print(f"[CORE CAMERA ERROR] Percobaan {attempt}/2 gagal mendapat frame.")
                    self._kill_gst_proc(hard=True)
                    if attempt == 1:
                        self._reset_nvargus_daemon()

                print("[CORE CAMERA ERROR] Kamera tetap tidak bisa dibuka. "
                      "Jalankan manual: sudo systemctl restart nvargus-daemon")
                return False

        except Exception as e:
            print(f"[CORE CAMERA CRITICAL ERROR] Gagal membuka kamera: {e}")
            import traceback; traceback.print_exc()
            return False

    def _gst_command(self):
        # Injeksi exposure & gain dari pengaturan kamera (shutter µs, ISO).
        src = ["nvarguscamerasrc", f"sensor-id={self.sensor_id}"]
        if self.shutter_speed:
            ns = int(float(self.shutter_speed)) * 1000  # µs -> ns
            src.append(f"exposuretimerange={ns} {ns}")
            src.append("aelock=true")
        if self.iso:
            g = max(1.0, min(22.0, float(self.iso) / 100.0))   # ISO ~-> analog gain
            src.append(f"gainrange={g:.2f} {g:.2f}")
            src.append("aelock=true")
        return [
            "gst-launch-1.0", "-q", *src, "!",
            "video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1,format=NV12", "!",
            "queue", "max-size-buffers=2", "leaky=2", "!",
            "nvvidconv", "!",
            f"video/x-raw,width={self.width},height={self.height},format=(string)I420", "!",
            "queue", "max-size-buffers=2", "leaky=2", "!",
            "jpegenc", "quality=80", "!",
            "queue", "max-size-buffers=2", "leaky=2", "!",
            "multipartmux", "boundary=FRAME", "!",
            "fdsink", "fd=1", "sync=false",
        ]

    def _spawn_gst_pipeline(self):
        """Jalankan gst-launch dan kembalikan frame JPEG pertama (atau None)."""
        # Hanya SIGINT (tidak pernah SIGKILL) ke gst-launch lama supaya
        # nvargus-daemon sempat melepas sensor dengan bersih.
        subprocess.run(["killall", "-2", "gst-launch-1.0"],
                       stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        time.sleep(3.0)  # beri argus daemon waktu melepas & re-enumerate sensor

        print(f"[CORE CAMERA] Membuka IMX477 via GStreamer JPEG-pipe (Sensor ID: {self.sensor_id})...")
        self.proc = subprocess.Popen(self._gst_command(), stdout=subprocess.PIPE,
                                     stderr=None, bufsize=0)
        return self._read_one_multipart_frame(timeout=8.0)

    def _kill_gst_proc(self, hard=False):
        """Hentikan proses gst-launch milik instance ini. SIGINT dulu (EOS bersih),
        SIGKILL hanya bila hard=True dan proses membandel."""
        if self.proc is None:
            return
        import signal as _signal
        try:
            if self.proc.stdout:
                try:
                    self.proc.stdout.close()  # kirim EOF ke fdsink -> pipeline EOS
                except Exception:
                    pass
            self.proc.send_signal(_signal.SIGINT)
            try:
                self.proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    if hard:
                        print("[CORE CAMERA WARNING] gst-launch tidak merespons, SIGKILL. "
                              "nvargus-daemon mungkin perlu di-restart.")
                        self.proc.kill()
        finally:
            self.proc = None

    def _reset_nvargus_daemon(self):
        """Best-effort restart nvargus-daemon untuk melepas sensor yang 'bocor'.
        Butuh sudo tanpa password (tambahkan rule sudoers) agar berjalan otomatis."""
        print("[CORE CAMERA] Mereset nvargus-daemon untuk melepas sensor...")
        # Pastikan tidak ada klien gst yang menahan socket saat daemon direstart.
        subprocess.run(["killall", "-9", "gst-launch-1.0"],
                       stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        for cmd in (["sudo", "-n", "systemctl", "restart", "nvargus-daemon"],
                    ["systemctl", "restart", "nvargus-daemon"]):
            try:
                r = subprocess.run(cmd, stderr=subprocess.DEVNULL,
                                   stdout=subprocess.DEVNULL, timeout=15)
                if r.returncode == 0:
                    time.sleep(5.0)  # daemon butuh beberapa detik untuk enumerasi sensor CSI
                    return True
            except Exception:
                pass
        print("[CORE CAMERA WARNING] Gagal restart nvargus-daemon otomatis "
              "(perlu 'sudo NOPASSWD' untuk systemctl). Jika kamera tetap gagal, "
              "channel VI kernel kemungkinan bocor — perlu reboot Jetson.")
        return False

    def _read_one_multipart_frame(self, timeout=5.0):
        """Membaca satu frame JPEG dari multipart/x-mixed-replace stream.
        Format: --FRAME\r\nContent-Type: image/jpeg\r\nContent-Length: N\r\n\r\n<data>\r\n
        """
        if self.proc is None or self.proc.stdout is None:
            return None

        start = time.time()
        buf = b""
        
        # Import fcntl for non-blocking read
        import fcntl
        import select

        fd = self.proc.stdout.fileno()
        # Set non-blocking
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

        try:
            while time.time() - start < timeout:
                # Wait for data with timeout
                ready, _, _ = select.select([self.proc.stdout], [], [], 0.5)
                if not ready:
                    if self.proc.poll() is not None:
                        print("[CORE CAMERA ERROR] GStreamer process exited unexpectedly.")
                        return None
                    continue

                chunk = self.proc.stdout.read(4096)
                if not chunk:
                    continue
                buf += chunk

                # Look for JPEG magic bytes: FF D8 ... FF D9
                start_idx = buf.find(b'\xff\xd8')
                if start_idx == -1:
                    # Keep last few bytes in case marker spans chunks
                    buf = buf[-4:]
                    continue

                end_idx = buf.find(b'\xff\xd9', start_idx + 2)
                if end_idx == -1:
                    # Frame not complete yet, keep reading
                    continue

                # Extract JPEG bytes
                jpeg_bytes = buf[start_idx:end_idx + 2]
                return jpeg_bytes

        except Exception as e:
            print(f"[CORE CAMERA] Frame read error: {e}")
        
        return None

    def _update_loop_gst(self):
        """Loop internal thread: terus membaca frame JPEG dari GStreamer pipe."""
        sleep_time = 1.0 / self.fps
        
        while self.started:
            try:
                frame = self._read_one_multipart_frame(timeout=2.0)
                if frame is not None:
                    with self.read_lock:
                        self.jpeg_frame = frame
                else:
                    # GStreamer mungkin putus
                    if self.proc and self.proc.poll() is not None:
                        print("[CORE CAMERA WARNING] GStreamer process mati, menghentikan loop.")
                        break
                    time.sleep(sleep_time)
            except Exception as e:
                print(f"[CORE CAMERA STREAM ERROR] {e}")
                time.sleep(0.1)

    def _update_loop_opencv(self):
        """Loop internal thread untuk OpenCV VideoCapture (Windows)."""
        while self.started and self.cap is not None:
            success, frame = self.cap.read()
            if success:
                cv2.putText(frame, "WEBCAM LAPTOP (MOCK IMX477)", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    with self.read_lock:
                        self.jpeg_frame = buf.tobytes()
            else:
                time.sleep(0.02)
            time.sleep(1.0 / self.fps)

    def capture_frame(self):
        """Mengambil satu frame teranyar sebagai numpy array BGR.
        Decode dari JPEG bytes yang tersimpan di buffer.
        """
        if not self.started:
            return None

        with self.read_lock:
            jpeg_bytes = self.jpeg_frame

        if jpeg_bytes is None:
            return None

        try:
            import numpy as np
            arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return frame
        except Exception as e:
            print(f"[CORE CAMERA] Decode error: {e}")
            return None

    def capture_jpeg_bytes(self):
        """Mengambil JPEG bytes langsung (tanpa perlu decode ke numpy).
        Lebih efisien untuk streaming via WebSocket.
        """
        if not self.started:
            return None
        with self.read_lock:
            return self.jpeg_frame

    def apply_settings(self, shutter_speed=None, iso=None):
        if shutter_speed is not None:
            self.shutter_speed = int(float(shutter_speed))
        if iso is not None:
            self.iso = int(float(iso))

        # Windows/OpenCV: coba set properti langsung.
        if self.cap is not None:
            try:
                if self.shutter_speed:
                    self.cap.set(cv2.CAP_PROP_EXPOSURE, self.shutter_speed)
                if self.iso:
                    self.cap.set(cv2.CAP_PROP_GAIN, self.iso)
            except Exception:
                pass
            return True

        # Jetson/GStreamer: props exposure/gain hanya bisa saat pipeline dibuat,
        # jadi RESTART pipeline sekali dengan _gst_command() yang sudah memuat
        # nilai baru. Kalau belum streaming, cukup simpan (berlaku saat dinyalakan).
        if self.started and self.proc is not None:
            print(f"[CORE CAMERA] Menerapkan shutter={self.shutter_speed}µs iso={self.iso} (restart pipeline)...")
            try:
                self.close()
                time.sleep(0.6)
                return self.open()
            except Exception as e:
                print(f"[CORE CAMERA ERROR] Gagal restart untuk apply settings: {e}")
                return False
        return True

    def capture_to_bytes(self, quality=75):
        """Mengembalikan JPEG bytes untuk kompatibilitas backward."""
        # Di Jetson mode, jpeg_frame sudah terkompresi, langsung kembalikan
        return self.capture_jpeg_bytes()

    def save_snapshot(self, output_dir, prefix="IMG", coord_x=0.0, coord_y=0.0, requested_filename=None):
        """Menyimpan gambar fisik resolusi tinggi untuk kebutuhan pengumpulan dataset."""
        jpeg_bytes = self.capture_jpeg_bytes()
        if jpeg_bytes is None:
            return None

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if requested_filename:
            filename = requested_filename
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}_{timestamp}_X{str(coord_x).replace('.','_')}_Y{str(coord_y).replace('.','_')}.jpg"

        file_path = os.path.join(output_dir, filename)

        try:
            with open(file_path, 'wb') as f:
                f.write(jpeg_bytes)
            print(f"[CORE CAMERA SUCCESS] Gambar dataset tersimpan: {file_path}")
            return filename
        except Exception as e:
            print(f"[CORE CAMERA ERROR] Gagal menulis berkas gambar ke disk: {e}")
            return None

    def close(self):
        """Melepas sensor kamera dan mematikan background thread dengan aman."""
        if not self.started and self.proc is None and self.cap is None:
            return

        # Hentikan loop pembaca DULU agar tidak ada race saat pipe ditutup.
        self.started = False
        if self.thread:
            self.thread.join(timeout=3.0)
            self.thread = None

        if self.proc:
            print("[CORE CAMERA] Menutup pipeline GStreamer dengan bersih...")
            self._kill_gst_proc(hard=True)

        if self.cap:
            self.cap.release()
            self.cap = None

        print("[CORE CAMERA] Kunci sensor hardware kamera resmi dibebaskan.")


camera_core = IMX477CameraCore()