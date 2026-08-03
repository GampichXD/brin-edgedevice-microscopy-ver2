import cv2
import platform
import os
import threading
import time
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
        
        # Penambahan sirkuit threading untuk live streaming anti-blocking
        self.frame = None
        self.started = False
        self.read_lock = threading.Lock()
        self.thread = None

    def _get_gstreamer_pipeline(self):
        """Membuat sirkuit pipa GStreamer khusus untuk akselerasi perangkat keras Jetson Orin Nano."""
        return (
            f"nvarguscamerasrc sensor-id={self.sensor_id} ! "
            f"video/x-raw(memory:NVMM), width=(int){self.width}, height=(int){self.height}, "
            f"format=(string)NV12, framerate=(fraction){self.fps}/1 ! "
            f"nvvidconv flip-method=0 ! "
            f"video/x-raw, width=(int){self.width}, height=(int){self.height}, format=(string)BGRx ! "
            f"videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=true max-buffers=1"
        )

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
            else:
                pipeline = self._get_gstreamer_pipeline()
                print(f"[CORE CAMERA] Berjalan di Jetson Linux. Membuka MIPI CSI via GStreamer (ID: {self.sensor_id})...")
                self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                
                if not self.cap.isOpened():
                    print("[CORE CAMERA WARNING] GStreamer menolak. Mencoba fallback ke standar V4L2 (/dev/video0)...")
                    self.cap = cv2.VideoCapture(0)
        except Exception as e:
            print(f"[CORE CAMERA CRITICAL ERROR] Gagal menginisialisasi kamera hardware: {e}")
            return False

        if self.cap.isOpened():
            # Ambil satu frame pemicu awal sebelum melemparnya ke background thread
            success, self.frame = self.cap.read()
            self.started = True
            
            # Jalankan sirkuit background thread pembaca buffer kamera otomatis
            self.thread = threading.Thread(target=self._update_loop, args=())
            self.thread.daemon = True # Otomatis mati jika skrip utama dihentikan
            self.thread.start()
            
            print("[CORE CAMERA SUCCESS] Sensor kamera berhasil dikunci dan background thread aktif.")
            return True
            
        print("[CORE CAMERA ERROR] Gagal membuka kamera. Sensor sibuk atau tidak terdeteksi.")
        return False

    def apply_settings(self, shutter_speed=None, iso=None):
        self.shutter_speed = shutter_speed if shutter_speed is not None else self.shutter_speed
        self.iso = iso if iso is not None else self.iso

        if not self.cap or not self.cap.isOpened():
            return False

        try:
            if shutter_speed is not None:
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                self.cap.set(cv2.CAP_PROP_EXPOSURE, float(shutter_speed))
            if iso is not None:
                self.cap.set(cv2.CAP_PROP_GAIN, float(iso))
            print(f"[CORE CAMERA] Pengaturan diterapkan: shutter={self.shutter_speed}, iso={self.iso}")
            return True
        except Exception as e:
            print(f"[CORE CAMERA WARNING] Gagal menerapkan setting kamera: {e}")
            return False

    def _update_loop(self):
        """Loop internal thread untuk terus-menerus menguras buffer kamera."""
        import platform
        import time
        import cv2
        
        while self.started:
            if platform.system() == "Windows":
                # Baca frame riil dari webcam laptop secara background
                success, frame = self.cap.read()
                if success:
                    cv2.putText(frame, "WEBCAM LAPTOP (MOCK IMX477)", (30, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    with self.read_lock:
                        self.frame = frame
                else:
                    time.sleep(0.01)
                time.sleep(1.0 / self.fps)
            else:
                # Jalur asli GStreamer di Jetson Linux
                success, frame = self.cap.read()
                if success:
                    with self.read_lock:
                        self.frame = frame
                time.sleep(1.0 / self.fps)

    def capture_frame(self):
        """Mengambil satu frame teranyar secara instan dari memori thread atau hardware langsung."""
        if not self.started:
            print("[CORE CAMERA ERROR] Operasi capture gagal. Kamera belum dibuka.")
            return None
        
        with self.read_lock:
            frame_copy = self.frame.copy() if self.frame is not None else None
        return frame_copy

    def capture_to_bytes(self, quality=75):
        """Mengonversi frame mentah OpenCV menjadi biner JPEG (tobytes) untuk dikirim ke web dashboard."""
        frame = self.capture_frame()
        if frame is None:
            return None
        
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ret:
            return None
        return buffer.tobytes()

    def save_snapshot(self, output_dir, prefix="IMG", coord_x=0.0, coord_y=0.0, requested_filename=None):
        """Menyimpan gambar fisik resolusi tinggi untuk kebutuhan pengumpulan dataset."""
        frame = self.capture_frame()
        if frame is None:
            return None

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if requested_filename:
            filename = requested_filename
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}_{timestamp}_X{str(coord_x).replace('.','_')}_Y{str(coord_y).replace('.','_')}.jpg"
        
        file_path = os.path.join(output_dir, filename)

        success = cv2.imwrite(file_path, frame)
        if success:
            print(f"[CORE CAMERA SUCCESS] Gambar dataset tersimpan: {file_path}")
            return filename
        else:
            print(f"[CORE CAMERA ERROR] Gagal menulis berkas gambar ke disk: {file_path}")
            return None

    def close(self):
        """Melepas sensor kamera dan mematikan background thread dengan aman."""
        if self.started:
            self.started = False
            if self.thread:
                self.thread.join(timeout=1.0)
            if self.cap:
                self.cap.release()
                self.cap = None
            print("[CORE CAMERA] Kunci sensor hardware kamera resmi dibebaskan.")

camera_core = IMX477CameraCore()