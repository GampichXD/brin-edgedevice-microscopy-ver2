import cv2
import os
import json
import numpy as np

class DeepLearningColonyCounter:
    def __init__(self, model_path="./Computer_Vision/models/colony_yolov8.onnx", output_dir="../Software/backend/static/uploads"):
        self.model_path = model_path
        self.output_dir = os.path.abspath(output_dir)
        self.net = None
        self.classes = ["colony"] # Nama kelas objek sesuai dataset training TA-mu
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        self._load_yolo_model()

    def _load_yolo_model(self):
        """Memuat arsitektur model YOLO ONNX langsung ke GPU CUDA Jetson Orin Nano."""
        if not os.path.exists(self.model_path):
            print(f"[AI COUNTER WARNING] Bobot model {self.model_path} tidak ditemukan.")
            print("[AI COUNTER] Berjalan dalam MODE SIMULASI DETEKSI...")
            return

        try:
            # Membaca jaringan saraf tiruan YOLO dari berkas ONNX
            self.net = cv2.dnn.readNetFromONNX(self.model_path)
            
            # KUNCI UTAMA EDGE COMPUTING: Paksa eksekusi berjalan di GPU Jetson (Akselerasi CUDA)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print("[AI COUNTER SUCCESS] Model Deep Learning YOLO berhasil dikunci di GPU CUDA Jetson.")
        except Exception as e:
            print(f"[AI COUNTER ERROR] Gagal mengaktifkan akselerasi hardware CUDA: {e}. Fallback ke CPU.")

    def analyze_image(self, image_path, conf_threshold=0.45, nms_threshold=0.4):
        """
        Memproses gambar cawan petri hasil stitching, mendeteksi koordinat bakteri,
        dan menggambar bounding box hasil prediksi AI.
        """
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"[AI COUNTER ERROR] Gambar tidak ditemukan atau korup: {image_path}")
            return None, 0

        # Jika model tidak ada, jalankan simulasi deteksi (Mock AI) agar sistem tidak crash
        if self.net is None:
            return self._execute_mock_detection(frame, image_path)

        h_img, w_img, _ = frame.shape
        
        # Preprocessing: Ubah gambar ke format blob sesuai standar YOLO (Square resize 640x640, normalisasi 1/255)
        blob = cv2.dnn.blobFromImage(frame, 1.0 / 255.0, (640, 640), (0, 0, 0), swapRB=True, crop=False)
        self.net.setInput(blob)
        
        # Eksekusi Inferensi Deep Learning di GPU Jetson
        outputs = self.net.forward()

        # Array untuk menampung hasil lokalisasi objek bakteri
        boxes = []
        confidences = []
        class_ids = []

        # YOLOv8 ONNX output biasanya berbentuk [1, 5, 8400] -> (x, y, w, h, score)
        rows = outputs[0].shape[1] if len(outputs.shape) == 3 else outputs.shape[0]
        predictions = outputs[0] if len(outputs.shape) == 3 else outputs

        # Berjalan melintasi seluruh tensor prediksi AI
        for i in range(predictions.shape[1]):
            row = predictions[:, i]
            score = row[4] # Nilai confidence level objek bakteri
            
            if score >= conf_threshold:
                # Transformasi koordinat kembali ke ukuran asli gambar mikroskop
                x_center, y_center, w, h = row[0], row[1], row[2], row[3]
                
                # Konversi dari center-coordinates ke top-left coordinates
                x = int((x_center - w / 2) * (w_img / 640.0))
                y = int((y_center - h / 2) * (h_img / 640.0))
                width = int(w * (w_img / 640.0))
                height = int(h * (h_img / 640.0))
                
                boxes.append([x, y, width, height])
                confidences.append(float(score))
                class_ids.append(0)

        # Jalankan Non-Maximum Suppression (NMS) untuk menghapus bounding box yang tumpang tindih
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        
        total_colonies = len(indices)
        print(f"[AI COUNTER SUCCESS] Deteksi selesai. Menemukan {total_colonies} koloni sel.")

        # Gambar Bounding Box ke citra untuk output visual user
        for i in indices:
            # Kompatibilitas versi OpenCV (beberapa mereturn array bersarang atau flat)
            idx = i[0] if isinstance(i, (list, np.ndarray)) else i
            box = boxes[idx]
            x, y, w, h = box[0], box[1], box[2], box[3]
            
            # Gambar kotak penanda warna hijau di sekeliling bakteri
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Sel: {confidences[idx]:.2f}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Simpan gambar hasil prediksi Deep Learning ke folder static uploads
        output_filename = "yolo_" + os.path.basename(image_path)
        output_path = os.path.join(self.output_dir, output_filename)
        cv2.imwrite(output_path, frame)

        return output_filename, total_colonies

    def _execute_mock_detection(self, frame, image_path):
        """Simulasi hitung sel jika berkas model .onnx belum siap/luring di laptop."""
        h, w, _ = frame.shape
        # Membuat lingkaran tiruan sebagai representasi letak koordinat deteksi AI
        total_mock = 14
        for i in range(total_mock):
            cx, cy = (150 + i * 35) % w, (200 + i * 40) % h
            cv2.circle(frame, (cx, cy), 15, (0, 255, 0), 2)
            cv2.putText(frame, "Sel (Mock)", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        output_filename = "yolo_" + os.path.basename(image_path)
        cv2.imwrite(os.path.join(self.output_dir, output_filename), frame)
        return output_filename, total_mock

# Instansiasi objek tunggal
colony_counter = DeepLearningColonyCounter()