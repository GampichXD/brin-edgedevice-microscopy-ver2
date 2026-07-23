import cv2
import os
import numpy as np

class ClassicCVImageEditor:
    def __init__(self, output_dir="../Software/backend/static/uploads"):
        self.output_dir = os.path.abspath(output_dir)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def apply_brightness_contrast(self, image_path, brightness=0, contrast=0):
        """
        Mengubah kecerahan (brightness) dan kontras (contrast) gambar secara klasik.
        brightness: rentang -255 sampai 255
        contrast: rentang -127 sampai 127
        """
        img = cv2.imread(image_path)
        if img is None:
            return None

        # Rumus klasik koreksi kecerahan dan kontras matematika matriks OpenCV
        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + brightness
            alpha_b = (highlight - shadow) / 255
            gamma_b = shadow
            buf = cv2.addWeighted(img, alpha_b, img, 0, gamma_b)
        else:
            buf = img.copy()

        if contrast != 0:
            f = 131 * (contrast + 127) / (127 * (131 - contrast))
            alpha_c = f
            gamma_c = 127 * (1 - f)
            buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

        output_filename = f"edited_bc_{os.path.basename(image_path)}"
        output_path = os.path.join(self.output_dir, output_filename)
        cv2.imwrite(output_path, buf)
        return output_filename

    def apply_adaptive_threshold(self, image_path, block_size=11, c_value=2):
        """
        Mengubah citra menjadi biner (Hitam-Putih) menggunakan Gaussian Adaptive Thresholding.
        Sangat berguna untuk memisahkan objek bakteri dari latar belakang cawan petri 
        yang pencahayaannya tidak rata.
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None

        # Pastikan block size bernilai ganjil (Syarat wajib fungsi OpenCV)
        if block_size % 2 == 0:
            block_size += 1

        # Menerapkan thresholding adaptif klasik
        thresh = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, block_size, c_value
        )

        output_filename = f"edited_thresh_{os.path.basename(image_path)}"
        output_path = os.path.join(self.output_dir, output_filename)
        cv2.imwrite(output_path, thresh)
        return output_filename

    def extract_and_draw_contours(self, image_path):
        """
        Mendeteksi tepi geometri dan menggambar outline kontur luar dari objek 
        menggunakan algoritma Canny + FindContours klasik.
        """
        img = cv2.imread(image_path)
        if img is None:
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Ekstraksi filter blur untuk mereduksi noise mikroskopis
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Deteksi Tepi Canny
        edged = cv2.Canny(blurred, 30, 150)
        
        # Pencarian kontur geometris
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Gambar garis kontur berwarna merah (BGR: 0, 0, 255) di atas gambar asli
        result_img = img.copy()
        cv2.drawContours(result_img, contours, -1, (0, 0, 255), 2)

        output_filename = f"edited_contours_{os.path.basename(image_path)}"
        output_path = os.path.join(self.output_dir, output_filename)
        cv2.imwrite(output_path, result_img)
        return output_filename

    def apply_sobel_edge(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None: return None
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        sobel_combined = cv2.magnitude(sobelx, sobely)
        sobel_combined = np.uint8(np.clip(sobel_combined, 0, 255))
        output_filename = f"edited_sobel_{os.path.basename(image_path)}"
        cv2.imwrite(os.path.join(self.output_dir, output_filename), sobel_combined)
        return output_filename

    def calculate_morphology(self, image_path):
        import json
        img = cv2.imread(image_path)
        if img is None: return None, None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Binarisasi adaptif
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        total_area = 0
        total_perimeter = 0
        valid_cells = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 15: # Filter mikro noise
                perimeter = cv2.arcLength(cnt, True)
                total_area += area
                total_perimeter += perimeter
                valid_cells += 1
                cv2.drawContours(img, [cnt], -1, (255, 0, 0), 2)
        
        output_filename = f"edited_morphology_{os.path.basename(image_path)}"
        cv2.imwrite(os.path.join(self.output_dir, output_filename), img)
        
        stats = {
            "total_cells": valid_cells,
            "avg_area": round(total_area / valid_cells if valid_cells > 0 else 0, 2),
            "avg_perimeter": round(total_perimeter / valid_cells if valid_cells > 0 else 0, 2)
        }
        return output_filename, stats

    def auto_roi_crop(self, image_path):
        img = cv2.imread(image_path)
        if img is None: return None
        h, w = img.shape[:2]
        ch, cw = h // 2, w // 2
        crop_h, crop_w = h // 2, w // 2
        y1, y2 = ch - crop_h // 2, ch + crop_h // 2
        x1, x2 = cw - crop_w // 2, cw + crop_w // 2
        roi = img[y1:y2, x1:x2]
        output_filename = f"edited_roi_{os.path.basename(image_path)}"
        cv2.imwrite(os.path.join(self.output_dir, output_filename), roi)
        return output_filename

    def draw_scale_calibration(self, image_path):
        img = cv2.imread(image_path)
        if img is None: return None
        h, w = img.shape[:2]
        bar_length = w // 5
        x1, y1 = 20, h - 40
        x2, y2 = x1 + bar_length, h - 20
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 2)
        cv2.putText(img, "100 um", (x1 + 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
        cv2.putText(img, "100 um", (x1 + 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        output_filename = f"edited_calibrate_{os.path.basename(image_path)}"
        cv2.imwrite(os.path.join(self.output_dir, output_filename), img)
        return output_filename

    def split_color_channels(self, image_path):
        img = cv2.imread(image_path)
        if img is None: return None
        b, g, r = cv2.split(img)
        zeros = np.zeros_like(b)
        img_r = cv2.merge([zeros, zeros, r])
        img_g = cv2.merge([zeros, g, zeros])
        img_b = cv2.merge([b, zeros, zeros])
        
        top = np.hstack((img, img_r))
        bottom = np.hstack((img_g, img_b))
        grid = np.vstack((top, bottom))
        
        # Resize jika grid terlalu besar
        h, w = grid.shape[:2]
        if max(h, w) > 2000:
            scale = 2000 / max(h, w)
            grid = cv2.resize(grid, (0, 0), fx=scale, fy=scale)
            
        output_filename = f"edited_colorsplit_{os.path.basename(image_path)}"
        cv2.imwrite(os.path.join(self.output_dir, output_filename), grid)
        return output_filename

# Instansiasi objek tunggal editor citra klasik
classic_cv_editor = ClassicCVImageEditor()