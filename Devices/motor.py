import serial
import serial.tools.list_ports
import platform
import time
import re
import json
import os

class GRBLMotorCore:
    def __init__(self, baudrate=115200, timeout=1):
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self.serial_lock = __import__('threading').Lock()
        self.is_mock_mode = False  # Flag simulasi jika tidak terhubung ke Arduino asli
        self.is_relative_mode = False # Melacak G90/G91
        self.last_known_pos = {"X": 0.0, "Y": 0.0, "Z": 0.0}
        self.motion_settings = {
            "feed_rate": 250.0,
            "backlash": 0.0,
            "acceleration": 10.0,
            "settle_time": 0,
        }
        self.last_axis_direction = {"X": None, "Y": None, "Z": None}
        self.position_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motor_position.json")
        self.last_status = "Idle"
        self._last_saved_pos = None  # snapshot terakhir yang sudah ditulis ke JSON (throttle)
        self._load_position()

    def _save_position(self):
        try:
            with open(self.position_file, "w") as f:
                json.dump(self.last_known_pos, f)
            print(f"[CORE MOTOR] Berhasil menyimpan koordinat ke: {self.position_file}")
        except Exception as e:
            print(f"[CORE MOTOR ERROR] Gagal menyimpan posisi ke JSON: {e}")

    def _persist_if_changed(self):
        """Tulis last_known_pos ke motor_position.json, hanya jika berubah sejak
        penulisan terakhir. Dipanggil dari SEMUA jalur pergerakan (jog relatif,
        move absolut, homing) dan dari setiap polling get_status(), sehingga isi
        file selalu identik dengan koordinat yang sedang tampil di web."""
        snapshot = {
            "X": round(float(self.last_known_pos.get("X", 0.0)), 3),
            "Y": round(float(self.last_known_pos.get("Y", 0.0)), 3),
            "Z": round(float(self.last_known_pos.get("Z", 0.0)), 3),
        }
        if self._last_saved_pos != snapshot:
            self._save_position()
            self._last_saved_pos = snapshot

    def _track_position_from_gcode(self, clean_cmd: str):
        """Perbarui last_known_pos dari perintah G0/G1 (absolut, atau relatif saat
        is_relative_mode aktif via G91), lalu persist. Dipakai baik mode mock
        maupun serial fisik supaya koordinat konsisten di semua kondisi."""
        upper = clean_cmd.upper()
        if not (upper.startswith("G1 ") or upper.startswith("G0 ")):
            return
        changed = False
        for ax in ("X", "Y", "Z"):
            m = re.search(rf"{ax}([-\d.]+)", clean_cmd, re.IGNORECASE)
            if not m:
                continue
            val = float(m.group(1))
            if self.is_relative_mode:
                self.last_known_pos[ax] = round(self.last_known_pos.get(ax, 0.0) + val, 3)
            else:
                self.last_known_pos[ax] = round(val, 3)
            changed = True
        if changed:
            self._persist_if_changed()

    def _load_position(self):
        try:
            if os.path.exists(self.position_file):
                with open(self.position_file, "r") as f:
                    pos = json.load(f)
                    self.last_known_pos = {
                        "X": float(pos.get("X", 0.0)),
                        "Y": float(pos.get("Y", 0.0)),
                        "Z": float(pos.get("Z", 0.0))
                    }
                print(f"[CORE MOTOR] Memuat koordinat terakhir dari JSON: {self.last_known_pos}")
        except Exception as e:
            print(f"[CORE MOTOR ERROR] Gagal memuat posisi: {e}")
        # Seed snapshot agar tidak langsung menulis ulang nilai yang baru dimuat
        self._last_saved_pos = {
            "X": round(float(self.last_known_pos.get("X", 0.0)), 3),
            "Y": round(float(self.last_known_pos.get("Y", 0.0)), 3),
            "Z": round(float(self.last_known_pos.get("Z", 0.0)), 3),
        }

    def _find_hardware_port(self):
        """Mencari port fisik USB-Serial Arduino CNC Shield dengan proteksi link Bluetooth."""
        current_os = platform.system()
        ports = list(serial.tools.list_ports.comports())
        
        if current_os == "Windows":
            for port in ports:
                desc = port.description.lower()
                hwid = port.hwid.lower()
                if "bluetooth" in desc or "bthnum" in hwid or "standard serial over" in desc:
                    continue
                if "com" in port.device.lower() and ("ch340" in desc or "arduino" in desc or "usb" in desc or "serial" in desc):
                    print(f"[CORE MOTOR] Terdeteksi Arduino CNC Shield di: {port.device} ({port.description})")
                    return port.device
            return None
        else:
            for port in ports:
                if "ttyUSB" in port.device or "ttyACM" in port.device:
                    print(f"[CORE MOTOR] Terdeteksi port serial Linux Jetson: {port.device}")
                    return port.device
            return "/dev/ttyUSB0"

    def open(self):
        """Membuka jalur serial port murni menuju kontroler mesin."""
        if self.ser and self.ser.is_open:
            return True

        target_port = self._find_hardware_port()
        
        if target_port is None and platform.system() == "Windows":
            print("[CORE MOTOR WARNING] Perangkat keras tidak ditemukan di port Windows COM.")
            print("[CORE MOTOR] Mengaktifkan MODE SIMULASI LURING untuk core hardware script...")
            self.is_mock_mode = True
            return True

        try:
            print(f"[CORE MOTOR] Membuka sirkuit komunikasi serial ke {target_port}...")
            self.ser = serial.Serial(target_port, self.baudrate, timeout=self.timeout)
            
            time.sleep(2)  # Antisipasi Arduino reboot otomatis akibat DTR pin pull
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            
            self.ser.write(b"\r\n\r\n")
            time.sleep(0.5)
            
            # Kuras baris sambutan selamat datang dari GRBL (welcome message)
            while self.ser.in_waiting:
                self.ser.readline()
                
            # Sinkronisasi koordinat yang sudah dimuat ke GRBL
            self.send_command(f"G92 X{self.last_known_pos['X']} Y{self.last_known_pos['Y']} Z{self.last_known_pos['Z']}")
                
            print("[CORE MOTOR SUCCESS] Mesin CNC berhasil terhubung penuh secara fisik.")
            self.is_mock_mode = False
            return True
        except Exception as e:
            print(f"[CORE MOTOR ERROR] Gagal mengunci port fisik serial: {e}")
            print("[CORE MOTOR] Mengalihkan sistem ke MODE SIMULASI LURING...")
            self.is_mock_mode = True
            return False

    def send_command(self, command: str) -> str:
        """Mengirimkan G-Code dan membaca semua baris respon dari GRBL hingga tuntas."""
        clean_cmd = command.strip()
        if not clean_cmd:
            return "N/A"

        # Lacak mode koordinat modal GRBL (G90 absolut / G91 relatif) untuk
        # SEMUA mode (mock maupun serial fisik).
        if clean_cmd == "G91":
            self.is_relative_mode = True
        elif clean_cmd == "G90":
            self.is_relative_mode = False

        if self.is_mock_mode:
            print(f"[MOCK MOTOR] Menembak: {clean_cmd} -> Balasan: ok")
            self._track_position_from_gcode(clean_cmd)
            return "ok"

        if self.ser is None or not self.ser.is_open:
            return "ERROR: Jalur komunikasi serial tidak aktif."

        try:
            with self.serial_lock:
                self.ser.write(f"{clean_cmd}\n".encode('utf-8'))
                
                responses = []
                timeout_start = time.time()
                
                # Membaca berbaris-baris respon GRBL sampai menerima kata 'ok' atau 'error:'
                while True:
                    if self.ser.in_waiting or (time.time() - timeout_start < self.timeout):
                        line = self.ser.readline().decode('utf-8').strip()
                        if line:
                            responses.append(line)
                            if line == "ok" or line.startswith("error:"):
                                break
                    else:
                        break # Terkena timeout proteksi serial
                        
            full_response = " | ".join(responses)
            print(f"[CORE MOTOR SERIAL] Kirim: {clean_cmd} | Respon: {full_response}")

            # GROUND TRUTH: kadang laporan status GRBL '<...WPos:x,y,z...>' ikut
            # terbaca di respon (mis. akibat query '?' dari thread telemetri yang
            # ter-interleave). Jika ada, pakai posisi ITU — lebih andal daripada
            # menebak dari G-code yang kita kirim (yang bisa saja 'G1 X0 Y0' keliru).
            pos_from_grbl = None
            for ln in responses:
                m = (re.search(r'WPos:(-?[\d.]+),(-?[\d.]+),(-?[\d.]+)', ln)
                     or re.search(r'MPos:(-?[\d.]+),(-?[\d.]+),(-?[\d.]+)', ln))
                if m:
                    pos_from_grbl = {"X": float(m.group(1)), "Y": float(m.group(2)), "Z": float(m.group(3))}

            if pos_from_grbl is not None:
                self.last_known_pos = pos_from_grbl
                self._persist_if_changed()
            elif not full_response.lower().startswith("error"):
                # Tidak ada echo status -> sinkronkan dari G-code gerak yang dikirim.
                self._track_position_from_gcode(clean_cmd)

            return full_response if full_response else "ok"
            
        except Exception as e:
            print(f"[CORE MOTOR ERROR] Gagal melakukan operasi I/O Serial: {e}")
            return f"ERROR: {e}"

    def apply_motion_settings(self, feed_rate=None, backlash=None, acceleration=None, settle_time=None):
        if feed_rate is not None:
            self.motion_settings["feed_rate"] = float(feed_rate)
        if backlash is not None:
            self.motion_settings["backlash"] = max(0.0, float(backlash))
        if acceleration is not None:
            self.motion_settings["acceleration"] = float(acceleration)
        if settle_time is not None:
            self.motion_settings["settle_time"] = max(0, int(settle_time))

        if self.is_mock_mode:
            print(f"[MOCK MOTOR] Motion settings updated: {self.motion_settings}")
            return "ok"

        responses = [
            self.send_command(f"$120={self.motion_settings['acceleration']}"),
            self.send_command(f"$121={self.motion_settings['acceleration']}"),
            self.send_command(f"$122={self.motion_settings['acceleration']}"),
        ]
        print(f"[CORE MOTOR] Motion settings updated: {self.motion_settings}")
        return " | ".join(responses)

    def jog_relative(self, axis: str, delta_mm: float, feedrate=None):
        axis = axis.upper()
        if axis not in {"X", "Y", "Z"}:
            return f"ERROR: Axis tidak valid ({axis})"

        effective_delta = float(delta_mm)
        direction = "+" if effective_delta >= 0 else "-"
        backlash = self.motion_settings.get("backlash", 0.0)
        last_direction = self.last_axis_direction.get(axis)

        if axis in {"X", "Y"} and backlash > 0 and last_direction and last_direction != direction:
            effective_delta += backlash if effective_delta > 0 else -backlash

        feedrate_value = feedrate if feedrate is not None else self.motion_settings.get("feed_rate", 250.0)
        self.last_axis_direction[axis] = direction

        # last_known_pos + persist ke JSON diurus oleh send_command() di bawah,
        # lewat logika modal G91 (berlaku untuk mode mock maupun serial fisik).
        response = self.send_command("G91")
        if response.startswith("ERROR"):
            return response

        jog_response = self.send_command(f"G1 {axis}{effective_delta:.3f} F{feedrate_value}")
        self.send_command("G90")

        settle_time = self.motion_settings.get("settle_time", 0)
        if settle_time > 0:
            time.sleep(settle_time / 1000.0)

        return jog_response

    def jog_from_gcode(self, command: str):
        # Gunakan strict regex ^...$ agar hanya mencegat single-axis G1 (joystick)
        # Jika multi-axis (contoh: G1 X10 Y20), match akan gagal dan diteruskan utuh ke send_command (sebagai absolute move)
        match = re.search(r"^G1\s+([XYZ])\s*([+-]?[\d.]+)(?:\s+F([\d.]+))?$", command.strip(), re.IGNORECASE)
        if not match:
            return self.send_command(command)

        axis = match.group(1).upper()
        delta = float(match.group(2))
        feedrate = float(match.group(3)) if match.group(3) else None
        return self.jog_relative(axis, delta, feedrate)

    def get_status(self):
        """
        Mengirimkan perintah real-time query '?' ke GRBL tanpa mengganggu pergerakan.
        Berguna untuk parsing koordinat XYZ real-time untuk dilempar ke VPS.
        """
        if self.is_mock_mode:
            self._persist_if_changed()
            return {"status": "Idle", "X": self.last_known_pos.get("X", 0.0), "Y": self.last_known_pos.get("Y", 0.0), "Z": self.last_known_pos.get("Z", 0.0), "limit_switch": "N/A"}
            
        if self.ser is None or not self.ser.is_open:
            return {"status": "Offline", "X": self.last_known_pos.get("X", 0.0), "Y": self.last_known_pos.get("Y", 0.0), "Z": self.last_known_pos.get("Z", 0.0), "limit_switch": "N/A"}
            
        try:
            response = ""
            with self.serial_lock:
                self.ser.write(b"?")  # Karakter status query instan GRBL
                
                # Gunakan blocking readline (dilindungi timeout serial)
                timeout_start = time.time()
                while time.time() - timeout_start < 0.2:
                    if self.ser.in_waiting:
                        line = self.ser.readline().decode('utf-8').strip()
                        if line.startswith("<"):
                            response = line
                            break
                    else:
                        time.sleep(0.01)
                        
            if not response:
                return {"status": "Run", "X": self.last_known_pos.get("X", 0.0), "Y": self.last_known_pos.get("Y", 0.0), "Z": self.last_known_pos.get("Z", 0.0), "limit_switch": "N/A"}
                
            # Cari status utama (Idle, Run, dll)
            match_status = re.search(r'<(.*?)[,|]', response)
            new_status = match_status.group(1) if match_status else "Unknown"
            
            # Prioritaskan WPos, jika tidak ada baru gunakan MPos
            match_pos = re.search(r'WPos:([-\d.]+),([-\d.]+),([-\d.]+)', response) or re.search(r'MPos:([-\d.]+),([-\d.]+),([-\d.]+)', response)
            
            match_pn = re.search(r'Pn:([^>|,]+)', response)
            limit_switch = match_pn.group(1) if match_pn else "N/A"

            if match_pos:
                self.last_known_pos = {
                    "X": float(match_pos.group(1)),
                    "Y": float(match_pos.group(2)),
                    "Z": float(match_pos.group(3)),
                }

                # Selalu cerminkan posisi laporan GRBL ke motor_position.json
                # (tanpa filter Run/Jog) supaya isi file = koordinat yang tampil
                # di web. _persist_if_changed() sudah men-throttle penulisan.
                self._persist_if_changed()

                self.last_status = new_status
                
                return {
                    "status": new_status,
                    "X": self.last_known_pos["X"],
                    "Y": self.last_known_pos["Y"],
                    "Z": self.last_known_pos["Z"],
                    "limit_switch": limit_switch,
                }
        except Exception:
            pass
        return {"status": "Run", "X": self.last_known_pos.get("X", 0.0), "Y": self.last_known_pos.get("Y", 0.0), "Z": self.last_known_pos.get("Z", 0.0), "limit_switch": "N/A"}

    def move_xyz(self, x=None, y=None, z=None, feedrate=250):
        """Fungsi pembantu berlevel tinggi khusus untuk pergerakan interpolasi linier."""
        gcode_parts = ["G1"]
        if x is not None: gcode_parts.append(f"X{x}")
        if y is not None: gcode_parts.append(f"Y{y}")
        if z is not None: gcode_parts.append(f"Z{z}")
        gcode_parts.append(f"F{feedrate}")

        # move_xyz selalu absolut; pastikan mode modal benar sebelum mengirim.
        if self.is_relative_mode:
            self.send_command("G90")

        # send_command() akan meng-update last_known_pos + persist ke JSON.
        gcode_str = " ".join(gcode_parts)
        return self.send_command(gcode_str)

    def homing(self):
        self.last_known_pos = {"X": 0.0, "Y": 0.0, "Z": 0.0}
        self.is_relative_mode = False
        self._persist_if_changed()
        if self.is_mock_mode:
            print("[MOCK MOTOR] Motor kembali ke posisi 0 (0, 0, 0)")
            return "ok"
        self.send_command("G90")
        return self.send_command("G0 X0 Y0 Z0")

    def unlock(self):
        return self.send_command("$X")

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            self.ser = None
            print("[CORE MOTOR] Sirkuit serial perangkat dibebaskan murni.")

motor_core = GRBLMotorCore()