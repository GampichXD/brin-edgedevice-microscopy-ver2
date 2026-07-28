import serial
import serial.tools.list_ports
import platform
import time
import re

class GRBLMotorCore:
    def __init__(self, baudrate=115200, timeout=1):
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self.is_mock_mode = False  # Flag simulasi jika tidak terhubung ke Arduino asli
        self.last_known_pos = {"X": 0.0, "Y": 0.0, "Z": 0.0}
        self.motion_settings = {
            "feed_rate": 250.0,
            "backlash": 0.0,
            "acceleration": 10.0,
            "settle_time": 0,
        }
        self.last_axis_direction = {"X": None, "Y": None, "Z": None}

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

        if self.is_mock_mode:
            print(f"[MOCK MOTOR] Menembak: {clean_cmd} -> Balasan: ok")
            return "ok"

        if self.ser is None or not self.ser.is_open:
            return "ERROR: Jalur komunikasi serial tidak aktif."

        try:
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
        self.last_known_pos[axis] = round(self.last_known_pos.get(axis, 0.0) + effective_delta, 3)

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
        match = re.search(r"G1\s+([XYZ])\s*([+-]?[\d.]+)(?:\s+F([\d.]+))?", command.strip(), re.IGNORECASE)
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
            return {"status": "Idle", "X": self.last_known_pos.get("X", 0.0), "Y": self.last_known_pos.get("Y", 0.0), "Z": self.last_known_pos.get("Z", 0.0), "limit_switch": "N/A"}
            
        if self.ser is None or not self.ser.is_open:
            return {"status": "Offline", "X": self.last_known_pos.get("X", 0.0), "Y": self.last_known_pos.get("Y", 0.0), "Z": self.last_known_pos.get("Z", 0.0), "limit_switch": "N/A"}
            
        try:
            self.ser.write(b"?")  # Karakter status query instan GRBL
            time.sleep(0.05)
            response = self.ser.readline().decode('utf-8').strip()
            
            # Regex untuk membedah data koordinat MPos atau WPos dari GRBL: <Idle|MPos:0.000,0.000,0.000|...>
            match = re.search(r'<(.*?)\|(?:MPos|WPos):([-\d.]+),([-\d.]+),([-\d.]+)(?:.*?Pn:([^>|]+))?', response)
            if match:
                limit_switch = match.group(5) if match.group(5) else "N/A"
                self.last_known_pos = {
                    "X": float(match.group(2)),
                    "Y": float(match.group(3)),
                    "Z": float(match.group(4)),
                }
                return {
                    "status": match.group(1),
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
        
        if x is not None: self.last_known_pos["X"] = float(x)
        if y is not None: self.last_known_pos["Y"] = float(y)
        if z is not None: self.last_known_pos["Z"] = float(z)
            
        gcode_str = " ".join(gcode_parts)
        return self.send_command(gcode_str)

    def homing(self):
        self.last_known_pos = {"X": 0.0, "Y": 0.0, "Z": 0.0}
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