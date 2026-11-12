"""B2: feed hold dan pelepasannya, memakai serial palsu."""
import pytest

from Hardware.Devices.motor import GRBLMotorCore


class FakeSerial:
    is_open = True
    in_waiting = 0

    def __init__(self):
        self.writes = []
        self.lines = []

    def write(self, data):
        self.writes.append(data)
        if data == b"?":
            self.lines.append(b"<Hold:0|WPos:1.500,2.000,3.000|FS:0,0>\r\n")
            self.in_waiting = 1
        elif data == b"\x18":
            self.lines.append(b"Grbl 1.1h ['$' for help]\r\n")
        elif data.endswith(b"\n"):
            self.lines.append(b"ok\r\n")
            self.in_waiting = 1

    def readline(self):
        self.in_waiting = 0
        return self.lines.pop(0) if self.lines else b""

    def sent_lines(self):
        return [w.decode().strip() for w in self.writes if w.endswith(b"\n")]


@pytest.fixture
def motor(tmp_path):
    m = GRBLMotorCore()
    m.position_file = str(tmp_path / "pos.json")
    m.last_known_pos = {"X": 0.0, "Y": 0.0, "Z": 0.0}
    m.ser = FakeSerial()
    return m


def test_feed_hold_mengirim_byte_realtime_tanpa_newline(motor):
    assert motor.feed_hold() is True
    assert motor.ser.writes == [b"!"]
    assert motor.hold_latched is True


def test_jog_cancel_mengirim_0x85(motor):
    assert motor.jog_cancel() is True
    assert motor.ser.writes == [b"\x85"]


def test_feed_hold_tidak_menunggu_serial_lock(motor):
    with motor.serial_lock:
        assert motor.feed_hold() is True


def test_tidak_pernah_resume_otomatis(motor):
    motor.feed_hold()
    motor.jog_from_gcode("G1 X1.000 F250")
    assert b"~" not in motor.ser.writes


def test_perintah_baru_melepas_hold_lewat_soft_reset(motor):
    motor.is_relative_mode = True
    motor.feed_hold()
    motor.jog_from_gcode("G1 X1.000 F250")
    assert motor.hold_latched is False
    assert b"\x18" in motor.ser.writes
    lines = motor.ser.sent_lines()
    # Offset G92 dipulihkan dari posisi terakhir GRBL sebelum gerak baru dikirim.
    assert lines[:2] == ["$X", "G92 X1.5 Y2.0 Z3.0"]
    assert lines[2:] == ["G91", "G1 X1.000 F250.0", "G90"]


def test_tanpa_hold_tidak_ada_soft_reset(motor):
    motor.jog_from_gcode("G1 X1.000 F250")
    assert b"\x18" not in motor.ser.writes


def test_mode_mock_tidak_butuh_serial(tmp_path):
    m = GRBLMotorCore()
    m.position_file = str(tmp_path / "pos.json")
    m.is_mock_mode = True
    assert m.feed_hold() is True
    m.clear_hold()
    assert m.hold_latched is False


def test_B4_cmd_csv_mencatat_waktu(tmp_path, monkeypatch):
    import asyncio
    import csv
    from Hardware.Devices import motor as motor_mod
    from Hardware.Handlers import motor_handler
    from Hardware.Utils import test_logger
    monkeypatch.setattr(test_logger, "LOG_DIR", str(tmp_path))
    monkeypatch.setenv("TEST_LOGGING", "1")
    monkeypatch.setattr(motor_handler.motor_core, "jog_from_gcode", lambda g: "ok")

    class Sock:
        async def send(self, text):
            pass

    asyncio.run(motor_handler.handle_motor_action(
        "MOVE_MOTOR", {"gcode": "G1 X0.05 F250", "cmd_id": 7, "_t_rx_ns": 123}, Sock(), asyncio.Lock()))
    with open(tmp_path / "cmd.csv") as f:
        row = next(csv.DictReader(f))
    assert list(row) == ["cmd_id", "gcode", "t_rx_ns", "t_ok_ns"]
    assert row["cmd_id"] == "7" and row["t_rx_ns"] == "123" and int(row["t_ok_ns"]) > 123


def test_tanpa_grbl_tidak_ada_simulasi_diam_diam(tmp_path, monkeypatch):
    monkeypatch.delenv("ALLOW_MOCK_HARDWARE", raising=False)
    monkeypatch.setattr(GRBLMotorCore, "_find_hardware_port", lambda self: "/dev/port-tidak-ada")
    m = GRBLMotorCore()
    m.position_file = str(tmp_path / "pos.json")
    m.last_known_pos = {"X": 5.0, "Y": 6.0, "Z": 7.0}
    assert m.open() is False and m.is_mock_mode is False
    assert m.get_status()["status"] == "Offline"
    assert m.jog_from_gcode("G1 X1.000 F250").startswith("ERROR")
    assert m.homing().startswith("ERROR")
    assert m.set_position(0, 0, 0).startswith("ERROR")
    assert m.last_known_pos == {"X": 5.0, "Y": 6.0, "Z": 7.0}


def test_simulasi_hanya_kalau_diminta(tmp_path, monkeypatch):
    monkeypatch.setenv("ALLOW_MOCK_HARDWARE", "1")
    monkeypatch.setattr(GRBLMotorCore, "_find_hardware_port", lambda self: "/dev/port-tidak-ada")
    m = GRBLMotorCore()
    m.position_file = str(tmp_path / "pos.json")
    m.open()
    assert m.is_mock_mode is True
    assert m.jog_from_gcode("G1 X1.000 F250") == "ok"
