"""B3 dan B8: soft limit."""
import pytest

from Hardware.Devices.motor import GRBLMotorCore


def _motor(tmp_path):
    m = GRBLMotorCore()
    m.position_file = str(tmp_path / "pos.json")
    m.is_mock_mode = True
    m.last_known_pos = {"X": 10.0, "Y": 10.0, "Z": 0.0}
    return m


def test_default_aktif_tanpa_batas_tidak_memotong(tmp_path, monkeypatch):
    for ax in "XYZ":
        monkeypatch.delenv(f"SOFT_LIMIT_{ax}_MIN", raising=False)
        monkeypatch.delenv(f"SOFT_LIMIT_{ax}_MAX", raising=False)
    monkeypatch.delenv("SOFT_LIMITS_ENABLED", raising=False)
    m = _motor(tmp_path)
    assert m.soft_limits_enabled is True
    assert m._clamp_axis("X", 99999.0) == (99999.0, 0)


def test_batas_dari_env_dan_flag_mati(tmp_path, monkeypatch):
    monkeypatch.setenv("SOFT_LIMIT_X_MAX", "160")
    monkeypatch.setenv("SOFT_LIMIT_Y_MIN", "bukan-angka")
    m = _motor(tmp_path)
    assert m.soft_limits["X"] == [None, 160.0]
    assert m.soft_limits["Y"] == [None, None]
    monkeypatch.setenv("SOFT_LIMITS_ENABLED", "0")
    assert _motor(tmp_path).soft_limits_enabled is False


def test_B8_gerak_relatif_dua_sumbu_dipotong(tmp_path):
    m = _motor(tmp_path)
    m.soft_limits_enabled = True
    m.soft_limits = {"X": [0.0, 12.0], "Y": [5.0, None], "Z": [None, None]}
    m.send_command("G91")
    m.send_command("G1 X5.000 Y-20.000 F250")
    assert m.last_known_pos["X"] == 12.0
    assert m.last_known_pos["Y"] == 5.0


def test_B8_gerak_relatif_dalam_batas_tidak_berubah(tmp_path):
    m = _motor(tmp_path)
    m.soft_limits_enabled = True
    m.soft_limits = {"X": [0.0, 100.0], "Y": [0.0, 100.0], "Z": [None, None]}
    m.send_command("G91")
    m.send_command("G1 X1.000 Y-1.000 F250")
    assert (m.last_known_pos["X"], m.last_known_pos["Y"]) == (11.0, 9.0)


def test_jog_tetap_dipotong_satu_kali(tmp_path):
    m = _motor(tmp_path)
    m.soft_limits_enabled = True
    m.soft_limits = {"X": [0.0, 12.0], "Y": [None, None], "Z": [None, None]}
    m.jog_from_gcode("G1 X5.000 F250")
    assert m.last_known_pos["X"] == 12.0
