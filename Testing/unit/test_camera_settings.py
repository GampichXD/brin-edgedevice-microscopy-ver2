"""Pengaturan kamera: nilai dipotong ke batas sensor, dan kamera tidak dibiarkan mati
kalau pipeline gagal start dengan nilai baru (sebelumnya perlu restart main.py)."""
from Hardware.Devices import camera as cam


def _fake_camera(monkeypatch, fail_when=None):
    c = cam.IMX477CameraCore()
    c.shutter_speed, c.iso = 15000, 200
    c.started, c.proc = True, object()  # pipeline sedang jalan
    calls = []

    def fake_close():
        calls.append(("close", c.shutter_speed, c.iso))
        c.started, c.proc = False, None

    def fake_open():
        ok = not (fail_when and fail_when(c.shutter_speed, c.iso))
        calls.append(("open", c.shutter_speed, c.iso, ok))
        if ok:
            c.started, c.proc = True, object()
        return ok

    monkeypatch.setattr(c, "close", fake_close)
    monkeypatch.setattr(c, "open", fake_open)
    monkeypatch.setattr(cam.time, "sleep", lambda s: None)
    return c, calls


def test_nilai_dipotong_ke_batas_sensor(monkeypatch):
    c, _ = _fake_camera(monkeypatch)
    assert c.apply_settings(50000, 5000) is True
    assert (c.shutter_speed, c.iso) == (cam.SHUTTER_MAX_US, cam.ISO_MAX)
    c.apply_settings(1, 10)
    assert (c.shutter_speed, c.iso) == (cam.SHUTTER_MIN_US, cam.ISO_MIN)


def test_gagal_start_kembali_ke_nilai_lama(monkeypatch):
    c, calls = _fake_camera(monkeypatch, fail_when=lambda sh, iso: sh == 20000)
    assert c.apply_settings(20000, 400) is False
    assert (c.shutter_speed, c.iso) == (15000, 200)
    assert calls[-1] == ("open", 15000, 200, True)  # kamera hidup lagi dengan nilai lama
    assert c.applying is False


def test_kamera_mati_dibuka_lagi_saat_terapkan(monkeypatch):
    c, calls = _fake_camera(monkeypatch)
    c.started, c.proc = False, None  # pipeline sudah mati sebelumnya
    assert c.apply_settings(15000, 200) is True
    assert calls[-1] == ("open", 15000, 200, True)
