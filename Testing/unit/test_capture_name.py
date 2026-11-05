"""B6: nama file capture tidak boleh keluar dari folder tujuan."""
import os

import pytest

from Hardware.Devices.camera import camera_core


@pytest.fixture
def cam(monkeypatch):
    monkeypatch.setattr(camera_core, "capture_jpeg_bytes", lambda: b"\xff\xd8\xff\xd9")
    return camera_core


def test_nama_valid_dipakai_apa_adanya(cam, tmp_path):
    assert cam.save_snapshot(str(tmp_path), requested_filename="IMG_MANUAL_1.jpg") == "IMG_MANUAL_1.jpg"


@pytest.mark.parametrize("name", ["../../evil.sh", "a b.jpg", "x.py", "..jpg/"])
def test_nama_tidak_valid_diganti_nama_otomatis(cam, tmp_path, name):
    out = tmp_path / "out"
    saved = cam.save_snapshot(str(out), requested_filename=name)
    assert saved.startswith("IMG_")
    assert os.listdir(out) == [saved]
    assert sorted(os.listdir(tmp_path)) == ["out"]


def test_traversal_hanya_memakai_basename(cam, tmp_path):
    out = tmp_path / "out"
    assert cam.save_snapshot(str(out), requested_filename="../../ok.jpg") == "ok.jpg"
    assert sorted(os.listdir(tmp_path)) == ["out"]
