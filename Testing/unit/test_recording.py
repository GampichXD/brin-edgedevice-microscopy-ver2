"""Rekaman: MJPG dikonversi ke H.264 MP4, file final baru muncul setelah utuh."""
import os
import shutil
import time

import cv2
import numpy as np
import pytest

from Hardware.Devices.camera import IMX477CameraCore
from Hardware.Handlers.camera_handler import _finalize_recording

needs_gst = pytest.mark.skipif(shutil.which("gst-launch-1.0") is None, reason="GStreamer tidak ada")


def _avi(path, frames=30):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (320, 240))
    for i in range(frames):
        frame = np.full((240, 320, 3), i * 5 % 255, np.uint8)
        writer.write(frame)
    writer.release()


@needs_gst
def test_hasil_final_h264_baseline_dan_sumber_terhapus(tmp_path):
    src, dst = tmp_path / "_temp_v.avi", tmp_path / "v.mp4"
    _avi(src)
    assert _finalize_recording(str(src), str(dst)) is True
    assert not src.exists() and dst.exists()
    assert not (tmp_path / "_temp_v.mp4").exists()
    data = dst.read_bytes()
    assert b"avc1" in data
    # moov di depan mdat: browser bisa langsung memutar tanpa mengunduh seluruh file
    assert data.index(b"moov") < data.index(b"mdat")
    video = cv2.VideoCapture(str(dst))
    assert int(video.get(cv2.CAP_PROP_FRAME_COUNT)) == 30


def test_konversi_gagal_rekaman_mentah_tetap_ada(tmp_path, monkeypatch):
    from Hardware.Handlers import camera_handler

    def boom(src, dst):
        raise RuntimeError("encoder rusak")
    monkeypatch.setattr(camera_handler, "_transcode_to_h264", boom)
    src, dst = tmp_path / "_temp_v.avi", tmp_path / "v.mp4"
    _avi(src, 3)
    assert _finalize_recording(str(src), str(dst)) is False
    assert src.exists() and not dst.exists()


def test_rekaman_kosong_dibuang(tmp_path):
    src = tmp_path / "_temp_v.avi"
    src.write_bytes(b"")
    assert _finalize_recording(str(src), str(tmp_path / "v.mp4")) is False
    assert not src.exists()


def test_kamera_dengan_frame_basi_dianggap_mati(tmp_path):
    cam = IMX477CameraCore()
    assert cam.is_alive() is False
    cam.started = True
    cam.jpeg_frame = b"\xff\xd8\xff\xd9"
    cam.last_frame_time_ns = time.time_ns()
    assert cam.is_alive() is True
    assert cam.save_snapshot(str(tmp_path), requested_filename="baru.jpg") == "baru.jpg"
    cam.last_frame_time_ns = time.time_ns() - 6_000_000_000
    assert cam.is_alive() is False
    assert cam.save_snapshot(str(tmp_path), requested_filename="basi.jpg") is None
    cam.started = False
