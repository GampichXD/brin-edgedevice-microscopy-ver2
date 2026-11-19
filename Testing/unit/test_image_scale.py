"""Skala µm/px untuk luas koloni: FOV tile kamera, tag JPEG, dan olahan CV klasik."""
import cv2
import numpy as np
import pytest

from Hardware.Computer_Vision import image_scale


def _jpg(path, w=900, h=506):
    cv2.imwrite(str(path), np.full((h, w, 3), 128, np.uint8))
    return str(path)


def test_tile_kamera_900x506_piksel_persegi_7um():
    assert image_scale.tile_scale_for_dims(900, 506) == pytest.approx((7.0, 7.0))
    assert image_scale.TILE_FOV_MM_X == pytest.approx(6.3)
    assert image_scale.TILE_FOV_MM_Y == pytest.approx(3.542)


def test_frame_kamera_resolusi_lain_diskalakan_dari_lebar():
    assert image_scale.tile_scale_for_dims(1800, 1012) == pytest.approx((3.5, 3.5))
    assert image_scale.tile_scale_for_dims(450, 253) == pytest.approx((14.0, 14.0))


def test_gambar_bukan_frame_kamera_tidak_terkalibrasi():
    assert image_scale.tile_scale_for_dims(2048, 2048) is None
    assert image_scale.tile_scale_for_dims(4056, 3040) is None


def test_tag_jpeg_bolak_balik_dan_gambar_tetap_terbaca(tmp_path):
    p = _jpg(tmp_path / "pano.jpg", 3000, 2000)
    assert image_scale.read_scale(p) is None
    assert image_scale.write_scale(p, 14.0, 15.02, "stitch uji")
    assert image_scale.read_scale(p) == (14.0, 15.02, "stitch uji")
    assert cv2.imread(p).shape == (2000, 3000, 3)


def test_tag_ditimpa_bukan_digandakan(tmp_path):
    p = _jpg(tmp_path / "a.jpg")
    image_scale.write_scale(p, 1.0, 1.0, "lama")
    size = len(open(p, "rb").read())
    image_scale.write_scale(p, 2.0, 3.0, "baru")
    assert image_scale.read_scale(p) == (2.0, 3.0, "baru")
    assert abs(len(open(p, "rb").read()) - size) < 8


def test_urutan_resolve_override_tag_fov(tmp_path):
    p = _jpg(tmp_path / "tile.jpg")
    assert image_scale.resolve_scale(p, 900, 506)["um_per_px_x"] == pytest.approx(7.0)
    image_scale.write_scale(p, 14.0, 15.02, "stitch")
    assert image_scale.resolve_scale(p, 900, 506)["source"] == "stitch"
    assert image_scale.resolve_scale(p, 900, 506, override=(5.0, None)) == {
        "um_per_px_x": 5.0, "um_per_px_y": 5.0, "source": "manual"}


def test_bukan_jpeg_tidak_rusak(tmp_path):
    p = tmp_path / "x.png"
    cv2.imwrite(str(p), np.zeros((10, 10, 3), np.uint8))
    before = p.read_bytes()
    assert image_scale.write_scale(str(p), 1, 1, "x") is False
    assert p.read_bytes() == before and image_scale.read_scale(str(p)) is None


def test_olahan_cv_klasik_mempertahankan_skala(tmp_path):
    from Hardware.Computer_Vision.classic_cv_edit import ClassicCVImageEditor
    src = _jpg(tmp_path / "pano.jpg", 1200, 800)
    image_scale.write_scale(src, 14.0, 15.02, "stitch")
    ed = ClassicCVImageEditor(output_dir=str(tmp_path / "out"))
    out = ed.auto_roi_crop(src, {"x": 0.1, "y": 0.1, "w": 0.5, "h": 0.5})
    assert image_scale.read_scale(str(tmp_path / "out" / out)) == (14.0, 15.02, "stitch")
