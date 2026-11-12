"""B9: flat-field correction."""
import numpy as np
import pytest

from Hardware.Utils import flatfield


def _vignette(h=40, w=60):
    # Tengah gelap, pojok terang, kanal merah lemah (seperti slide kalibrasi).
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot((xx - w / 2) / w, (yy - h / 2) / h)
    pattern = 0.5 + r
    return np.dstack([pattern * 150, pattern * 130, pattern * 30]).astype(np.float32)


def test_koreksi_meratakan_pola():
    flat = _vignette()
    dark = np.full_like(flat, 2.0)
    out = flatfield.correct(flat + 0, flat, dark).astype(np.float64)
    for ch in range(3):
        assert out[..., ch].std() < 1.0
    assert flat[..., 0].std() > 5


def test_piksel_tanpa_informasi_tidak_meledak():
    flat = _vignette()
    dark = flat.copy()
    dark[:5, :5] = flat[:5, :5]
    dark[5:] = 0
    out = flatfield.correct(flat, flat, dark)
    assert out.max() <= 255 and out.dtype == np.uint8


def test_capture_reference_rata_rata(tmp_path, monkeypatch):
    monkeypatch.setattr(flatfield, "CALIB_DIR", str(tmp_path))
    monkeypatch.setattr(flatfield, "FLAT_PATH", str(tmp_path / "flat.npy"))
    monkeypatch.setattr(flatfield, "DARK_PATH", str(tmp_path / "dark.npy"))
    assert flatfield.load_reference() is None
    frames = iter([np.full((2, 2, 3), 10, np.uint8), None, np.full((2, 2, 3), 20, np.uint8)])
    flatfield.capture_reference(lambda: next(frames), "flat", frames=3)
    flatfield.capture_reference(lambda: np.zeros((2, 2, 3), np.uint8), "dark", frames=2)
    flat, dark = flatfield.load_reference()
    assert flat[0, 0, 0] == 15 and dark.max() == 0


def test_kamera_mati_dan_mode_salah(tmp_path, monkeypatch):
    monkeypatch.setattr(flatfield, "CALIB_DIR", str(tmp_path))
    with pytest.raises(RuntimeError):
        flatfield.capture_reference(lambda: None, "flat", frames=2)
    with pytest.raises(ValueError):
        flatfield.capture_reference(lambda: None, "abu")


def test_default_mati(monkeypatch):
    monkeypatch.delenv("FLATFIELD_ENABLED", raising=False)
    assert flatfield.enabled() is False
