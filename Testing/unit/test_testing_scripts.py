"""Skrip uji: --help jalan tanpa perangkat keras, CSV sesuai skema, rumus posisi benar."""
import csv
import os
import subprocess
import sys

import cv2
import numpy as np
import pytest

TESTING = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, TESTING)

import common  # noqa: E402
import positioning_math as pm  # noqa: E402

SCRIPTS = ["t00_check_clock.py", "t01_positioning.py", "t02_cmd_latency_local.py", "t03_cmd_latency_client.py",
           "t04_video_client.py", "t05_eta_scan.py", "t05b_settle_sharpness.py", "t06_ratelimit_burst.py"]


@pytest.mark.parametrize("script", SCRIPTS)
def test_help_jalan_tanpa_perangkat_keras(script):
    out = subprocess.run([sys.executable, os.path.join(TESTING, script), "--help"],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert "--log-dir" in out.stdout


def test_append_row_menambah_tanpa_menimpa(tmp_path):
    common.append_row(str(tmp_path), "eta", grid="3x3", run=1)
    common.append_row(str(tmp_path), "eta", grid="3x3", run=2)
    with open(tmp_path / "eta.csv") as f:
        rows = list(csv.DictReader(f))
    assert [r["run"] for r in rows] == ["1", "2"]


def test_salinan_rumus_t05_sama_dengan_backend():
    import t05_eta_scan as t05
    backend = os.path.abspath(os.path.join(TESTING, "..", "..", "Software", "backend", "app", "services", "scan_planner.py"))
    if not os.path.exists(backend):
        pytest.skip("Repo Software tidak ada di sebelah repo ini.")
    namespace = {}
    with open(backend) as f:
        exec(compile(f.read(), backend, "exec"), namespace)
    expected = namespace["estimate_scan_seconds"](3, 4, 0.5, 0.5, 250.0, 500)
    assert t05.predict_seconds(3, 4, 0.5, 250.0, 500) == pytest.approx(expected)


UM_PER_PX = 1.7
MINOR_UM = 10.0
MINOR_PX = MINOR_UM / UM_PER_PX


def _slide(shift_px, width=800, height=400, seed=1):
    """Citra sintetis micrometer: garis periodik + penanda unik, bergeser shift_px (sub-piksel)."""
    xx = np.arange(width, dtype=np.float64)
    profile = 200 - 90 * np.exp(-((((xx - shift_px) % MINOR_PX) - 1.5) ** 2) / 0.8)
    img = np.tile(profile, (height, 1)).astype(np.float32)
    # penanda unik di luar pita garis
    marker = np.zeros((40, 60), np.float32)
    rng = np.random.default_rng(seed)
    marker[rng.random(marker.shape) > 0.7] = 1.0
    marker = cv2.GaussianBlur(marker, (0, 0), 1.0)
    M = np.float32([[1, 0, 200 + shift_px], [0, 1, 40]])
    layer = cv2.warpAffine(marker, M, (width, height), flags=cv2.INTER_CUBIC)
    img[:120] = 180
    img = img - layer * 150
    noise = np.random.default_rng(2).normal(0, 1.0, img.shape)
    return np.clip(np.dstack([img, img, img]) + noise[..., None], 0, 255).astype(np.uint8)


BAND = (20, 180, 760, 200)


def test_periode_garis_terdeteksi():
    gray = pm.preprocess(_slide(0.0))
    period = pm.estimate_period_px(pm.line_profile(gray, BAND, "X"))
    assert period == pytest.approx(MINOR_PX, rel=0.01)


@pytest.mark.parametrize("shift", [0.37, 1.5, 7.3, 30.0, 88.25, 150.6])
def test_gabungan_kasar_halus_memulihkan_pergeseran(shift):
    gray0 = pm.preprocess(_slide(0.0))
    template = gray0[40:80, 200:260].copy()
    x0, _, _ = pm.locate_marker(gray0, template)
    fine0 = pm.fine_phase_px(pm.line_profile(gray0, BAND, "X"), MINOR_PX)
    pos0, _ = pm.combine(x0, fine0, MINOR_PX)

    gray = pm.preprocess(_slide(shift))
    x, _, score = pm.locate_marker(gray, template)
    fine = pm.fine_phase_px(pm.line_profile(gray, BAND, "X"), MINOR_PX)
    pos, valid = pm.combine(x, fine, MINOR_PX)
    assert valid and score > 0.8
    assert pos - pos0 == pytest.approx(shift, abs=0.15)


def test_combine_menandai_kasar_yang_berselisih():
    position, valid = pm.combine(10.0, 2.0, 8.0)
    assert position == 10.0 and valid
    _, valid = pm.combine(13.5, 2.0, 8.0)
    assert not valid


def test_arah_tercermin_terukur_dari_gerak():
    assert pm.motion_direction(-30.0, 0.0) == (-1.0, 0.0)
    with pytest.raises(ValueError):
        pm.motion_direction(0.2, 0.1)


def test_target_keluar_bidang_pandang_ditolak():
    bad = pm.target_stays_in_frame((400, 100), (60, 40), (800, 400), (1.0, 0.0), 1.7, [0, 0.3, 0.8])
    assert bad == [0.8]
    assert pm.target_stays_in_frame((400, 100), (60, 40), (800, 400), (1.0, 0.0), 1.7, [0, 0.3]) == []
