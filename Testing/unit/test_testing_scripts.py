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

SCRIPTS = ["t00_check_clock.py", "t01_positioning.py", "t02_cmd_latency_local.py", "t03_cmd_latency_client.py",
           "t04_video_client.py", "t05_eta_scan.py", "t05b_settle_sharpness.py", "t06_ratelimit_burst.py"]


@pytest.mark.parametrize("script", SCRIPTS)
def test_help_jalan_tanpa_perangkat_keras(script):
    out = subprocess.run([sys.executable, os.path.join(TESTING, script), "--help"],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert "usage" in out.stdout


def test_append_row_menambah_tanpa_menimpa(tmp_path):
    common.append_row(str(tmp_path), "eta", grid="3x3", run=1)
    common.append_row(str(tmp_path), "eta", grid="3x3", run=2)
    with open(tmp_path / "eta.csv") as f:
        rows = list(csv.DictReader(f))
    assert [r["run"] for r in rows] == ["1", "2"]


def test_salinan_rumus_t05_sama_dengan_backend():
    import t05_eta_scan as t05
    root = os.environ.get("SOFTWARE_REPO") or os.path.join(TESTING, "..", "..", "Software")
    backend = os.path.abspath(os.path.join(root, "backend", "app", "services", "scan_planner.py"))
    if not os.path.exists(backend):
        pytest.skip("Repo Software tidak ada di sebelah repo ini.")
    namespace = {}
    with open(backend) as f:
        exec(compile(f.read(), backend, "exec"), namespace)
    expected = namespace["estimate_scan_seconds"](3, 4, 0.5, 0.5, 250.0, 500)
    assert t05.predict_seconds(3, 4, 0.5, 250.0, 500) == pytest.approx(expected)


def _slide(path):
    """Citra sintetis micrometer: pita garis periodik, angka unik, dan vignetting."""
    h, w = 480, 800
    img = np.full((h, w, 3), 190, np.uint8)
    for i, x in enumerate(np.arange(0, w, 17.0)):
        y1 = 300 if i % 10 == 0 else 330
        cv2.line(img, (int(x), y1), (int(x), 380), (60, 60, 60), 2)
    for i, x in enumerate(range(0, w, 170)):
        cv2.putText(img, str(i), (x + 4, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2)
    cv2.imwrite(str(path), img)


def test_t01_selftest_memulihkan_pergeseran_sintetis(tmp_path):
    photo = tmp_path / "slide.png"
    _slide(photo)
    out = subprocess.run([sys.executable, os.path.join(TESTING, "t01_positioning.py"), "selftest",
                          "--image", str(photo), "--roi", "344,260,40,30"],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stdout + out.stderr


def _csv(folder, name, header, rows):
    with open(folder / name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def test_analyze_hw_cocok_dengan_csv_skrip_ini(tmp_path):
    data = tmp_path / "logs"
    data.mkdir()
    _csv(data, "positioning.csv", ["axis", "target_mm", "run", "direction", "measured_mm", "valid", "score"],
         [["X", t, r, d, t + 0.002 * r, 1, 0.9] for t in (0, 0.13) for r in (1, 2) for d in ("fwd", "bwd")])
    _csv(data, "cmd_local.csv", ["cmd_id", "t_sent_ns", "t_ok_ns"], [[i, 100 * i, 100 * i + 5_000_000] for i in range(5)])
    _csv(data, "cmd_client.csv", ["cmd_id", "path", "t_sent_ns"], [[i, "lan", 100 * i] for i in range(5)])
    _csv(data, "cmd.csv", ["cmd_id", "gcode", "t_rx_ns", "t_ok_ns"],
         [[i, "G1 X0.05 F250", 100 * i + 10**7, 100 * i + 2 * 10**7] for i in range(4)] + [["", "G90", 1, 2]])
    _csv(data, "video.csv", ["frame_id", "t_cap_ns", "t_recv_ns", "bytes"],
         [[i, i * 33 * 10**6, i * 33 * 10**6 + 80 * 10**6, 50000] for i in range(100)])
    _csv(data, "eta.csv", ["grid", "run", "n_tiles", "predicted_s", "actual_s"], [["3x3", 1, 9, 9.0, 12.0], ["3x3", 2, 9, 9.0, 12.5]])
    _csv(data, "ratelimit.csv", ["t_send_ns", "channel", "result"],
         [[i * 50 * 10**6, ch, "ACCEPTED" if i % 5 == 0 else "REJECTED"] for ch in ("rest", "ws") for i in range(40)])
    _csv(data, "telemetry.csv", ["t_ns", "status", "x", "y", "z", "cpu", "cpu_temp", "gpu_temp", "ram_used_gb", "ram_total_gb"],
         [[i * 10**9, "Hold:0" if i > 5 else "Run", 1, 2, 3, 20, 50, 45, 3.0, 8.0] for i in range(10)])
    _csv(data, "watchdog_log.csv", ["t_ns", "event", "last_rx_age_ms", "motor_state"],
         [[3 * 10**9, "WATCHDOG_TRIGGERED", 3100, "Run"], [3 * 10**9, "FEED_HOLD_SENT", 3100, "Run"]])
    _csv(data, "settle.csv", ["move_type", "delay_s", "rep", "sharpness"],
         [[m, d, r, 100 + d * 10] for m in ("in_row", "row_change") for d in (0, 0.5) for r in (0, 1)])
    out_dir = tmp_path / "out"
    out = subprocess.run([sys.executable, os.path.join(TESTING, "analyze_hw.py"), "--data", str(data), "--out", str(out_dir)],
                         capture_output=True, text=True, timeout=180)
    assert out.returncode == 0, out.stdout + out.stderr
    summary = (out_dir / "summary.md").read_text()
    for section in ("D1.", "D2.", "D3.", "D4.", "D5.", "D6.", "D7.", "D8."):
        assert section in summary
    assert "| X |" in summary
    figures = os.listdir(out_dir / "figures")
    assert "D1_positioning_X.png" in figures and "D7_watchdog.png" in figures
