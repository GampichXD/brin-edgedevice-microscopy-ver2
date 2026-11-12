"""B7: PURGE_DATASET_FILES hanya menghapus berkas yang diminta."""
import asyncio
import os

from Hardware.Handlers import camera_handler
from Hardware.Handlers.camera_handler import handle_camera_action


def _purge(tmp_path, monkeypatch, files, filenames):
    # handler menyusun path dari lokasi file-nya: arahkan ke folder sementara
    folder = tmp_path / "Handlers"
    folder.mkdir(exist_ok=True)
    monkeypatch.setattr(camera_handler, "__file__", str(folder / "camera_handler.py"))
    target = tmp_path / "local_datasets" / "abc12345"
    target.mkdir(parents=True, exist_ok=True)
    for name in files:
        (target / name).write_text("x")
    asyncio.run(handle_camera_action(
        "PURGE_DATASET_FILES", {"folder_id": "abc12345", "filenames": filenames}, None, None, {}))
    return sorted(os.listdir(target))


def test_tidak_menghapus_tile_lain(tmp_path, monkeypatch):
    left = _purge(tmp_path, monkeypatch, ["tile.jpg", "tile_r0_c0.jpg", "tile_r0_c1.jpg"], ["tile.jpg"])
    assert left == ["tile_r0_c0.jpg", "tile_r0_c1.jpg"]


def test_nama_kosong_diabaikan(tmp_path, monkeypatch):
    left = _purge(tmp_path, monkeypatch, ["_a.jpg", "_b.jpg"], ["", None])
    assert left == ["_a.jpg", "_b.jpg"]


def test_varian_timestamp_dan_marker_ikut_terhapus(tmp_path, monkeypatch):
    files = ["IMG_1.jpg", "IMG_1.jpg.synced", "IMG_1_20261105_101010.jpg",
             "IMG_1_20261105_101010.jpg.synced", "IMG_1_salinan.jpg", "IMG_10.jpg"]
    assert _purge(tmp_path, monkeypatch, files, ["IMG_1.jpg"]) == ["IMG_10.jpg", "IMG_1_salinan.jpg"]


def test_traversal_hanya_basename(tmp_path, monkeypatch):
    (tmp_path / "korban.jpg").write_text("x")
    assert _purge(tmp_path, monkeypatch, ["a.jpg"], ["../../korban.jpg"]) == ["a.jpg"]
    assert (tmp_path / "korban.jpg").exists()
