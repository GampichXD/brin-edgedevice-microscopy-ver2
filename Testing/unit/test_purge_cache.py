"""Purge di Jetson: hanya menghapus yang semestinya."""
import asyncio

import pytest

from Hardware.Handlers import camera_handler
from Hardware.Handlers.camera_handler import handle_camera_action


@pytest.fixture
def edge(tmp_path, monkeypatch):
    handlers = tmp_path / "Handlers"
    handlers.mkdir()
    monkeypatch.setattr(camera_handler, "__file__", str(handlers / "camera_handler.py"))
    tmp_images = tmp_path / "tmp_images"
    (tmp_images / "20261105_abc").mkdir(parents=True)
    (tmp_images / "20261105_abc" / "tile_r0_c0.jpg").write_text("x")
    (tmp_images / "20261105_def").mkdir()
    (tmp_images / "20261105_def" / "tile_r0_c0.jpg").write_text("x")
    (tmp_images / "manual.jpg").write_text("x")
    monkeypatch.setattr(camera_handler, "LOCAL_TMP_DIR", str(tmp_images))
    dataset = tmp_path / "local_datasets" / "abc12345"
    dataset.mkdir(parents=True)
    (dataset / "a.jpg").write_text("x")
    return tmp_path


def _run(action, **data):
    asyncio.run(handle_camera_action(action, data, None, None, {}))


def test_purge_cache_membersihkan_tmp_tetapi_tidak_menyentuh_dataset(edge):
    _run("PURGE_CACHE")
    assert list((edge / "tmp_images").iterdir()) == []
    assert (edge / "local_datasets" / "abc12345" / "a.jpg").exists()


def test_purge_all_membersihkan_tmp_dan_dataset(edge):
    _run("PURGE_ALL_DATASETS")
    assert list((edge / "tmp_images").iterdir()) == []
    assert list((edge / "local_datasets").iterdir()) == []


def test_purge_folder_hanya_folder_itu(edge):
    other = edge / "local_datasets" / "def67890"
    other.mkdir()
    _run("PURGE_DATASET_FOLDER", folder_id="abc12345")
    assert not (edge / "local_datasets" / "abc12345").exists()
    assert other.exists()


def test_purge_session_hanya_sesi_itu(edge):
    tmp = edge / "tmp_images"
    (tmp / "_stage_20261105_abc").mkdir()
    (tmp / "_stage_scaled_20261105_abc").mkdir()
    _run("PURGE_SESSION", session="20261105_abc")
    assert sorted(p.name for p in tmp.iterdir()) == ["20261105_def", "manual.jpg"]
    assert (edge / "local_datasets" / "abc12345" / "a.jpg").exists()


@pytest.mark.parametrize("session", ["", None])
def test_purge_session_tanpa_nama_tidak_menghapus(edge, session):
    _run("PURGE_SESSION", session=session)
    assert (edge / "tmp_images" / "20261105_abc").exists()


def test_purge_session_traversal_tidak_keluar(edge):
    _run("PURGE_SESSION", session="../local_datasets")
    assert (edge / "local_datasets").exists()
