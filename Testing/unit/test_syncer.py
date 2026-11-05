"""vps_syncer: nama dari server disaring, index kosong tidak menghapus dataset lokal."""
import os
import time

from Hardware.Backend_Connector import vps_syncer


class Resp:
    def __init__(self, payload):
        self.status_code = 200
        self.payload = payload

    def json(self):
        return self.payload


def _run(tmp_path, monkeypatch, index):
    monkeypatch.setattr(vps_syncer, "DATASET_DIR", str(tmp_path))
    monkeypatch.setattr(vps_syncer.requests, "get", lambda *a, **k: Resp({"folders": index}))
    vps_syncer.sync_datasets()


def _old_folder(tmp_path, name):
    folder = tmp_path / name
    folder.mkdir()
    (folder / "a.jpg.synced").write_text("x")
    old = time.time() - 30 * 24 * 3600
    os.utime(folder, (old, old))
    return folder


def test_index_kosong_tidak_menghapus_dataset(tmp_path, monkeypatch):
    folder = _old_folder(tmp_path, "abc12345")
    _run(tmp_path, monkeypatch, {})
    assert folder.exists()


def test_folder_yatim_tetap_dibersihkan_kalau_index_berisi(tmp_path, monkeypatch):
    folder = _old_folder(tmp_path, "abc12345")
    _run(tmp_path, monkeypatch, {"lain": {"files": []}})
    assert not folder.exists()


def test_nama_folder_traversal_dari_server_tidak_keluar(tmp_path, monkeypatch):
    root = tmp_path / "datasets"
    root.mkdir()
    _run(root, monkeypatch, {"../../jahat": {"files": []}, "..": {"files": []}})
    assert not (tmp_path / "jahat").exists()
    assert (root / "jahat").exists()


def test_sync_disabled(monkeypatch, capsys):
    monkeypatch.setenv("SYNC_DISABLED", "1")
    vps_syncer.run_daemon()
    assert "dimatikan" in capsys.readouterr().out
