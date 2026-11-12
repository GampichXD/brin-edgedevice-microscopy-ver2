"""B2: watchdog koneksi di vps_bridge, tanpa jaringan."""
import asyncio
import csv
import os
import time

import pytest

from Hardware.Backend_Connector import vps_bridge
from Hardware.Utils import test_logger


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(test_logger, "LOG_DIR", str(tmp_path))
    monkeypatch.setattr(vps_bridge, "WATCHDOG_TIMEOUT", 0.3)
    monkeypatch.setattr(vps_bridge, "WATCHDOG_TICK_S", 0.02)
    holds = []
    monkeypatch.setattr(vps_bridge.motor_core, "feed_hold", lambda: holds.append(time.monotonic()) or True)
    return holds, tmp_path


def _events(tmp_path):
    with open(os.path.join(tmp_path, "watchdog_log.csv")) as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == ["t_ns", "event", "last_rx_age_ms", "motor_state"]
    return [r["event"] for r in rows]


async def _run_watchdog(link, seconds):
    task = asyncio.create_task(vps_bridge.watchdog_task(link))
    await asyncio.sleep(seconds)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


def test_memicu_hold_kalau_ping_berhenti(setup):
    holds, tmp = setup
    link = vps_bridge.LinkState()
    link.armed = True
    asyncio.run(_run_watchdog(link, 0.5))
    assert len(holds) == 1
    assert _events(tmp) == ["WATCHDOG_TRIGGERED", "FEED_HOLD_SENT"]


def test_belum_armed_tidak_memicu_hold(setup):
    holds, _ = setup
    link = vps_bridge.LinkState()
    asyncio.run(_run_watchdog(link, 0.5))
    assert holds == []


def test_ping_rutin_tidak_memicu_hold(setup):
    holds, _ = setup

    async def go():
        link = vps_bridge.LinkState()
        link.armed = True
        task = asyncio.create_task(vps_bridge.watchdog_task(link))
        for _ in range(10):
            await asyncio.sleep(0.05)
            link.last_rx = time.monotonic()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(go())
    assert holds == []


class FakeSocket:
    def __init__(self, messages):
        self.messages = list(messages)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.messages:
            raise StopAsyncIteration
        return self.messages.pop(0)


def test_reader_memisahkan_ping_dari_perintah(setup):
    _, tmp = setup

    async def go():
        link = vps_bridge.LinkState()
        link.hold_sent = True
        commands = asyncio.Queue()
        await vps_bridge.receive_handler(
            FakeSocket([vps_bridge.PING_MESSAGE, '{"action": "HOMING"}']), link, commands)
        return link, commands

    link, commands = asyncio.run(go())
    assert link.armed is True and link.hold_sent is False
    assert commands.qsize() == 1
    assert _events(tmp) == ["LINK_RESTORED"]


def test_ping_sama_persis_dengan_kiriman_server():
    # hardware.py mengirim json.dumps({"action": "PING"}).
    import json
    assert vps_bridge.PING_MESSAGE == json.dumps({"action": "PING"})


def test_logger_mati_secara_default(tmp_path, monkeypatch):
    monkeypatch.setattr(test_logger, "LOG_DIR", str(tmp_path))
    monkeypatch.delenv("TEST_LOGGING", raising=False)
    test_logger.log("cmd", cmd_id=1)
    assert os.listdir(tmp_path) == []
    monkeypatch.setenv("TEST_LOGGING", "1")
    test_logger.log("cmd", cmd_id=1)
    test_logger.log("cmd", cmd_id=2)
    with open(os.path.join(tmp_path, "cmd.csv")) as f:
        assert f.read().split() == ["cmd_id", "1", "2"]
