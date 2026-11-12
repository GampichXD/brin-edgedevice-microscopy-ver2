"""Tes unit Jetson. Modul di-import sebagai paket `Hardware`, jadi folder repo harus bernama Hardware."""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ai_handler memuat YOLO dan TensorRT saat di-import. Tes unit tidak butuh keduanya.
import types  # noqa: E402

_ai_stub = types.ModuleType("Hardware.Handlers.ai_handler")


async def _handle_ai_action(*args, **kwargs):
    raise AssertionError("ai_handler tidak dipakai di tes unit")

_ai_stub.handle_ai_action = _handle_ai_action
sys.modules["Hardware.Handlers.ai_handler"] = _ai_stub
