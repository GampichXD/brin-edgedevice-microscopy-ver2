import json
import asyncio
import time
from Hardware.Devices.motor import motor_core
from Hardware.Utils import test_logger

def _result_status(grbl_resp) -> str:
    return "ERROR" if str(grbl_resp).startswith("ERROR") else "SUCCESS"


async def handle_motor_action(action: str, data: dict, websocket, ws_lock):
    """Router untuk instruksi mekanika CNC/Motor.
    PENTING: Semua operasi serial motor dijalankan di thread pool agar
    tidak memblokir asyncio event loop (dan memutus stream video/WebSocket).
    """
    if action == "MOVE_MOTOR":
        gcode = data.get("gcode", "")
        print(f"[BRIDGE] Eksekusi G-Code: {gcode}")
        # Jalankan operasi serial blocking di thread pool
        t_rx_ns = data.get("_t_rx_ns") or time.time_ns()
        grbl_resp = await asyncio.to_thread(motor_core.jog_from_gcode, gcode)
        test_logger.log("cmd", cmd_id=data.get("cmd_id", ""), gcode=gcode,
                        t_rx_ns=t_rx_ns, t_ok_ns=time.time_ns())
        
        async with ws_lock:
            await websocket.send(json.dumps({
                "event": "MOTOR_MOVED",
                "status": _result_status(grbl_resp),
                "grbl_response": grbl_resp
            }))

    elif action == "HOMING":
        print("[BRIDGE] Perintah VPS: Homing motor.")
        grbl_resp = await asyncio.to_thread(motor_core.homing)
        async with ws_lock:
            await websocket.send(json.dumps({
                "event": "MOTOR_MOVED",
                "status": _result_status(grbl_resp),
                "grbl_response": grbl_resp
            }))

    elif action == "UNLOCK":
        print("[BRIDGE] Perintah VPS: Unlock motor.")
        grbl_resp = await asyncio.to_thread(motor_core.unlock)
        async with ws_lock:
            await websocket.send(json.dumps({
                "event": "MOTOR_MOVED",
                "status": _result_status(grbl_resp),
                "grbl_response": grbl_resp
            }))

    elif action == "SET_POSITION":
        print(f"[BRIDGE CNC] Set koordinat kerja manual: X={data.get('x')} Y={data.get('y')} Z={data.get('z')}")
        grbl_resp = await asyncio.to_thread(
            motor_core.set_position, data.get("x", 0.0), data.get("y", 0.0), data.get("z", 0.0)
        )
        async with ws_lock:
            await websocket.send(json.dumps({
                "event": "MOTOR_MOVED", "status": _result_status(grbl_resp), "grbl_response": grbl_resp
            }))

    elif action == "APPLY_CNC_SETTINGS":
        print(
            f"[BRIDGE CNC] Settings diterima: feed={data.get('feed_rate')} backlash={data.get('backlash')} accel={data.get('acceleration')} settle={data.get('settle_time')}"
        )
        await asyncio.to_thread(
            motor_core.apply_motion_settings,
            data.get("feed_rate"),
            data.get("backlash"),
            data.get("acceleration"),
            data.get("settle_time"),
        )

    elif action == "APPLY_SOFT_LIMITS":
        print(f"[BRIDGE CNC] Soft limits diterima: enabled={data.get('enabled')} limits={data.get('limits')}")
        await asyncio.to_thread(
            motor_core.set_soft_limits,
            data.get("enabled"),
            data.get("limits"),
        )

