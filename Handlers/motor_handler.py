import json
import asyncio
from Hardware.Devices.motor import motor_core

async def handle_motor_action(action: str, data: dict, websocket, ws_lock):
    """Router untuk instruksi mekanika CNC/Motor.
    PENTING: Semua operasi serial motor dijalankan di thread pool agar
    tidak memblokir asyncio event loop (dan memutus stream video/WebSocket).
    """
    if action == "MOVE_MOTOR":
        gcode = data.get("gcode", "")
        print(f"[BRIDGE] Eksekusi G-Code: {gcode}")
        # Jalankan operasi serial blocking di thread pool
        grbl_resp = await asyncio.to_thread(motor_core.jog_from_gcode, gcode)
        
        async with ws_lock:
            await websocket.send(json.dumps({
                "event": "MOTOR_MOVED",
                "status": "SUCCESS",
                "grbl_response": grbl_resp
            }))

    elif action == "HOMING":
        print("[BRIDGE] Perintah VPS: Homing motor.")
        grbl_resp = await asyncio.to_thread(motor_core.homing)
        async with ws_lock:
            await websocket.send(json.dumps({
                "event": "MOTOR_MOVED",
                "status": "SUCCESS",
                "grbl_response": grbl_resp
            }))

    elif action == "UNLOCK":
        print("[BRIDGE] Perintah VPS: Unlock motor.")
        grbl_resp = await asyncio.to_thread(motor_core.unlock)
        async with ws_lock:
            await websocket.send(json.dumps({
                "event": "MOTOR_MOVED",
                "status": "SUCCESS",
                "grbl_response": grbl_resp
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

