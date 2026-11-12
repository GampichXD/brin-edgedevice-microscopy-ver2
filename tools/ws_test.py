import asyncio
import websockets

async def test():
    try:
        async with websockets.connect('ws://127.0.0.1:8000/api/hardware/ws') as ws:
            print('Connected!')
    except Exception as e:
        print('Error:', type(e), e)

asyncio.run(test())
