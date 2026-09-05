import asyncio
import websockets

async def test():
       async with websockets.connect("ws://localhost:8765") as websocket:
           await websocket.send("Hello from test client!")
           print("Message sent!")

asyncio.run(test())