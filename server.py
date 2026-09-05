import asyncio
import websockets

async def handler(websocket):
       print("Someone connected!")
       async for message in websocket:
           print(f"Received: {message}")

async def main():
       async with websockets.serve(handler, "localhost", 8765):
           print("Server running on ws://localhost:8765")
           await asyncio.Future()  # run forever

asyncio.run(main())