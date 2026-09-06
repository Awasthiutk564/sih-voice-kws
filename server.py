import asyncio
import websockets

async def handler(websocket):
    print("ESP32 connected!")
    async for message in websocket:
        if isinstance(message, bytes):
            print(f"Received audio data: {len(message)} bytes")
            with open("received_audio.raw", "wb") as f:
                f.write(message)
            print("Saved as received_audio.raw")
        else:
            print(f"Received text: {message}")

async def main():
    async with websockets.serve(handler, "0.0.0.0", 8765):
        print("Server running on ws://0.0.0.0:8765")
        await asyncio.Future()

asyncio.run(main())