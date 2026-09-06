import asyncio
import websockets

async def test():
       async with websockets.connect("ws://localhost:8766") as websocket:
           await websocket.send("Hello from test client!")
           print("Message sent!")

import asyncio
import websockets
import sounddevice as sd
import queue

# Create a queue to hold audio chunks
q = queue.Queue()

# This function grabs audio from your laptop mic
def callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(bytes(indata))

async def test():
    print("Connecting to server...")
    async with websockets.connect("ws://localhost:8766") as websocket:
        print("Connected! Start speaking into your laptop microphone...")
        # Open a 16kHz microphone stream
        with sd.RawInputStream(samplerate=16000, blocksize=4000, dtype='int16',
                               channels=1, callback=callback):
            while True:
                # Get the next chunk of audio and send it over WebSocket
                data = q.get()
                await websocket.send(data)

if __name__ == "__main__":
    asyncio.run(test())
