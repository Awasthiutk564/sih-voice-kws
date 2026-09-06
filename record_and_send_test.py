import asyncio
import websockets
import sounddevice as sd
import queue
import time
import sys

# Create a queue to hold audio chunks
q = queue.Queue()

# This function grabs audio from your laptop mic
def callback(indata, frames, time_info, status):
    if status:
        print(status)
    q.put(bytes(indata))

async def test():
    print("Connecting to server...")
    try:
        async with websockets.connect("ws://localhost:8766", ping_interval=None) as websocket:
            print("Connected!")
            
            while True:
                await asyncio.to_thread(input, "\nPress ENTER to simulate WAKE and send 5 seconds of audio (or Ctrl+C to quit)...")
                
                # Simulate WAKE trigger
                wake_msg = f"WAKE:{int(time.time()*1000)}"
                await websocket.send(wake_msg)
                print(f"Sent: {wake_msg}")
                
                print("Recording and streaming... Speak now!")
                
                # Clear any old audio in the queue
                while not q.empty():
                    q.get()
                
                # Open a 16kHz microphone stream
                with sd.RawInputStream(samplerate=16000, blocksize=4000, dtype='int16',
                                       channels=1, callback=callback):
                    start_time = time.time()
                    while time.time() - start_time < 5.0: # Record for 5 seconds
                        data = await asyncio.to_thread(q.get)
                        await websocket.send(data)
                        
                print("Finished sending audio. Check server terminal for latency metrics.")
                
    except ConnectionRefusedError:
        print("Could not connect to server. Make sure server.py is running!")

if __name__ == "__main__":
    try:
        asyncio.run(test())
    except KeyboardInterrupt:
        print("\nExiting.")
        sys.exit(0)
