import asyncio
import websockets
import sounddevice as sd
import queue
import sys

# Thread-safe queue for audio chunks from the sounddevice callback
audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"[Audio Status] {status}", file=sys.stderr)
    audio_queue.put(bytes(indata))

async def main():
    uri = "ws://localhost:8766"
    print(f"Connecting to Vosk server at {uri}...")
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected! Start speaking into your laptop microphone (Press Ctrl+C to stop)...")
            
            # Open a 16kHz, 16-bit mono microphone stream
            with sd.RawInputStream(samplerate=16000, blocksize=4000, dtype='int16',
                                   channels=1, callback=audio_callback):
                while True:
                    # Fetch audio chunk asynchronously without blocking the event loop
                    data = await asyncio.to_thread(audio_queue.get)
                    await websocket.send(data)
    except ConnectionRefusedError:
        print(f"\n[Error] Could not connect to {uri}. Make sure 'python server.py' is running first!")
    except websockets.exceptions.ConnectionClosed:
        print("\n[Notice] Server closed the connection.")
    except KeyboardInterrupt:
        print("\nStopping client...")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
