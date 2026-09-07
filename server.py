import asyncio
import websockets
import json
import time
from vosk import Model, KaldiRecognizer

print("Loading Vosk Model... (This might take a few seconds)")
# Loads the model from the "model" folder
model = Model("model/vosk-model-small-en-us-0.15") 
print("Model loaded successfully!")

async def handler(websocket):
    print("\nDevice connected! Listening for WAKE trigger...")
    rec = KaldiRecognizer(model, 16000)
    
    t_wake = 0
    t_first_byte = 0
    
    async for message in websocket:
        # Check if the message is a text command (like WAKE)
        if isinstance(message, str):
            if message.startswith("WAKE:"):
                # Using the server's time for accurate latency calculation
                t_wake = time.time() * 1000
                esp32_millis = message.split(":")[1]
                print(f"\n[LATENCY] WAKE triggered at {t_wake:.0f} ms (ESP32 time: {esp32_millis})")
                t_first_byte = 0 # reset for the new stream
                
        # Check if the message is raw binary audio data
        elif isinstance(message, bytes):
            # Record the time the first byte arrives after a wake
            if t_first_byte == 0 and t_wake != 0:
                t_first_byte = time.time() * 1000
                wake_to_network_latency = t_first_byte - t_wake
                print(f"[LATENCY] Wake-to-Network Latency: {wake_to_network_latency:.2f} ms")
            
            # Feed the audio bytes into the Vosk recognizer
            if rec.AcceptWaveform(message):
                result = json.loads(rec.Result())
                if result.get("text"):
                    print(f"Final Sentencew0: {result['text']}")
                    
                    if t_wake != 0:
                        t_end = time.time() * 1000
                        total_latency = t_end - t_wake
                        print(f"[LATENCY] Total Pipeline Latency: {total_latency:.2f} ms")
                        
                    # Reset wake tracker until next wake
                    t_wake = 0 
            else:
                partial = json.loads(rec.PartialResult())
                if partial.get("partial"):
                    # Print partials on the same line to reduce terminal clutter
                    print(f"Hearing: {partial['partial']}          ", end="\r")

async def main():
    async with websockets.serve(handler, "0.0.0.0", 8766):
        print("Vosk Server running on ws://0.0.0.0:8766")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
