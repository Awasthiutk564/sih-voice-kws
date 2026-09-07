import asyncio
import websockets
import json
import time
import os
import glob
import numpy as np
import torch

# Try importing the Moonshine Streaming model from model.py
try:
    from model import MoonshineStreamingASR, StreamingAudioSession
    MOONSHINE_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] model.py import error: {e}")
    MOONSHINE_AVAILABLE = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Look for any .pt or .pth trained weights file in workspace
weight_files = glob.glob("*.pt") + glob.glob("*.pth") + glob.glob("model/*.pt")
trained_weights_path = weight_files[0] if weight_files else "trained_model.pt"

use_moonshine = False
moonshine_model = None

if MOONSHINE_AVAILABLE and os.path.exists(trained_weights_path):
    print(f"\n=======================================================")
    print(f"Loading Trained PyTorch Streaming Model: {trained_weights_path}")
    print(f"Inference Device: {DEVICE}")
    print(f"=======================================================")
    try:
        moonshine_model = MoonshineStreamingASR.create_tiny()
        checkpoint = torch.load(trained_weights_path, map_location=DEVICE)
        
        # Support various checkpoint save formats
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            moonshine_model.load_state_dict(checkpoint["state_dict"])
        elif isinstance(checkpoint, dict) and any(k.startswith("conv_stem") or k.startswith("layers") for k in checkpoint.keys()):
            moonshine_model.load_state_dict(checkpoint)
        else:
            moonshine_model.load_state_dict(checkpoint)
            
        moonshine_model.to(DEVICE)
        moonshine_model.eval()
        use_moonshine = True
        print("Trained Model loaded successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to load {trained_weights_path}: {e}")
        print("Falling back to Vosk model...")
else:
    print("\n-------------------------------------------------------")
    if not weight_files:
        print(f"[INFO] No .pt/.pth weights file found yet (e.g., '{trained_weights_path}').")
        print("Copy your trained .pt file to this folder to activate it automatically!")
    print("Using Vosk Model as default/fallback...")
    print("-------------------------------------------------------")

# Vosk Fallback initialization
vosk_model = None
if not use_moonshine:
    from vosk import Model, KaldiRecognizer
    vosk_model_path = "model/vosk-model-small-en-us-0.15"
    if os.path.exists(vosk_model_path):
        print("Loading Vosk Model from folder...")
        vosk_model = Model(vosk_model_path)
        print("Vosk model loaded successfully!")
    else:
        print(f"[WARNING] Vosk model folder '{vosk_model_path}' not found.")

async def handler(websocket):
    print("\n[DEVICE CONNECTED] Listening for audio stream...")
    
    # Initialize recognizer session for this client
    if use_moonshine and moonshine_model is not None:
        session = StreamingAudioSession(moonshine_model, chunk_samples=6144, device=DEVICE)
        active_engine = "Moonshine-Trained"
    else:
        from vosk import KaldiRecognizer
        rec = KaldiRecognizer(vosk_model, 16000) if vosk_model else None
        active_engine = "Vosk-Fallback"
        
    print(f"[ENGINE ACTIVE] Using: {active_engine}")
    
    t_wake = 0
    t_first_byte = 0
    last_spoken_time = 0
    accumulated_transcript = ""
    
    async for message in websocket:
        # 1. Text commands (e.g. WAKE trigger from ESP32 or test client)
        if isinstance(message, str):
            if message.startswith("WAKE:"):
                t_wake = time.time() * 1000
                esp32_millis = message.split(":")[1] if ":" in message else "0"
                print(f"\n[LATENCY] WAKE triggered at {t_wake:.0f} ms (ESP32 timestamp: {esp32_millis})")
                t_first_byte = 0
                accumulated_transcript = ""
                if use_moonshine:
                    session.reset()
                    
        # 2. Raw Binary Audio Data (16kHz, 16-bit PCM mono)
        elif isinstance(message, bytes):
            # Telemetry: Calculate Wake-to-Network latency on the first packet
            if t_first_byte == 0 and t_wake != 0:
                t_first_byte = time.time() * 1000
                wake_to_network = t_first_byte - t_wake
                print(f"[LATENCY] Wake-to-Network Latency: {wake_to_network:.2f} ms")
            
            if use_moonshine:
                # Convert 16-bit PCM bytes to normalized float32 (-1.0 to +1.0)
                audio_samples = np.frombuffer(message, dtype=np.int16).astype(np.float32) / 32768.0
                emissions = session.feed_samples(audio_samples)
                
                if emissions:
                    last_spoken_time = time.time()
                    transcript = session.current_transcript
                    if transcript.strip():
                        print(f"[HEARING] {transcript}          ", end="\r")
                        accumulated_transcript = transcript
                        
                        if t_wake != 0:
                            total_lat = (time.time() * 1000) - t_wake
                            print(f"\n[LATENCY] Total Pipeline Latency: {total_lat:.2f} ms")
                            t_wake = 0
            else:
                # Vosk processing
                if rec and rec.AcceptWaveform(message):
                    result = json.loads(rec.Result())
                    if result.get("text"):
                        print(f"\n[FINAL TRANSCRIPT] {result['text']}")
                        if t_wake != 0:
                            total_latency = (time.time() * 1000) - t_wake
                            print(f"[LATENCY] Total Pipeline Latency: {total_latency:.2f} ms")
                            t_wake = 0
                elif rec:
                    partial = json.loads(rec.PartialResult())
                    if partial.get("partial"):
                        print(f"[HEARING] {partial['partial']}          ", end="\r")

async def main():
    port = 8766
    async with websockets.serve(handler, "0.0.0.0", port):
        print(f"\n==================================================")
        print(f" ASR Server running on ws://0.0.0.0:{port}")
        print(f" Ready for ESP32 & Test Clients")
        print(f"==================================================\n")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())

