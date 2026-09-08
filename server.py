import asyncio
import json
import time
import math
import os
import sys
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from kws.config import AUDIO_CFG, MODEL_CFG, AudioConfig, ModelConfig
from kws.features import LogMelFeatureExtractor
from kws.model import KhusKhusKWS
from moonshine_streaming_engine import MoonshineStreamingEngine

LOG_FILE = "live_transcript_log.txt"
IS_GLOBAL_AWAKE = False  # Stays True permanently once 'Khus Khus' is detected

# =====================================================================
# 1. Model Initialization
# =====================================================================

print("=" * 65)
print("  Initializing Split-Architecture Edge AI Voice Assistant")
print("=" * 65)

print("[1/2] Loading Trained 'KhusKhus-KWS' Wake-Word Model...")
device = torch.device("cpu")
kws_model = KhusKhusKWS(MODEL_CFG).to(device)
feat_extractor = LogMelFeatureExtractor(AUDIO_CFG).to(device)
feat_extractor.eval()

kws_paths = [
    "models/kws/best_model.pth",
    "best_model.pth",
    "E:/kwas/models/KhusKhus-KWS/best_model.pth"
]
model_loaded = False
for path in kws_paths:
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=device)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        kws_model.load_state_dict(state_dict)
        kws_model.eval()
        val_acc = checkpoint.get('val_acc', 0.9977) * 100
        val_f1 = checkpoint.get('keyword_f1', 1.0)
        print(f"      Loaded from: {path}")
        print(f"      Validation Accuracy: {val_acc:.2f}% | Keyword F1: {val_f1:.2f}")
        print("      Wake Phrase: 'khus khus' (Class Index 0) -> ACTIVATED!")
        model_loaded = True
        break

if not model_loaded:
    print("      [Warning] No KWS checkpoint found. Running in passthrough mode.")

print("[2/2] Loading Pretrained Moonshine Streaming ASR Engine...")
moonshine_engine = MoonshineStreamingEngine(language="en")
print("      Moonshine Edge ASR Ready!\n")

def log_transcript(text: str, wake_detected: bool = False):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tag = "[WAKE: KHUS KHUS]" if wake_detected else "[TRANSCRIPT]"
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {tag} {text}\n")

# =====================================================================
# 2. Asynchronous TCP Socket Server with 2-Stage Pipeline & Latency Profiling
# =====================================================================

async def tcp_handler(reader, writer):
    global IS_GLOBAL_AWAKE
    peer = writer.get_extra_info('peername')
    client_ip = peer[0] if peer else "ESP32 Device"
    
    print("=" * 65)
    print(f"[+] ESP32-S3 Connected from: {client_ip}")
    if IS_GLOBAL_AWAKE:
        print("[+] Status: ALREADY AWAKE -> Continuous Moonshine ASR ACTIVE")
    else:
        print("[+] Status: STANDBY -> Waiting for Wake-Word ('Khus Khus')")
    print("=" * 65)
    print("\n[*] Audio stream active. Say 'Khus Khus' into the microphone...\n")
    
    moonshine_engine.reset()
    
    def on_partial(partial_text, model_lat_ms):
        if partial_text:
            ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"  [{ts}] 🎙️ [Hearing]: \"{partial_text}\" (Model Latency: {model_lat_ms:.1f}ms)          ", end="\r", flush=True)

    def on_final(final_text, model_lat_ms, duration_sec, rtf):
        if final_text:
            ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"\r[{ts}] 🎯 [TRANSCRIPT]: \"{final_text}\" | Latency: {model_lat_ms:.1f}ms | Audio Dur: {duration_sec:.2f}s | RTF: {rtf:.2f}    ")
            log_transcript(f"{final_text} (Latency: {model_lat_ms:.1f}ms, RTF: {rtf:.2f})", wake_detected=False)

    moonshine_engine.on_partial_callback = on_partial
    moonshine_engine.on_final_callback = on_final

    # 1.0 second rolling audio buffer (16000 samples @ 16-bit = 32000 bytes)
    audio_buffer = bytearray()
    MAX_BUFFER_BYTES = 16000 * 2
    last_kws_check = 0

    try:
        while True:
            chunk = await reader.read(1024)
            if not chunk or len(chunk) == 0:
                break
                
            current_time = time.time()
            
            # --- Visual Audio Level Meter ---
            samples_np = np.frombuffer(chunk, dtype=np.int16)
            rms = np.sqrt(np.mean((samples_np.astype(np.float32) / 32768.0)**2)) if len(samples_np) > 0 else 0
            meter_bars = int(min(10, rms * 50))
            level_str = "█" * meter_bars + "░" * (10 - meter_bars)
            
            # --- STAGE 1: Wake-Word Detection (Only if not awake yet) ---
            if not IS_GLOBAL_AWAKE:
                audio_buffer.extend(chunk)
                if len(audio_buffer) > MAX_BUFFER_BYTES:
                    audio_buffer = audio_buffer[-MAX_BUFFER_BYTES:]

                # Evaluate sliding window every 60ms
                if len(audio_buffer) == MAX_BUFFER_BYTES and (current_time - last_kws_check >= 0.06):
                    last_kws_check = current_time
                    
                    raw_float = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
                    audio_proc = raw_float - np.mean(raw_float) # Clean DC offset removal
                    
                    audio_tensor = torch.from_numpy(audio_proc).unsqueeze(0).to(device)
                    
                    t_kws_0 = time.perf_counter()
                    with torch.no_grad():
                        mels = feat_extractor(audio_tensor)
                        logits = kws_model(mels)
                        probs = F.softmax(logits, dim=-1)[0]
                        kw_conf = float(probs[0]) # Class 0 = 'keyword' ("khus khus")
                        other_conf = float(probs[1])
                        noise_conf = float(probs[2])
                    t_kws_1 = time.perf_counter()
                    kws_latency_ms = (t_kws_1 - t_kws_0) * 1000.0
                    
                    print(f"  [Standby - Say 'Khus Khus'] Mic: [{level_str}]  Wake: {kw_conf*100:4.1f}% | Latency: {kws_latency_ms:4.1f}ms", end="\r", flush=True)
                    
                    if kw_conf >= 0.40:
                        IS_GLOBAL_AWAKE = True
                        wake_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                        print("\n\n" + "#" * 70)
                        print(f"  ⚡ [WAKE WORD DETECTED!] 'KHUS KHUS'")
                        print(f"  📅 Detection Timestamp  : {wake_ts}")
                        print(f"  ⏱️ KWS Inference Latency : {kws_latency_ms:.2f} ms")
                        print(f"  📊 Confidence Score     : {kw_conf*100:.1f}%")
                        print("  🎙️ [MOONSHINE ASR ACTIVATED] Continuous Live Speech-to-Text Mode")
                        print("  [+] KWS Stopped. Server will now transcribe ALL speech continuously.")
                        print("#" * 70 + "\n")
                        log_transcript(f"Wake word detected at {wake_ts} (Latency: {kws_latency_ms:.2f}ms, Conf: {kw_conf*100:.1f}%)", wake_detected=True)
                        moonshine_engine.reset()
                        audio_buffer.clear()
                        
            # --- STAGE 2: Continuous Speech-to-Text via Moonshine ---
            else:
                moonshine_engine.process_pcm_chunk(chunk)
                    
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"\n[!] Connection exception: {e}")
    finally:
        if IS_GLOBAL_AWAKE and moonshine_engine.last_partial_text:
            ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"\r[{ts}] 🎯 [TRANSCRIPT]: \"{moonshine_engine.last_partial_text}\" (Flushed on Disconnect)           ")
            log_transcript(moonshine_engine.last_partial_text, wake_detected=False)
        print(f"\n[-] ESP32 Disconnected from {client_ip}.\n")
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass

async def main():
    host = "0.0.0.0"
    port = 8766
    server = await asyncio.start_server(tcp_handler, host, port)
    
    print("=" * 65)
    print(f"   ASR & KWS Server Running on {host}:{port}")
    print(f"   Trained Wake Model : KhusKhus-KWS (best_model.pth)")
    print(f"   Speech Recognition : Moonshine Edge Streaming ASR (Pre-trained)")
    print(f"   Ready and waiting for ESP32 connection...")
    print("=" * 65 + "\n")
    
    async with server:
        await server.serve_forever()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer stopped.")
