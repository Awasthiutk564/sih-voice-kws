# sih-voice-kws
### Low-Latency and Efficient Voice Activator & Streaming ASR for Edge Devices

`sih-voice-kws` is a split-architecture (**Edge + Server**) voice activation and speech recognition system designed to replace resource-heavy, privacy-invasive cloud listeners with a localized TinyML wake-word engine and ultra-fast streaming ASR.

---

## 🌟 Overview & Architecture

```
                                      ┌──────────────────────────────────────────────┐
                                      │              Server / Host Side              │
                                      │                                              │
┌───────────────────────────┐         │  ┌────────────────┐     ┌──────────────────┐ │
│   ESP32-S3 Edge Device    │         │  │ WebSocket      │     │ Streaming        │ │
│                           │         │  │ Server         │────▶│ ASR Engine       │ │
│  ┌─────────────────────┐  │  Audio  │  │ (server.py)    │     │ (model.py)       │ │
│  │ I2S DMA Ring Buffer │──┼─────────┼─▶└────────────────┘     └──────────────────┘ │
│  │ (2s Pre-roll audio) │  │ Stream  │                                              │
│  └──────────┬──────────┘  │ (ws://) │  ┌─────────────────────────────────────────┐ │
│             │             │         │  │ Real-Time Low-Latency Text Output       │ │
│  ┌──────────▼──────────┐  │         │  │ ("Intent / Command Parsing")            │ │
│  │ Local TinyML KWS    │  │         │  └─────────────────────────────────────────┘ │
│  │ (TFLM Wake-Word)    │  │         │                                              │
│  └─────────────────────┘  │         └──────────────────────────────────────────────┘
└───────────────────────────┘
```

1. **ESP32-S3 Edge (TinyML KWS)**: Continuously samples I2S audio via Direct Memory Access (DMA) into a 2-second pre-roll ring buffer while keeping the CPU idle. A highly quantized CNN powered by TensorFlow Lite for Microcontrollers (TFLM) analyzes the buffer locally.
2. **Zero-Clipping Retroactive Stream**: Upon detecting a custom wake word, it instantly opens a WebSocket connection and streams the buffered pre-roll audio along with the live audio feed (eliminating first-word clipping).
3. **Streaming ASR Engine**: Real-time causal streaming Automated Speech Recognition engine with $O(1)$ stateful KV caching for instant speech-to-text transcriptions.

---

## 📁 Repository Structure

```text
.
├── esp32_firmware/
│   ├── esp32_firmware.ino                # ESP32-S3 I2S DMA audio capture & TCP streaming
│   ├── khuskhus_kws_tflite_model.h       # C byte array header of trained "Khus Khus" KWS
│   └── streaming_asr_tflite_model.h      # C byte array header of Streaming ASR for ESP32
├── server.py                             # 2-Stage TCP server (KhusKhus-KWS -> TFLite ASR)
├── tflite_asr_model.py                   # TensorFlow/Keras replica of model.py + TFLite & C export
├── tflite_asr_engine.py                  # TFLite Interpreter streaming ASR engine with CTC decoder
├── model.py                              # PyTorch Streaming ASR architecture & session engine
├── models/
│   ├── kws/best_model.pth                # Trained KhusKhus-KWS PyTorch weights
│   └── asr/streaming_asr.tflite          # Exported TensorFlow Lite Streaming ASR model
├── live_transcript_log.txt               # Real-time transcript logging
├── requirements.txt                      # Python dependencies (torch, tensorflow, numpy)
└── README.md                             # System documentation & usage guide
```

---

## 🚀 Getting Started

### 1. Installation

```bash
git clone https://github.com/Awasthiutk564/sih-voice-kws.git
cd sih-voice-kws
pip install -r requirements.txt
```

### 2. Streaming ASR Usage (`model.py`)

#### Instantiation
```python
from model import StreamingASR, StreamingAudioSession

# Instantiate compact ESP32-S3 Micro architecture (~1.5M parameters)
model = StreamingASR.create_esp32_micro()

# Or instantiate standard Tiny architecture (~17M parameters)
# model = StreamingASR.create_tiny()

print(f"Total Parameters: {model.count_parameters():,}")
print(f"Model Size: {model.model_size_mb():.2f} MB")
```

#### Real-Time Streaming Audio Session
```python
import numpy as np
from model import StreamingASR, StreamingAudioSession

# Initialize model & streaming session
model = StreamingASR.create_esp32_micro()
session = StreamingAudioSession(model=model, chunk_samples=6144)

# Stream 16kHz audio in real-time chunks (e.g. from WebSocket)
sample_chunk = np.zeros(6144, dtype=np.float32)
emissions = session.feed_samples(sample_chunk)

for delta_text, latency_ms in emissions:
    print(f"Emitted: '{delta_text}' (Latency: {latency_ms:.2f} ms)")

# Finalize stream
final_emissions = session.flush()
print(f"Full Transcript: '{session.current_transcript}'")
```

### 3. Running the Server & Voice Assistant

```bash
python server.py
```
- **Stage 1 (Wake Listener)**: Continuously monitors the live 16kHz audio stream from the ESP32-S3 INMP441 microphone for the wake word **"Khus Khus"**.
- **Stage 2 (Continuous Live Transcription)**: As soon as "Khus Khus" is detected, KWS stops and high-accuracy ASR takes over indefinitely, transcribing all continuous speech in real time to the console and `live_transcript_log.txt`.

### 4. Running the ESP32-S3 Firmware
Flash `esp32_firmware/esp32_firmware.ino` to your ESP32-S3 board with the INMP441 microphone connected:
- **WS / LRCL**: `GPIO 5`
- **SD / DOUT**: `GPIO 4`
- **SCK / BCLK**: `GPIO 6`
- **L/R**: `GND` (Left channel)
- **VDD**: `3.3V`, **GND**: `GND`

---

## 🔬 Key Features & Specifications

- **100% Open-Source**: No proprietary SDKs (e.g., Porcupine, Picovoice).
- **Ultra-Low Edge Resources**: Optimized to run within < 256KB RAM and < 10% CPU utilization on ESP32-S3.
- **Zero-Clipping Latency**: The DMA ring buffer ensures retroactive audio streaming, eliminating wake-up delay data loss.
- **Causal Streaming ASR**:
  - **3-Layer 1D Causal Convolutional Audio Stem**: Direct raw waveform input downsampled by 384x (~24ms per frame at 16kHz).
  - **Rotary Position Embeddings (RoPE)**: Dynamic positional encoding supporting variable audio lengths and streaming frame offsets.
  - **Pre-LN Streaming Causal Transformer Encoder**: Stateful key-value (KV) caching for $O(1)$ constant-time chunk processing.
  - **Low-Latency CTC Token Decoder**: Real-time non-blocking token collapse and character emissions.