# Moonshine Streaming ASR Model

A standalone, self-contained implementation of the **Moonshine Streaming Automatic Speech Recognition (ASR)** architecture designed for edge devices, microcontrollers (e.g., ESP32-S3), and low-latency applications.

---

## 📁 Repository Structure

```text
.
├── model.py            # Complete self-contained Moonshine Streaming model architecture & session engine
├── requirements.txt    # Minimal dependencies (torch, numpy)
├── README.md           # Model documentation & usage guide
└── .gitignore          # Strict ignore rules for upload
```

---

## 🚀 Quick Usage

### 1. Model Instantiation

```python
from model import MoonshineStreamingASR, StreamingAudioSession

# Instantiate compact ESP32-S3 Micro architecture (~1.5M parameters)
model = MoonshineStreamingASR.create_esp32_micro()

# Or instantiate standard Tiny architecture (~17M parameters)
# model = MoonshineStreamingASR.create_tiny()

print(f"Total Parameters: {model.count_parameters():,}")
print(f"Model Size: {model.model_size_mb():.2f} MB")
```

### 2. Real-Time Streaming Audio Session

```python
import numpy as np
from model import MoonshineStreamingASR, StreamingAudioSession

# Initialize model & streaming session
model = MoonshineStreamingASR.create_esp32_micro()
session = StreamingAudioSession(model=model, chunk_samples=6144)

# Stream 16kHz audio in real-time chunks
# chunk_pcm: np.ndarray of shape (N,) at 16000 Hz
sample_chunk = np.zeros(6144, dtype=np.float32)
emissions = session.feed_samples(sample_chunk)

for delta_text, latency_ms in emissions:
    print(f"Emitted: '{delta_text}' (Latency: {latency_ms:.2f} ms)")

# Finalize stream
final_emissions = session.flush()
print(f"Full Transcript: '{session.current_transcript}'")
```

---

## 🔬 Architecture Highlights

- **3-Layer 1D Causal Convolutional Audio Stem**: Direct raw waveform input downsampled by 384x (~24ms per frame at 16kHz).
- **Rotary Position Embeddings (RoPE)**: Dynamic positional encoding supporting variable audio lengths and streaming frame offsets.
- **Pre-LN Streaming Causal Transformer Encoder**: Stateful key-value (KV) caching for $O(1)$ constant-time chunk processing.
- **Low-Latency CTC Token Decoder**: Real-time non-blocking token collapse and character emissions.
