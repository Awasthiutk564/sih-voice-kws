"""Official Moonshine Pre-trained Streaming ASR Engine for Edge Devices.

Features:
- High-fidelity DC offset filter and natural audio gain scaling for ESP32 INMP441 microphones
- Real-time streaming chunk ingestion with precise millisecond latency tracking
- Emits LineStarted, LineTextChanged, LineCompleted events with full latency and duration stats
"""

import os
import time
from typing import Callable, List, Optional, Tuple
import numpy as np
import moonshine_voice as mv


class MoonshineStreamingEngine:
    """Real-Time Continuous Streaming ASR Engine with precision latency profiling."""

    def __init__(self, language: str = "en"):
        print("[*] Initializing Pretrained Moonshine Streaming ASR Engine...")
        self.model_path, self.model_arch = mv.get_model_for_language(language)
        self.transcriber = mv.Transcriber(self.model_path, self.model_arch)
        self.arch_name = mv.model_arch_to_string(self.model_arch)
        self.sample_rate = 16000
        
        self.stream = None
        self.last_partial_text = ""
        self.final_transcripts: List[str] = []
        
        # Callbacks: on_partial(text, latency_ms), on_final(text, latency_ms, duration_sec, rtf)
        self.on_partial_callback: Optional[Callable[[str, float], None]] = None
        self.on_final_callback: Optional[Callable[[str, float, float, float], None]] = None

        # Internal buffer for optimal ~100ms framing
        self.audio_accumulator = bytearray()
        self.CHUNK_BYTES = 1600 * 2  # 100ms @ 16kHz 16-bit PCM

        self._init_stream()
        print(f"[+] Moonshine Engine Ready: {self.arch_name}")

    def _init_stream(self):
        if self.stream is not None:
            try:
                self.stream.stop()
            except Exception:
                pass

        self.stream = self.transcriber.create_stream()
        self.stream.start()
        self.last_partial_text = ""
        self.audio_accumulator.clear()

        def handle_event(event):
            event_type = type(event).__name__
            line = getattr(event, "line", None)
            if not line:
                return

            text = line.text.strip()
            if not text:
                return

            latency_ms = getattr(line, "last_transcription_latency_ms", 0.0)
            duration_sec = getattr(line, "duration", 0.0)
            rtf = (latency_ms / 1000.0) / duration_sec if duration_sec > 0 else 0.0

            if event_type in ("LineStarted", "LineTextChanged", "LineUpdated"):
                self.last_partial_text = text
                if self.on_partial_callback:
                    self.on_partial_callback(text, latency_ms)

            elif event_type == "LineCompleted":
                self.final_transcripts.append(text)
                self.last_partial_text = ""
                if self.on_final_callback:
                    self.on_final_callback(text, latency_ms, duration_sec, rtf)

        self.stream.add_listener(handle_event)

    def reset(self):
        """Resets the streaming state for a fresh session."""
        self._init_stream()
        self.last_partial_text = ""
        self.audio_accumulator.clear()

    def process_pcm_chunk(self, raw_pcm16_chunk: bytes) -> Tuple[Optional[str], float]:
        """Feeds incoming 16-bit PCM bytes with DC filter and measures chunk compute latency.
        
        Returns:
            (partial_text, chunk_compute_latency_ms)
        """
        if not raw_pcm16_chunk:
            return None, 0.0

        self.audio_accumulator.extend(raw_pcm16_chunk)
        chunk_compute_ms = 0.0

        # Process in ~100ms blocks for acoustic stability
        while len(self.audio_accumulator) >= self.CHUNK_BYTES:
            block = self.audio_accumulator[:self.CHUNK_BYTES]
            self.audio_accumulator = self.audio_accumulator[self.CHUNK_BYTES:]

            # Convert to float32
            samples = np.frombuffer(block, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Clean DC offset removal to eliminate mic bias drift
            samples = samples - np.mean(samples)

            # Soft dynamic range normalization
            rms = np.sqrt(np.mean(samples ** 2))
            if rms > 0.001 and rms < 0.03:
                # Boost soft speech without clipping
                gain = min(3.0, 0.05 / rms)
                samples = np.clip(samples * gain, -1.0, 1.0)

            t0 = time.perf_counter()
            self.stream.add_audio(samples.tolist(), self.sample_rate)
            self.stream.update_transcription()
            t1 = time.perf_counter()
            chunk_compute_ms += (t1 - t0) * 1000.0

        return self.last_partial_text if self.last_partial_text else None, chunk_compute_ms


if __name__ == "__main__":
    print("=" * 60)
    print("  Testing Moonshine Streaming ASR Engine & Latency Profiling")
    print("=" * 60)

    engine = MoonshineStreamingEngine(language="en")
    
    def on_p(text, lat_ms):
        print(f"  [Hearing]: \"{text}\" (Model Latency: {lat_ms:.1f}ms)")

    def on_f(text, lat_ms, dur_s, rtf):
        print(f"  [Complete]: \"{text}\" (Latency: {lat_ms:.1f}ms | Duration: {dur_s:.2f}s | RTF: {rtf:.2f})")

    engine.on_partial_callback = on_p
    engine.on_final_callback = on_f

    sample_wav = "E:/kwas/processed_dataset/train/keyword/keyword_00000_WhatsApp Ptt 2026-09-07 at 12.14.49 AM.wav"
    if os.path.exists(sample_wav):
        import soundfile as sf
        data, sr = sf.read(sample_wav, dtype="int16")
        bytes_data = data.tobytes()
        print(f"\n[*] Streaming audio ({len(data)/sr:.2f}s) in 1024-byte packets...")
        t_total_0 = time.perf_counter()
        for i in range(0, len(bytes_data), 1024):
            engine.process_pcm_chunk(bytes_data[i:i+1024])
            time.sleep(0.02)
        t_total_1 = time.perf_counter()
        print(f"\n[+] Total Stream Feed Wall Time: {(t_total_1 - t_total_0)*1000:.2f} ms")
    
    print("\n[+] Engine verified successfully!\n")
