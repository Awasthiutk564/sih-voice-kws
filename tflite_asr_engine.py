"""TFLite Streaming ASR Inference Engine.

Runs streaming inference directly on the exported TensorFlow Lite ASR model:
- Uses tf.lite.Interpreter for lightweight, fast execution
- Real-time rolling audio chunk feeding
- Streaming Greedy CTC Character Decoder (collapses blanks/repeats)
- Non-blocking continuous live transcription for ESP32 streams
"""

import os
import time
from typing import List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf


class CTCTokenizer:
    """Character Tokenizer for CTC Decoding matching model.py."""

    DEFAULT_VOCAB = [
        "<blank>", " ",
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
        "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
        "u", "v", "w", "x", "y", "z",
        "'", ".", ",", "?",
    ]

    def __init__(self, vocab: Optional[List[str]] = None):
        self.vocab = vocab if vocab is not None else self.DEFAULT_VOCAB
        self.blank_id = 0
        self.char_to_id = {ch: idx for idx, ch in enumerate(self.vocab)}
        self.id_to_char = {idx: ch for idx, ch in enumerate(self.vocab)}

    def decode(self, token_ids: List[int]) -> str:
        return "".join(
            self.id_to_char[tid]
            for tid in token_ids
            if tid != self.blank_id and tid in self.id_to_char
        )

    def ctc_decode(self, token_ids: List[int]) -> str:
        collapsed = []
        prev = None
        for tid in token_ids:
            if tid != prev:
                if tid != self.blank_id:
                    collapsed.append(tid)
                prev = tid
        return self.decode(collapsed)


class StreamingCTCDecoder:
    """Stateful Streaming CTC Decoder."""

    def __init__(self, tokenizer: Optional[CTCTokenizer] = None, blank_id: int = 0):
        self.tokenizer = tokenizer if tokenizer is not None else CTCTokenizer()
        self.blank_id = blank_id
        self.prev_token_id: Optional[int] = None
        self.emitted_tokens: List[int] = []
        self.full_transcript: str = ""

    def reset(self):
        self.prev_token_id = None
        self.emitted_tokens.clear()
        self.full_transcript = ""

    def decode_chunk_logits(self, logits: np.ndarray) -> Tuple[str, str]:
        """Decodes raw model logits (Time, Vocab) into text."""
        if logits.ndim == 3:
            logits = logits[0]  # Strip batch dim
            
        pred_token_ids = np.argmax(logits, axis=-1).tolist()
        new_chars = []
        for tid in pred_token_ids:
            if tid != self.prev_token_id:
                if tid != self.blank_id:
                    self.emitted_tokens.append(tid)
                    char = self.tokenizer.id_to_char.get(tid, "")
                    new_chars.append(char)
                self.prev_token_id = tid
                
        delta_text = "".join(new_chars)
        self.full_transcript += delta_text
        return delta_text, self.full_transcript


class TFLiteStreamingASREngine:
    """TFLite Streaming ASR Engine for live continuous speech recognition."""

    def __init__(
        self,
        model_path: str = "models/asr/streaming_asr.tflite",
        sample_rate: int = 16000,
        chunk_samples: int = 6144, # ~384ms per forward chunk
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"TFLite model not found at {model_path}. Run tflite_asr_model.py first.")

        self.model_path = model_path
        self.sample_rate = sample_rate
        self.chunk_samples = chunk_samples

        # Initialize TFLite Interpreter
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.tokenizer = CTCTokenizer()
        self.decoder = StreamingCTCDecoder(self.tokenizer, blank_id=0)
        self.audio_buffer = bytearray()
        self.current_sentence: str = ""

    def reset(self):
        self.decoder.reset()
        self.audio_buffer.clear()
        self.current_sentence = ""

    def process_audio_chunk(self, raw_pcm16_chunk: bytes) -> Tuple[Optional[str], str]:
        """Processes incoming 16-bit PCM bytes.
        
        Returns:
            (new_text_delta, full_accumulated_transcript)
        """
        self.audio_buffer.extend(raw_pcm16_chunk)
        
        # Required bytes for one forward chunk: chunk_samples * 2 bytes
        needed_bytes = self.chunk_samples * 2
        new_emission = ""

        while len(self.audio_buffer) >= needed_bytes:
            chunk_bytes = self.audio_buffer[:needed_bytes]
            self.audio_buffer = self.audio_buffer[needed_bytes:]

            # Convert PCM 16-bit to normalized float32 (-1.0 to 1.0)
            samples = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            # DC Offset removal
            samples = samples - np.mean(samples)
            input_tensor = np.expand_dims(samples, axis=0) # (1, chunk_samples)

            # Run TFLite inference
            self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
            self.interpreter.invoke()
            logits = self.interpreter.get_tensor(self.output_details[0]['index']) # (1, Time, Vocab)

            # Decode logits with streaming CTC
            delta, full_text = self.decoder.decode_chunk_logits(logits)
            if delta:
                new_emission += delta
                self.current_sentence += delta

        return (new_emission if new_emission else None), self.decoder.full_transcript


if __name__ == "__main__":
    print("=" * 60)
    print("  Testing TFLite Streaming ASR Engine")
    print("=" * 60)
    
    engine = TFLiteStreamingASREngine("models/asr/streaming_asr.tflite")
    print("[+] TFLite Interpreter Loaded Successfully!")
    print(f"    Input Details: {engine.input_details[0]['shape']}")
    print(f"    Output Details: {engine.output_details[0]['shape']}")

    # Simulate 2 seconds of synthetic audio stream in 1024-byte packets
    dummy_bytes = (np.random.randn(32000) * 5000).astype(np.int16).tobytes()
    print("\n[*] Streaming synthetic audio through TFLite Engine...")
    
    t0 = time.perf_counter()
    for i in range(0, len(dummy_bytes), 1024):
        packet = dummy_bytes[i:i+1024]
        delta, full = engine.process_audio_chunk(packet)
    t1 = time.perf_counter()
    
    print(f"[+] Total Stream Duration: 2.00s | Process Time: {(t1 - t0)*1000:.2f} ms")
    print("[+] Engine verified successfully!\n")
