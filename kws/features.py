"""
KhusKhus-KWS Feature Extraction Module
Deterministic, lightweight Log-Mel Spectrogram computation for ESP32 KWS.
"""

import sys
from pathlib import Path
import time
from typing import Dict, Optional, Tuple, Union

# Ensure project root is in Python sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torchaudio.transforms as T

from kws.config import AUDIO_CFG, AudioConfig


class LogMelFeatureExtractor(nn.Module):
    """
    Lightweight, deterministic Log-Mel Spectrogram extractor.
    Designed to mirror microcontroller-side fixed-point/floating-point FFT + Mel filterbank.

    Input shape:  (batch_size, num_samples) or (num_samples,) with 16000 Hz audio.
    Output shape: (batch_size, n_mels, time_frames) -> e.g. (B, 40, 98) for 1-second audio.
    """

    def __init__(self, cfg: AudioConfig = AUDIO_CFG):
        super().__init__()
        self.cfg = cfg
        self.sample_rate = cfg.sample_rate
        self.n_mels = cfg.n_mels
        self.n_fft = cfg.n_fft
        self.win_length = cfg.win_length
        self.hop_length = cfg.hop_length
        self.f_min = cfg.f_min
        self.f_max = cfg.f_max
        self.center = cfg.center
        self.log_offset = cfg.log_offset

        # Standard torchaudio MelSpectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            f_min=self.f_min,
            f_max=self.f_max,
            n_mels=self.n_mels,
            power=self.cfg.power,
            center=self.center,
            norm="slaney",
            mel_scale="slaney",
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract Log-Mel features.
        Args:
            waveform: Tensor of shape (batch, samples) or (samples,) in range [-1.0, 1.0].
        Returns:
            log_mel: Tensor of shape (batch, n_mels, time_frames)
        """
        # Ensure 2D (batch, samples)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.dim() == 3:
            # e.g., (batch, 1, samples) -> (batch, samples)
            waveform = waveform.squeeze(1)

        # Mel Spectrogram: shape (batch, n_mels, time_frames)
        mel_spec = self.mel_transform(waveform)

        # Log compression: log(mel + eps)
        log_mel = torch.log(mel_spec + self.log_offset)

        # Standard per-utterance mean/std standardization
        mean = log_mel.mean(dim=(-2, -1), keepdim=True)
        std = log_mel.std(dim=(-2, -1), keepdim=True) + 1e-5
        normalized_log_mel = (log_mel - mean) / std

        return normalized_log_mel


def extract_log_mel_numpy(
    audio: np.ndarray,
    cfg: AudioConfig = AUDIO_CFG
) -> np.ndarray:
    """
    NumPy-compatible wrapper for feature extraction.
    Used for streaming audio buffers, benchmarking, and microcontroller simulation.
    """
    tensor_audio = torch.from_numpy(audio.astype(np.float32))
    extractor = LogMelFeatureExtractor(cfg)
    extractor.eval()
    with torch.no_grad():
        features = extractor(tensor_audio)
    return features.squeeze(0).numpy()


def benchmark_feature_extraction(
    num_iterations: int = 500,
    cfg: AudioConfig = AUDIO_CFG
) -> Dict[str, float]:
    """
    Benchmark standalone feature extraction time on 1-second audio chunk.
    Measures CPU execution latency in milliseconds.
    """
    extractor = LogMelFeatureExtractor(cfg)
    extractor.eval()

    # Synthetic 1-second 16kHz audio
    dummy_audio = torch.randn(1, cfg.window_samples, dtype=torch.float32)

    # Warmup
    for _ in range(30):
        _ = extractor(dummy_audio)

    # High-resolution benchmark
    latencies_ms = []
    for _ in range(num_iterations):
        t0 = time.perf_counter_ns()
        with torch.no_grad():
            _ = extractor(dummy_audio)
        t1 = time.perf_counter_ns()
        latencies_ms.append((t1 - t0) / 1_000_000.0)

    latencies = np.array(latencies_ms)
    return {
        "iterations": num_iterations,
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "min_ms": float(np.min(latencies)),
        "median_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "max_ms": float(np.max(latencies)),
    }


if __name__ == "__main__":
    extractor = LogMelFeatureExtractor()
    dummy = torch.randn(1, 16000)
    out = extractor(dummy)
    print(f"Feature Extractor Verification:")
    print(f"  Input Waveform Shape:  {dummy.shape}")
    print(f"  Output Log-Mel Shape:  {out.shape} (Channels/Mel={out.shape[1]}, Time={out.shape[2]})")
    stats = benchmark_feature_extraction(100)
    print(f"  Feature Extraction Latency (100 runs): Mean = {stats['mean_ms']:.3f} ms, P95 = {stats['p95_ms']:.3f} ms")
