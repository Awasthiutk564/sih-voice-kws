"""
PyTorch Dataset and Dataloaders for KhusKhus-KWS
Includes streaming window slicing, deterministic validation/testing,
and training-only audio augmentations (gain, time-shift, noise mixing, SpecAugment).
"""

import math
import os
from pathlib import Path
import random
import sys
from typing import Dict, List, Optional, Tuple

# Ensure project root is in Python sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset

from kws.config import (
    AUDIO_CFG,
    AUG_CFG,
    DATASET_CFG,
    TRAIN_CFG,
    AudioConfig,
    AugmentationConfig,
    DatasetConfig,
)
from kws.features import LogMelFeatureExtractor


class AudioAugmenter:
    """
    Lightweight audio augmentations for training data.
    Ensures wake-word robustness to background noise, speaker volume, and alignment.
    """

    def __init__(
        self,
        aug_cfg: AugmentationConfig = AUG_CFG,
        sample_rate: int = AUDIO_CFG.sample_rate,
        noise_files: Optional[List[Path]] = None
    ):
        self.cfg = aug_cfg
        self.sample_rate = sample_rate
        self.noise_files = noise_files or []

    def apply_gain(self, audio: np.ndarray) -> np.ndarray:
        """Apply random gain in dB."""
        gain_db = random.uniform(self.cfg.gain_db_range[0], self.cfg.gain_db_range[1])
        gain_linear = 10.0 ** (gain_db / 20.0)
        return audio * gain_linear

    def apply_time_shift(self, audio: np.ndarray) -> np.ndarray:
        """Apply random temporal roll/shift."""
        max_shift = int(self.cfg.time_shift_sec * self.sample_rate)
        shift = random.randint(-max_shift, max_shift)
        return np.roll(audio, shift)

    def apply_noise_mixing(self, audio: np.ndarray) -> np.ndarray:
        """Mix real background noise at a random SNR."""
        if not self.noise_files or random.random() > self.cfg.noise_prob:
            return audio

        noise_path = random.choice(self.noise_files)
        try:
            noise_audio, sr = sf.read(str(noise_path), dtype="float32")
            if noise_audio.ndim > 1:
                noise_audio = np.mean(noise_audio, axis=1)

            # Match length
            if len(noise_audio) < len(audio):
                repeats = int(np.ceil(len(audio) / len(noise_audio)))
                noise_audio = np.tile(noise_audio, repeats)[:len(audio)]
            elif len(noise_audio) > len(audio):
                start = random.randint(0, len(noise_audio) - len(audio))
                noise_audio = noise_audio[start:start + len(audio)]

            # Calculate RMS
            audio_rms = np.sqrt(np.mean(audio ** 2) + 1e-8)
            noise_rms = np.sqrt(np.mean(noise_audio ** 2) + 1e-8)

            target_snr_db = random.uniform(self.cfg.noise_snr_db_range[0], self.cfg.noise_snr_db_range[1])
            target_noise_rms = audio_rms / (10.0 ** (target_snr_db / 20.0))

            scaled_noise = noise_audio * (target_noise_rms / noise_rms)
            mixed = audio + scaled_noise
            return mixed
        except Exception:
            return audio

    def apply_spec_augment(self, log_mel: torch.Tensor) -> torch.Tensor:
        """
        Lightweight SpecAugment (Time & Frequency Masking on Log-Mel Spectrogram).
        Shape: (n_mels, time_frames)
        """
        # Frequency masking
        if self.cfg.freq_mask_param > 0 and random.random() < 0.5:
            num_mels, num_time = log_mel.shape
            f_mask_len = random.randint(1, self.cfg.freq_mask_param)
            f_start = random.randint(0, max(0, num_mels - f_mask_len))
            log_mel[f_start:f_start + f_mask_len, :] = 0.0

        # Time masking
        if self.cfg.time_mask_param > 0 and random.random() < 0.5:
            num_mels, num_time = log_mel.shape
            t_mask_len = random.randint(1, self.cfg.time_mask_param)
            t_start = random.randint(0, max(0, num_time - t_mask_len))
            log_mel[:, t_start:t_start + t_mask_len] = 0.0

        return log_mel


class KWSDataset(Dataset):
    """
    KWS PyTorch Dataset.
    Loads standardized 16 kHz audio files and generates Log-Mel spectrograms.
    """

    def __init__(
        self,
        split_dir: Path,
        audio_cfg: AudioConfig = AUDIO_CFG,
        dataset_cfg: DatasetConfig = DATASET_CFG,
        aug_cfg: AugmentationConfig = AUG_CFG,
        is_training: bool = False
    ):
        self.split_dir = split_dir
        self.audio_cfg = audio_cfg
        self.dataset_cfg = dataset_cfg
        self.aug_cfg = aug_cfg
        self.is_training = is_training
        self.target_samples = audio_cfg.window_samples

        self.samples: List[Tuple[Path, int]] = []
        self.feature_extractor = LogMelFeatureExtractor(audio_cfg)

        # Collect files per class
        noise_files = []
        for cls_name, cls_idx in dataset_cfg.class_to_idx.items():
            cls_dir = split_dir / cls_name
            if cls_dir.exists():
                for f in sorted(cls_dir.glob("*.wav")):
                    self.samples.append((f, cls_idx))
                    if cls_name == "noise":
                        noise_files.append(f)

        self.augmenter = AudioAugmenter(aug_cfg, audio_cfg.sample_rate, noise_files) if is_training else None

    def __len__(self) -> int:
        return len(self.samples)

    def _prepare_1s_audio(self, audio: np.ndarray) -> np.ndarray:
        """Crop or pad audio to exactly 1 second (16000 samples)."""
        num_samples = len(audio)
        if num_samples == self.target_samples:
            return audio
        elif num_samples > self.target_samples:
            if self.is_training:
                # Random window slice for training
                start = random.randint(0, num_samples - self.target_samples)
            else:
                # Center slice for validation/testing
                start = (num_samples - self.target_samples) // 2
            return audio[start:start + self.target_samples]
        else:
            # Pad with zeros
            pad_needed = self.target_samples - num_samples
            if self.is_training:
                pad_left = random.randint(0, pad_needed)
            else:
                pad_left = pad_needed // 2
            pad_right = pad_needed - pad_left
            return np.pad(audio, (pad_left, pad_right), mode="constant", constant_values=0.0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        filepath, label = self.samples[idx]

        try:
            audio, sr = sf.read(str(filepath), dtype="float32")
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
        except Exception:
            # Fallback zero array if corrupted
            audio = np.zeros(self.target_samples, dtype=np.float32)

        # Ensure exact 1.0 second duration
        audio = self._prepare_1s_audio(audio)

        # Apply raw audio augmentations if in training mode
        if self.is_training and self.augmenter and self.aug_cfg.enabled:
            audio = self.augmenter.apply_time_shift(audio)
            audio = self.augmenter.apply_gain(audio)
            if label != self.dataset_cfg.class_to_idx["noise"]:
                audio = self.augmenter.apply_noise_mixing(audio)

        # Extract features (Log-Mel)
        tensor_audio = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            features = self.feature_extractor(tensor_audio).squeeze(0)  # Shape: (n_mels, time_frames)

        # Apply SpecAugment if training
        if self.is_training and self.augmenter and self.aug_cfg.enabled:
            features = self.augmenter.apply_spec_augment(features)

        return features, label


def create_dataloaders(
    dataset_cfg: DatasetConfig = DATASET_CFG,
    train_cfg: TRAIN_CFG = TRAIN_CFG,
    audio_cfg: AudioConfig = AUDIO_CFG,
    aug_cfg: AugmentationConfig = AUG_CFG
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """
    Create PyTorch DataLoaders for train, validation, and test splits.
    """
    train_dir = dataset_cfg.processed_root / "train"
    val_dir = dataset_cfg.processed_root / "validation"
    test_dir = dataset_cfg.processed_root / "test"

    train_loader = None
    val_loader = None
    test_loader = None

    if train_dir.exists():
        train_ds = KWSDataset(train_dir, audio_cfg, dataset_cfg, aug_cfg, is_training=True)
        if len(train_ds) > 0:
            train_loader = DataLoader(
                train_ds,
                batch_size=train_cfg.batch_size,
                shuffle=True,
                num_workers=train_cfg.num_workers,
                pin_memory=True if train_cfg.device == "cuda" else False,
                drop_last=len(train_ds) > train_cfg.batch_size
            )

    if val_dir.exists():
        val_ds = KWSDataset(val_dir, audio_cfg, dataset_cfg, aug_cfg, is_training=False)
        if len(val_ds) > 0:
            val_loader = DataLoader(
                val_ds,
                batch_size=train_cfg.batch_size,
                shuffle=False,
                num_workers=train_cfg.num_workers,
                pin_memory=True if train_cfg.device == "cuda" else False
            )

    if test_dir.exists():
        test_ds = KWSDataset(test_dir, audio_cfg, dataset_cfg, aug_cfg, is_training=False)
        if len(test_ds) > 0:
            test_loader = DataLoader(
                test_ds,
                batch_size=train_cfg.batch_size,
                shuffle=False,
                num_workers=train_cfg.num_workers,
                pin_memory=True if train_cfg.device == "cuda" else False
            )

    return train_loader, val_loader, test_loader
