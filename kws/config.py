"""
Centralized Configuration for KhusKhus-KWS
Wake Phrase: "khus khus"
Target: Low-latency streaming Keyword Spotting on ESP32
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple
import torch


@dataclass
class AudioConfig:
    """Audio signal processing and feature extraction parameters."""
    sample_rate: int = 16000
    channels: int = 1  # Mono
    duration_sec: float = 1.0  # 1-second analysis window
    window_samples: int = 16000  # sample_rate * duration_sec

    # Log-Mel Spectrogram parameters
    n_mels: int = 40
    n_fft: int = 400      # 25 ms window at 16 kHz
    win_length: int = 400 # 25 ms
    hop_length: int = 160 # 10 ms hop at 16 kHz
    f_min: float = 20.0
    f_max: float = 7600.0
    power: float = 2.0
    log_offset: float = 1e-6

    # Expected time frames = (16000 - 400) // 160 + 1 = 98 (with center=False) or 101 (with center=True)
    center: bool = False
    expected_time_frames: int = 98  # (16000 - 400) // 160 + 1

    # Streaming parameters
    hop_sec: float = 0.03  # 30 ms streaming step (configurable 20 - 40 ms)
    hop_samples: int = 480 # sample_rate * hop_sec


@dataclass
class DatasetConfig:
    """Dataset paths, split ratios, and class mapping."""
    project_root: Path = Path(r"E:\kwas")
    dataset_root: Path = Path(r"E:\kwas\dataset")
    processed_root: Path = Path(r"E:\kwas\processed_dataset")
    models_dir: Path = Path(r"E:\kwas\models\KhusKhus-KWS")
    checkpoints_dir: Path = Path(r"E:\kwas\models\KhusKhus-KWS\checkpoints")
    results_dir: Path = Path(r"E:\kwas\results")

    # Split ratios
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42

    # Classes
    classes: List[str] = field(default_factory=lambda: ["keyword", "other_speech", "noise"])
    class_to_idx: Dict[str, int] = field(default_factory=lambda: {
        "keyword": 0,
        "other_speech": 1,
        "noise": 2,
    })
    idx_to_class: Dict[int, str] = field(default_factory=lambda: {
        0: "keyword",
        1: "other_speech",
        2: "noise",
    })
    num_classes: int = 3

    # Supported audio extensions
    supported_extensions: Tuple[str, ...] = (".wav", ".flac", ".mp3", ".m4a", ".ogg")


@dataclass
class AugmentationConfig:
    """Lightweight training-only data augmentation."""
    enabled: bool = True
    gain_db_range: Tuple[float, float] = (-6.0, 6.0)
    time_shift_sec: float = 0.10  # Max +/- 100 ms shift
    noise_prob: float = 0.60
    noise_snr_db_range: Tuple[float, float] = (5.0, 20.0)
    time_mask_param: int = 8      # Max time frames to mask
    freq_mask_param: int = 4      # Max mel bins to mask
    speed_perturbation_range: Tuple[float, float] = (0.95, 1.05)


@dataclass
class ModelConfig:
    """KhusKhus-KWS 1D Depthwise-Separable CNN architecture parameters."""
    name: str = "KhusKhus-KWS"
    in_channels: int = 40        # Number of Mel frequency bins
    stem_channels: int = 32
    stem_kernel: int = 5
    stem_stride: int = 1

    # DS Block 1: 32 -> 48, kernel=5, stride=2 (time downsampling)
    ds1_channels: int = 48
    ds1_kernel: int = 5
    ds1_stride: int = 2

    # DS Block 2: 48 -> 64, kernel=5, stride=2 (time downsampling)
    ds2_channels: int = 64
    ds2_kernel: int = 5
    ds2_stride: int = 2

    # DS Block 3: 64 -> 64, kernel=3, stride=1 (residual connection)
    ds3_channels: int = 64
    ds3_kernel: int = 3
    ds3_stride: int = 1

    num_classes: int = 3
    dropout: float = 0.10
    use_relu6: bool = True       # ReLU6 is highly compatible with TFLite Micro / INT8


@dataclass
class DecisionConfig:
    """Temporal smoothing and post-processing decision layer."""
    keyword_threshold: float = 0.80      # Probability threshold for keyword class
    consecutive_frames: int = 2          # Consecutive positive predictions required
    cooldown_ms: float = 1000.0          # Refractory period in ms after trigger
    hysteresis_low: float = 0.50         # Low threshold for hysteresis if enabled


@dataclass
class TrainingConfig:
    """Training hyperparameters and optimization settings."""
    batch_size: int = 32
    learning_rate: float = 1e-3
    min_learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    epochs: int = 25                     # 25 epochs on 175k samples is ideal for convergence
    patience: int = 8                    # Early stopping patience
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0                 # Safe default for Windows multiprocessing
    mixed_precision: bool = True if torch.cuda.is_available() else False
    gradient_clip: float = 1.0


# Single global default config instances for clean access
AUDIO_CFG = AudioConfig()
DATASET_CFG = DatasetConfig()
AUG_CFG = AugmentationConfig()
MODEL_CFG = ModelConfig()
DECISION_CFG = DecisionConfig()
TRAIN_CFG = TrainingConfig()
