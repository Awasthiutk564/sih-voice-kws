"""
Metrics and Temporal Decision Logic for KhusKhus-KWS
Includes precision/recall/F1, confusion matrix, False Activations per Hour (FAH),
and a stateful streaming temporal decision engine.
"""

from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

# Ensure project root is in Python sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from kws.config import DECISION_CFG, DecisionConfig


@dataclass
class EvaluationMetrics:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    keyword_precision: float
    keyword_recall: float
    keyword_f1: float
    keyword_false_positives: int
    keyword_false_negatives: int
    confusion_mat: np.ndarray


def calculate_metrics(
    y_true: Union[np.ndarray, List[int]],
    y_pred: Union[np.ndarray, List[int]],
    keyword_idx: int = 0
) -> EvaluationMetrics:
    """
    Calculate classification metrics focusing on wake-word detection quality.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    p_per_class, r_per_class, f1_per_class, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0
    )

    kw_p = float(p_per_class[keyword_idx])
    kw_r = float(r_per_class[keyword_idx])
    kw_f1 = float(f1_per_class[keyword_idx])

    # False Positives for keyword: predicted 0 when true was not 0
    kw_fp = int(np.sum((y_pred == keyword_idx) & (y_true != keyword_idx)))
    # False Negatives for keyword: predicted not 0 when true was 0
    kw_fn = int(np.sum((y_pred != keyword_idx) & (y_true == keyword_idx)))

    return EvaluationMetrics(
        accuracy=acc,
        macro_precision=float(p_macro),
        macro_recall=float(r_macro),
        macro_f1=float(f1_macro),
        keyword_precision=kw_p,
        keyword_recall=kw_r,
        keyword_f1=kw_f1,
        keyword_false_positives=kw_fp,
        keyword_false_negatives=kw_fn,
        confusion_mat=cm
    )


def calculate_fah(
    num_false_activations: int,
    total_audio_duration_seconds: float
) -> float:
    """
    Calculate False Activations per Hour (FAH).
    FAH = (Total False Triggers / Total Audio Duration in Hours)
    """
    if total_audio_duration_seconds <= 0:
        return 0.0
    total_hours = total_audio_duration_seconds / 3600.0
    return float(num_false_activations / total_hours)


class TemporalDecisionEngine:
    """
    Stateful streaming post-processing layer.
    Filters out momentary spurious noise triggers using consecutive confirmation and cooldown timer.
    """

    def __init__(self, cfg: DecisionConfig = DECISION_CFG, hop_sec: float = 0.03):
        self.cfg = cfg
        self.hop_sec = hop_sec
        self.threshold = cfg.keyword_threshold
        self.consecutive_required = cfg.consecutive_frames
        self.cooldown_steps = int((cfg.cooldown_ms / 1000.0) / hop_sec)

        # Stateful registers
        self.consecutive_count = 0
        self.cooldown_remaining = 0
        self.total_triggers = 0

    def reset(self):
        """Reset stateful registers."""
        self.consecutive_count = 0
        self.cooldown_remaining = 0
        self.total_triggers = 0

    def process_frame(self, keyword_prob: float) -> Tuple[bool, Dict]:
        """
        Process single inference step probability.
        Returns:
            triggered (bool): True if WAKE condition is met
            debug_info (dict): State details
        """
        t0 = time.perf_counter_ns()
        triggered = False

        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            self.consecutive_count = 0
        else:
            if keyword_prob >= self.threshold:
                self.consecutive_count += 1
                if self.consecutive_count >= self.consecutive_required:
                    triggered = True
                    self.total_triggers += 1
                    self.cooldown_remaining = self.cooldown_steps
                    self.consecutive_count = 0
            else:
                self.consecutive_count = 0

        t1 = time.perf_counter_ns()
        decision_latency_us = (t1 - t0) / 1000.0  # Microseconds

        return triggered, {
            "keyword_prob": keyword_prob,
            "consecutive_count": self.consecutive_count,
            "cooldown_remaining": self.cooldown_remaining,
            "triggered": triggered,
            "decision_latency_us": decision_latency_us
        }


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str,
    title: str = "KhusKhus-KWS Confusion Matrix"
):
    """Save normalized and raw confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label"
    )

    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_training_history(history: Dict, save_path: str):
    """Plot training and validation loss, accuracy, and Keyword F1 curves."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    # Loss
    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=3)
    ax1.plot(epochs, history["val_loss"], "r-s", label="Val Loss", markersize=3)
    ax1.set_ylabel("Loss")
    ax1.set_title("KhusKhus-KWS Training Dynamics")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # Accuracy
    ax2.plot(epochs, history["train_acc"], "b-o", label="Train Accuracy", markersize=3)
    ax2.plot(epochs, history["val_acc"], "r-s", label="Val Accuracy", markersize=3)
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    # Keyword F1
    ax3.plot(epochs, history["val_kw_f1"], "g-^", label="Val Keyword F1", markersize=3)
    ax3.plot(epochs, history["val_kw_recall"], "m-d", label="Val Keyword Recall", markersize=3)
    ax3.set_ylabel("Score")
    ax3.set_xlabel("Epoch")
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend()

    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
