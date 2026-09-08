"""
Testing Pipeline for KhusKhus-KWS
Evaluates model on held-out test set with threshold sweeping,
ROC/PR tradeoff analysis, and False Activations per Hour (FAH) estimation.
"""

import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

# Ensure project root is in Python sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from kws.config import (
    AUDIO_CFG,
    DATASET_CFG,
    DECISION_CFG,
    MODEL_CFG,
    TRAIN_CFG,
    AudioConfig,
    DatasetConfig,
    DecisionConfig,
    ModelConfig,
)
from kws.dataset import create_dataloaders
from kws.metrics import calculate_fah, calculate_metrics, plot_confusion_matrix
from kws.model import KhusKhusKWS


def perform_threshold_analysis(
    y_true: np.ndarray,
    kw_probs: np.ndarray,
    thresholds: List[float] = [0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
) -> List[Dict]:
    """Sweep decision threshold to analyze Precision, Recall, FP, and FN."""
    results = []
    is_keyword_true = (y_true == 0)

    for thresh in thresholds:
        predicted_keyword = (kw_probs >= thresh)

        tp = int(np.sum(predicted_keyword & is_keyword_true))
        fp = int(np.sum(predicted_keyword & (~is_keyword_true)))
        fn = int(np.sum((~predicted_keyword) & is_keyword_true))
        tn = int(np.sum((~predicted_keyword) & (~is_keyword_true)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results.append({
            "threshold": thresh,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn
        })

    return results


def plot_threshold_analysis(analysis: List[Dict], save_path: str):
    """Plot Precision, Recall, and F1 vs Detection Threshold."""
    thresh = [x["threshold"] for x in analysis]
    prec = [x["precision"] for x in analysis]
    rec = [x["recall"] for x in analysis]
    f1 = [x["f1"] for x in analysis]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(thresh, prec, "b-o", label="Precision", linewidth=2)
    ax.plot(thresh, rec, "r-s", label="Recall", linewidth=2)
    ax.plot(thresh, f1, "g-^", label="F1-Score", linewidth=2)

    ax.set_xlabel("Keyword Activation Threshold", fontsize=11)
    ax.set_ylabel("Metric Score", fontsize=11)
    ax.set_title("KhusKhus-KWS Threshold Trade-off Curve", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower left", fontsize=10)

    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def test_kws(
    model_path: Optional[Path] = None,
    dataset_cfg: DatasetConfig = DATASET_CFG,
    model_cfg: ModelConfig = MODEL_CFG,
    audio_cfg: AudioConfig = AUDIO_CFG,
    decision_cfg: DecisionConfig = DECISION_CFG
) -> Dict:
    """Run testing on held-out test split."""
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    if model_path is None:
        candidate_paths = [
            dataset_cfg.models_dir / "best_model.pth",
            dataset_cfg.checkpoints_dir / "best_model.pth",
            dataset_cfg.models_dir / "final_model.pth",
        ]
        for p in candidate_paths:
            if p.exists():
                model_path = p
                break

    if model_path is None or not model_path.exists():
        print(f"[ERROR] No trained model found in {dataset_cfg.models_dir}. Please run scripts/train_kws.bat first!")
        sys.exit(1)

    print(f"============================================================")
    print(f"KhusKhus-KWS Final Test Pipeline")
    print(f"============================================================")
    print(f"Model Path:     {model_path}")
    print(f"Device:         {device}")
    print(f"Test Set:       {dataset_cfg.processed_root / 'test'}")
    print(f"============================================================\n")

    # Load Model
    model = KhusKhusKWS(model_cfg).to(device)
    ckpt = torch.load(str(model_path), map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # Load Test Data
    _, _, test_loader = create_dataloaders(
        dataset_cfg=dataset_cfg,
        train_cfg=TRAIN_CFG,
        audio_cfg=audio_cfg
    )

    if test_loader is None or len(test_loader.dataset) == 0:
        print("[ERROR] Test dataset is empty. Run scripts/prepare_dataset.bat first!")
        sys.exit(1)

    all_preds = []
    all_targets = []
    all_kw_probs = []

    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probs = softmax(logits)
            preds = torch.argmax(probs, dim=1)

            kw_p = probs[:, 0].cpu().numpy().tolist()

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(batch_y.numpy().tolist())
            all_kw_probs.extend(kw_p)

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    kw_probs = np.array(all_kw_probs)

    # Calculate baseline metrics (standard argmax)
    metrics = calculate_metrics(y_true, y_pred, keyword_idx=0)

    # Threshold Analysis
    threshold_results = perform_threshold_analysis(y_true, kw_probs)

    # Estimate FAH on non-keyword test samples
    non_keyword_mask = (y_true != 0)
    non_kw_samples = int(np.sum(non_keyword_mask))
    total_non_kw_duration_sec = non_kw_samples * audio_cfg.duration_sec
    kw_false_positives = int(np.sum((kw_probs >= decision_cfg.keyword_threshold) & non_keyword_mask))
    fah = calculate_fah(kw_false_positives, total_non_kw_duration_sec)

    # Output paths
    (dataset_cfg.results_dir / "testing").mkdir(parents=True, exist_ok=True)
    report_txt_path = dataset_cfg.results_dir / "testing" / "test_report.txt"
    metrics_json_path = dataset_cfg.results_dir / "testing" / "test_metrics.json"
    thresh_json_path = dataset_cfg.results_dir / "testing" / "threshold_analysis.json"
    cm_png_path = dataset_cfg.results_dir / "testing" / "confusion_matrix.png"
    thresh_png_path = dataset_cfg.results_dir / "testing" / "threshold_tradeoff.png"

    # Save JSON metrics
    metrics_dict = {
        "model_path": str(model_path),
        "total_test_samples": len(all_targets),
        "overall_accuracy": metrics.accuracy,
        "macro_precision": metrics.macro_precision,
        "macro_recall": metrics.macro_recall,
        "macro_f1": metrics.macro_f1,
        "keyword_precision": metrics.keyword_precision,
        "keyword_recall": metrics.keyword_recall,
        "keyword_f1": metrics.keyword_f1,
        "keyword_false_positives": metrics.keyword_false_positives,
        "keyword_false_negatives": metrics.keyword_false_negatives,
        "confusion_matrix": metrics.confusion_mat.tolist(),
        "configured_threshold": decision_cfg.keyword_threshold,
        "false_activations_at_configured_thresh": kw_false_positives,
        "non_keyword_hours_evaluated": total_non_kw_duration_sec / 3600.0,
        "estimated_fah": fah,
        "threshold_sweep": threshold_results
    }

    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)

    with open(thresh_json_path, "w", encoding="utf-8") as f:
        json.dump(threshold_results, f, indent=2)

    # Save Text Report
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write("============================================================\n")
        f.write("KHUSKHUS-KWS FINAL TEST REPORT\n")
        f.write("============================================================\n\n")
        f.write(f"Model Evaluated:               {model_path}\n")
        f.write(f"Total Held-Out Test Samples:   {len(all_targets)}\n\n")
        f.write("Overall Metrics (Argmax Classifier):\n")
        f.write(f"  Accuracy:                    {metrics.accuracy * 100:.2f}%\n")
        f.write(f"  Macro Precision:             {metrics.macro_precision:.4f}\n")
        f.write(f"  Macro Recall:                {metrics.macro_recall:.4f}\n")
        f.write(f"  Macro F1-Score:              {metrics.macro_f1:.4f}\n\n")
        f.write("Keyword ('khus khus') Detection Performance:\n")
        f.write(f"  Keyword Precision:           {metrics.keyword_precision:.4f} ({metrics.keyword_precision*100:.2f}%)\n")
        f.write(f"  Keyword Recall:              {metrics.keyword_recall:.4f} ({metrics.keyword_recall*100:.2f}%)\n")
        f.write(f"  Keyword F1-Score:            {metrics.keyword_f1:.4f}\n")
        f.write(f"  False Positives (FP):        {metrics.keyword_false_positives}\n")
        f.write(f"  False Negatives (FN):        {metrics.keyword_false_negatives}\n\n")
        f.write("False Activation Rate Analysis:\n")
        f.write(f"  Configured Threshold:        {decision_cfg.keyword_threshold:.2f}\n")
        f.write(f"  Non-Keyword Audio Duration:  {total_non_kw_duration_sec:.1f} s ({total_non_kw_duration_sec/3600.0:.3f} hours)\n")
        f.write(f"  False Activations Count:     {kw_false_positives}\n")
        f.write(f"  Estimated FAH:               {fah:.2f} False Activations / Hour\n\n")
        f.write("Threshold Sweep Trade-off:\n")
        f.write(f"{'-'*60}\n")
        f.write(f"{'Thresh':<8} | {'Precision':<10} | {'Recall':<10} | {'F1':<10} | {'FP':<6} | {'FN':<6}\n")
        f.write(f"{'-'*60}\n")
        for row in threshold_results:
            f.write(f"{row['threshold']:<8.2f} | {row['precision']:<10.4f} | {row['recall']:<10.4f} | {row['f1']:<10.4f} | {row['fp']:<6} | {row['fn']:<6}\n")
        f.write(f"{'-'*60}\n\n")
        f.write("Confusion Matrix:\n")
        f.write("               Predicted ->\n")
        f.write(f"  Actual       {'keyword':<12} {'other_speech':<14} {'noise':<10}\n")
        for i, row_label in enumerate(dataset_cfg.classes):
            row = metrics.confusion_mat[i]
            f.write(f"  {row_label:<12} {row[0]:<12} {row[1]:<14} {row[2]:<10}\n")
        f.write("============================================================\n")

    # Plots
    plot_confusion_matrix(
        metrics.confusion_mat,
        dataset_cfg.classes,
        str(cm_png_path),
        title="KhusKhus-KWS Test Confusion Matrix"
    )
    plot_threshold_analysis(threshold_results, str(thresh_png_path))

    print("Test Evaluation Results:")
    print(f"  Overall Accuracy:      {metrics.accuracy * 100:.2f}%")
    print(f"  Keyword Precision:     {metrics.keyword_precision:.4f}")
    print(f"  Keyword Recall:        {metrics.keyword_recall:.4f}")
    print(f"  Keyword F1-Score:      {metrics.keyword_f1:.4f}")
    print(f"  False Activations/Hr:  {fah:.2f} FAH (at threshold {decision_cfg.keyword_threshold:.2f})")
    print(f"\nArtifacts saved:")
    print(f"  - Report:     {report_txt_path}")
    print(f"  - Metrics:    {metrics_json_path}")
    print(f"  - Matrix:     {cm_png_path}")
    print(f"  - Tradeoff:   {thresh_png_path}")

    return metrics_dict


if __name__ == "__main__":
    test_kws()
