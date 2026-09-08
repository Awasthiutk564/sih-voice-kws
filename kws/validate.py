"""
Validation Pipeline for KhusKhus-KWS
Evaluates the best trained model strictly on the held-out validation dataset.
Never modifies weights or executes training.
"""

import json
from pathlib import Path
import sys
from typing import Dict, Optional

# Ensure project root is in Python sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn

from kws.config import (
    AUDIO_CFG,
    DATASET_CFG,
    MODEL_CFG,
    TRAIN_CFG,
    AudioConfig,
    DatasetConfig,
    ModelConfig,
)
from kws.dataset import create_dataloaders
from kws.metrics import calculate_metrics, plot_confusion_matrix
from kws.model import KhusKhusKWS


def validate_kws(
    model_path: Optional[Path] = None,
    dataset_cfg: DatasetConfig = DATASET_CFG,
    model_cfg: ModelConfig = MODEL_CFG,
    audio_cfg: AudioConfig = AUDIO_CFG
) -> Dict:
    """Run validation evaluation on validation split."""
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
    print(f"KhusKhus-KWS Validation Pipeline")
    print(f"============================================================")
    print(f"Model Path:     {model_path}")
    print(f"Device:         {device}")
    print(f"Validation Set: {dataset_cfg.processed_root / 'validation'}")
    print(f"============================================================\n")

    # Load Model
    model = KhusKhusKWS(model_cfg).to(device)
    ckpt = torch.load(str(model_path), map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # Load Validation Data
    _, val_loader, _ = create_dataloaders(
        dataset_cfg=dataset_cfg,
        train_cfg=TRAIN_CFG,
        audio_cfg=audio_cfg
    )

    if val_loader is None or len(val_loader.dataset) == 0:
        print("[ERROR] Validation dataset is empty. Run scripts/prepare_dataset.bat first!")
        sys.exit(1)

    all_preds = []
    all_targets = []
    all_probs = []

    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probs = softmax(logits)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(batch_y.numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    metrics = calculate_metrics(all_targets, all_preds, keyword_idx=0)

    # Save results
    (dataset_cfg.results_dir / "validation").mkdir(parents=True, exist_ok=True)
    report_txt_path = dataset_cfg.results_dir / "validation" / "validation_report.txt"
    metrics_json_path = dataset_cfg.results_dir / "validation" / "validation_metrics.json"
    cm_png_path = dataset_cfg.results_dir / "validation" / "confusion_matrix.png"

    # Save JSON
    metrics_dict = {
        "model_path": str(model_path),
        "total_samples": len(all_targets),
        "accuracy": metrics.accuracy,
        "macro_precision": metrics.macro_precision,
        "macro_recall": metrics.macro_recall,
        "macro_f1": metrics.macro_f1,
        "keyword_precision": metrics.keyword_precision,
        "keyword_recall": metrics.keyword_recall,
        "keyword_f1": metrics.keyword_f1,
        "keyword_false_positives": metrics.keyword_false_positives,
        "keyword_false_negatives": metrics.keyword_false_negatives,
        "confusion_matrix": metrics.confusion_mat.tolist(),
        "classes": dataset_cfg.classes
    }

    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)

    # Save Text Report
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write("============================================================\n")
        f.write("KHUSKHUS-KWS VALIDATION REPORT\n")
        f.write("============================================================\n\n")
        f.write(f"Model Evaluated:            {model_path}\n")
        f.write(f"Total Validation Samples:   {len(all_targets)}\n\n")
        f.write(f"Overall Accuracy:           {metrics.accuracy * 100:.2f}%\n")
        f.write(f"Macro Precision:            {metrics.macro_precision:.4f}\n")
        f.write(f"Macro Recall:               {metrics.macro_recall:.4f}\n")
        f.write(f"Macro F1-Score:             {metrics.macro_f1:.4f}\n\n")
        f.write("Keyword ('khus khus') Specific Metrics:\n")
        f.write(f"  Keyword Precision:        {metrics.keyword_precision:.4f} ({metrics.keyword_precision*100:.2f}%)\n")
        f.write(f"  Keyword Recall:           {metrics.keyword_recall:.4f} ({metrics.keyword_recall*100:.2f}%)\n")
        f.write(f"  Keyword F1-Score:         {metrics.keyword_f1:.4f}\n")
        f.write(f"  False Positives (FP):     {metrics.keyword_false_positives}\n")
        f.write(f"  False Negatives (FN):     {metrics.keyword_false_negatives}\n\n")
        f.write("Confusion Matrix:\n")
        f.write("               Predicted ->\n")
        f.write(f"  Actual       {'keyword':<12} {'other_speech':<14} {'noise':<10}\n")
        for i, row_label in enumerate(dataset_cfg.classes):
            row = metrics.confusion_mat[i]
            f.write(f"  {row_label:<12} {row[0]:<12} {row[1]:<14} {row[2]:<10}\n")
        f.write("============================================================\n")

    # Plot Confusion Matrix
    plot_confusion_matrix(
        metrics.confusion_mat,
        dataset_cfg.classes,
        str(cm_png_path),
        title="KhusKhus-KWS Validation Confusion Matrix"
    )

    print("Validation Results:")
    print(f"  Overall Accuracy:      {metrics.accuracy * 100:.2f}%")
    print(f"  Keyword Precision:     {metrics.keyword_precision:.4f}")
    print(f"  Keyword Recall:        {metrics.keyword_recall:.4f}")
    print(f"  Keyword F1:            {metrics.keyword_f1:.4f}")
    print(f"  Keyword False Pos:     {metrics.keyword_false_positives}")
    print(f"  Keyword False Neg:     {metrics.keyword_false_negatives}")
    print(f"\nArtifacts saved:")
    print(f"  - Report:  {report_txt_path}")
    print(f"  - JSON:    {metrics_json_path}")
    print(f"  - Matrix:  {cm_png_path}")

    return metrics_dict


if __name__ == "__main__":
    validate_kws()
