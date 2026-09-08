"""
Training Pipeline for KhusKhus-KWS
Wake Phrase: "khus khus"
Features: Automatic CUDA detection, deterministic seed, AdamW + LR scheduler,
live intra-epoch progress bar with ETA & speed tracking, early stopping,
checkpoint management, and comprehensive metrics logging.
"""

import argparse
import datetime
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Dict, Optional, Tuple

# Ensure project root is in Python sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from kws.config import (
    AUDIO_CFG,
    AUG_CFG,
    DATASET_CFG,
    MODEL_CFG,
    TRAIN_CFG,
    AudioConfig,
    DatasetConfig,
    ModelConfig,
    TrainingConfig,
)
from kws.dataset import create_dataloaders
from kws.metrics import calculate_metrics, plot_training_history
from kws.model import KhusKhusKWS, compute_model_statistics


def set_seed(seed: int = 42):
    """Set deterministic random seeds across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def format_duration(seconds: float) -> str:
    """Format duration in seconds into human-readable string (e.g., 2m14s or 45s)."""
    seconds = max(0.0, float(seconds))
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    elif m > 0:
        return f"{m}m{s:02d}s"
    else:
        return f"{s:02d}s"


def render_progress_bar(
    current: int,
    total: int,
    prefix: str = "",
    suffix: str = "",
    bar_length: int = 22
):
    """Draw dynamic single-line in-place progress bar."""
    if total <= 0:
        return
    fraction = min(1.0, current / total)
    filled_len = int(round(bar_length * fraction))
    bar = "=" * filled_len + (">" if filled_len < bar_length else "") + "." * (bar_length - filled_len - 1)
    if filled_len == bar_length:
        bar = "=" * bar_length

    pct = fraction * 100.0
    status_line = f"\r{prefix} [{bar}] {current:,}/{total:,} ({pct:5.1f}%) {suffix}"
    sys.stdout.write(status_line)
    sys.stdout.flush()


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    scaler: Optional[torch.amp.GradScaler] = None,
    clip_grad: float = 1.0
) -> Tuple[float, float]:
    """Train model for a single epoch with live progress and ETA."""
    model.train()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    total_batches = len(loader)

    epoch_start_time = time.time()
    last_update_time = epoch_start_time

    for batch_idx, (batch_x, batch_y) in enumerate(loader, start=1):
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        batch_size = batch_y.size(0)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast(device_type="cuda" if "cuda" in device.type else "cpu"):
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
            scaler.scale(loss).backward()
            if clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

        total_loss += loss.item() * batch_size
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == batch_y).sum().item()
        total_samples += batch_size

        # Live Progress update (every 15 batches or on the final batch)
        now = time.time()
        if (batch_idx % 15 == 0) or (batch_idx == total_batches) or (now - last_update_time > 0.5):
            elapsed = now - epoch_start_time
            rate = batch_idx / max(0.001, elapsed)
            remaining_batches = total_batches - batch_idx
            eta = remaining_batches / max(0.001, rate)

            cur_loss = total_loss / max(1, total_samples)
            cur_acc = (correct / max(1, total_samples)) * 100.0

            prefix = f"Epoch {epoch:2d}/{total_epochs:2d}"
            suffix = f"| Loss: {cur_loss:.4f} | Acc: {cur_acc:5.1f}% | {rate:4.1f} b/s | ETA: {format_duration(eta)}"
            render_progress_bar(batch_idx, total_batches, prefix=prefix, suffix=suffix)
            last_update_time = now

    sys.stdout.write("\n")
    sys.stdout.flush()

    epoch_loss = total_loss / max(1, total_samples)
    epoch_acc = correct / max(1, total_samples)
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float, float, float]:
    """Evaluate model on validation or test loader with live progress."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []
    total_batches = len(loader)

    eval_start_time = time.time()

    with torch.no_grad():
        for batch_idx, (batch_x, batch_y) in enumerate(loader, start=1):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            batch_size = batch_y.size(0)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            total_loss += loss.item() * batch_size
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(batch_y.cpu().numpy().tolist())
            total_samples += batch_size

            # Update validation progress
            if (batch_idx % 20 == 0) or (batch_idx == total_batches):
                elapsed = time.time() - eval_start_time
                rate = batch_idx / max(0.001, elapsed)
                eta = (total_batches - batch_idx) / max(0.001, rate)
                render_progress_bar(
                    batch_idx,
                    total_batches,
                    prefix="  Validation",
                    suffix=f"| {rate:4.1f} b/s | ETA: {format_duration(eta)}"
                )

    sys.stdout.write("\n")
    sys.stdout.flush()

    epoch_loss = total_loss / max(1, total_samples)
    metrics = calculate_metrics(all_targets, all_preds, keyword_idx=0)

    return (
        epoch_loss,
        metrics.accuracy,
        metrics.keyword_precision,
        metrics.keyword_recall,
        metrics.keyword_f1
    )


def train_kws(
    resume_checkpoint: Optional[str] = None,
    dataset_cfg: DatasetConfig = DATASET_CFG,
    train_cfg: TrainingConfig = TRAIN_CFG,
    model_cfg: ModelConfig = MODEL_CFG,
    audio_cfg: AudioConfig = AUDIO_CFG
):
    """Full KWS training loop with live terminal progress, early stopping, and checkpointing."""
    set_seed(dataset_cfg.random_seed)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"============================================================")
    print(f"KhusKhus-KWS Training Pipeline")
    print(f"============================================================")
    print(f"Device:            {device} ({torch.cuda.get_device_name(0) if device_str == 'cuda' else 'CPU'})")
    print(f"Wake Phrase:       'khus khus'")
    print(f"Batch Size:        {train_cfg.batch_size}")
    print(f"Learning Rate:     {train_cfg.learning_rate}")
    print(f"Max Epochs:        {train_cfg.epochs}")
    print(f"Early Stop Patience: {train_cfg.patience}")
    print(f"============================================================\n")

    # Ensure output directories exist
    dataset_cfg.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    (dataset_cfg.results_dir / "training").mkdir(parents=True, exist_ok=True)

    # 1. Create DataLoaders
    train_loader, val_loader, _ = create_dataloaders(
        dataset_cfg=dataset_cfg,
        train_cfg=train_cfg,
        audio_cfg=audio_cfg,
        aug_cfg=AUG_CFG
    )

    if train_loader is None or len(train_loader.dataset) == 0:
        print("[ERROR] Training dataset is empty. Run scripts/prepare_dataset.bat first!")
        sys.exit(1)

    if val_loader is None or len(val_loader.dataset) == 0:
        print("[ERROR] Validation dataset is empty. Run scripts/prepare_dataset.bat first!")
        sys.exit(1)

    print(f"Dataset summary:")
    print(f"  Training samples:   {len(train_loader.dataset):,} ({len(train_loader):,} batches)")
    print(f"  Validation samples: {len(val_loader.dataset):,} ({len(val_loader):,} batches)")

    # 2. Build Model
    model = KhusKhusKWS(model_cfg).to(device)
    stats = compute_model_statistics(model, (1, model_cfg.in_channels, 98))
    print(f"Model parameters:     {stats['total_parameters']:,} | MACs: {stats['estimated_macs']:,}")

    # 3. Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=train_cfg.epochs,
        eta_min=train_cfg.min_learning_rate
    )
    scaler = torch.amp.GradScaler("cuda") if (train_cfg.mixed_precision and device_str == "cuda") else None

    start_epoch = 1
    best_val_f1 = -1.0
    best_val_loss = float("inf")
    patience_counter = 0

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_kw_precision": [],
        "val_kw_recall": [],
        "val_kw_f1": [],
        "lr": []
    }

    # Resume from checkpoint if specified
    if resume_checkpoint and Path(resume_checkpoint).exists():
        print(f"\n[Resume] Loading checkpoint from: {resume_checkpoint}")
        ckpt = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_f1 = ckpt.get("best_val_f1", -1.0)
        history = ckpt.get("history", history)
        print(f"[Resume] Resuming from epoch {start_epoch} with Best Val F1: {best_val_f1:.4f}")

    print("\n" + "="*88)
    print("Training Started (Live progress will update below for each batch and epoch)")
    print("="*88)

    # 4. Main Training Loop
    total_train_start = time.time()

    for epoch in range(start_epoch, train_cfg.epochs + 1):
        epoch_t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=train_cfg.epochs,
            scaler=scaler,
            clip_grad=train_cfg.gradient_clip
        )

        val_loss, val_acc, kw_p, kw_r, kw_f1 = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device
        )

        epoch_duration = time.time() - epoch_t0
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_kw_precision"].append(kw_p)
        history["val_kw_recall"].append(kw_r)
        history["val_kw_f1"].append(kw_f1)
        history["lr"].append(current_lr)

        # Print Summary Card for completed epoch
        print(
            f"-> [Summary Epoch {epoch:2d}/{train_cfg.epochs:2d}] ({format_duration(epoch_duration)})\n"
            f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:5.2f}%\n"
            f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:5.2f}%\n"
            f"   Keyword F1: {kw_f1:.4f} | Recall: {kw_r:.4f} | Precision: {kw_p:.4f}"
        )

        # Checkpoint paths
        latest_ckpt_path = dataset_cfg.checkpoints_dir / "latest_checkpoint.pth"
        best_model_path = dataset_cfg.checkpoints_dir / "best_model.pth"
        root_best_model_path = dataset_cfg.models_dir / "best_model.pth"

        # Save latest checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_f1": best_val_f1,
            "history": history,
            "config": model_cfg.__dict__
        }, latest_ckpt_path)

        # Check for best model
        is_best = (kw_f1 > best_val_f1) or (abs(kw_f1 - best_val_f1) < 1e-4 and val_loss < best_val_loss)
        if is_best:
            best_val_f1 = kw_f1
            best_val_loss = val_loss
            patience_counter = 0

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "keyword_precision": kw_p,
                "keyword_recall": kw_r,
                "keyword_f1": kw_f1,
                "config": model_cfg.__dict__
            }, best_model_path)
            shutil.copy(best_model_path, root_best_model_path)
            print(f"   ★ NEW BEST MODEL SAVED! (Val Keyword F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            print(f"   (No improvement for {patience_counter}/{train_cfg.patience} epochs)")

        print("-" * 88)

        # Early Stopping
        if patience_counter >= train_cfg.patience:
            print(f"\n[Early Stopping] Triggered after {train_cfg.patience} epochs without improvement.")
            break

    total_train_time = time.time() - total_train_start
    print("=" * 88)
    print(f"[Training Complete] Total Duration: {format_duration(total_train_time)} | Best Val Keyword F1: {best_val_f1:.4f}")

    # Save final model
    final_model_path = dataset_cfg.checkpoints_dir / "final_model.pth"
    root_final_model_path = dataset_cfg.models_dir / "final_model.pth"
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "history": history,
        "config": model_cfg.__dict__
    }, final_model_path)
    shutil.copy(final_model_path, root_final_model_path)

    # Save history and curves
    history_json_path = dataset_cfg.results_dir / "training" / "training_history.json"
    with open(history_json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    curves_path = dataset_cfg.results_dir / "training" / "training_curves.png"
    plot_training_history(history, str(curves_path))

    # Save training report text
    report_path = dataset_cfg.results_dir / "training" / "training_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("============================================================\n")
        f.write("KHUSKHUS-KWS TRAINING REPORT\n")
        f.write("============================================================\n\n")
        f.write(f"Timestamp:             {datetime.datetime.now().isoformat()}\n")
        f.write(f"Device:                {device_str}\n")
        f.write(f"Total Duration:        {format_duration(total_train_time)}\n")
        f.write(f"Total Epochs Trained:  {len(history['train_loss'])}\n")
        f.write(f"Best Val Keyword F1:   {best_val_f1:.4f}\n")
        f.write(f"Final Train Loss:      {history['train_loss'][-1]:.4f}\n")
        f.write(f"Final Val Loss:        {history['val_loss'][-1]:.4f}\n")
        f.write(f"Final Val Accuracy:    {history['val_acc'][-1]*100:.2f}%\n")
        f.write(f"Final Val KW Recall:   {history['val_kw_recall'][-1]:.4f}\n")
        f.write(f"Final Val KW Prec:     {history['val_kw_precision'][-1]:.4f}\n")
        f.write(f"Best Checkpoint:       {root_best_model_path}\n")
        f.write(f"Final Checkpoint:      {root_final_model_path}\n")
        f.write("============================================================\n")

    print(f"Artifacts saved:")
    print(f"  - Checkpoint:  {root_best_model_path}")
    print(f"  - Curves:      {curves_path}")
    print(f"  - Report:      {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train KhusKhus-KWS Model")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs (default: 25)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: 0.001)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    args = parser.parse_args()

    # Override config with CLI arguments if provided
    train_config = TRAIN_CFG
    if args.epochs is not None:
        train_config.epochs = args.epochs
    if args.batch_size is not None:
        train_config.batch_size = args.batch_size
    if args.lr is not None:
        train_config.learning_rate = args.lr

    train_kws(resume_checkpoint=args.resume, train_cfg=train_config)
