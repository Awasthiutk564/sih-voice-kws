"""
Dataset Analysis, Standardization, and Split Pipeline for KhusKhus-KWS
Processes original read-only dataset and prepares 70/15/15 train/val/test splits.
Target sample rate: 16000 Hz, Mono, 16-bit PCM WAV.
"""

import json
import os
from pathlib import Path
import random
import re
import sys
from typing import Dict, List, Optional, Tuple

# Ensure project root is in Python sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import soundfile as sf
import torchaudio
import torch

from kws.config import AUDIO_CFG, DATASET_CFG, AudioConfig, DatasetConfig


def find_audio_files(directory: Path, supported_exts: Tuple[str, ...]) -> List[Path]:
    """Recursively discover all audio files in a directory."""
    audio_files = []
    if not directory.exists():
        return audio_files
    for root, _, files in os.walk(directory):
        for file in files:
            p = Path(root) / file
            if p.suffix.lower() in supported_exts:
                audio_files.append(p)
    return sorted(audio_files)


def extract_speaker_id(filepath: Path) -> Optional[str]:
    """
    Attempt to extract speaker ID from filename or parent folder naming conventions.
    E.g., 'speaker_01_word.wav', 'spk12_khuskhus_01.wav', 'user_abc_...'.
    Returns None if no standard speaker ID pattern is detected.
    """
    stem = filepath.stem.lower()
    # Patterns like spk_01, speaker01, user_123, s01_
    match = re.search(r"(?:spk|speaker|user|s)[-_]?([a-zA-Z0-9]+)", stem)
    if match:
        return match.group(1)
    # Check parent folder name if it represents a speaker
    parent_name = filepath.parent.name.lower()
    if parent_name not in ["keyword", "noise", "noice", "other_speech", "dataset"]:
        match_parent = re.search(r"(?:spk|speaker|user|s)[-_]?([a-zA-Z0-9]+)", parent_name)
        if match_parent:
            return match_parent.group(1)
    return None


def inspect_dataset(
    dataset_root: Path,
    supported_exts: Tuple[str, ...] = DATASET_CFG.supported_extensions
) -> Dict:
    """
    Analyze the original read-only dataset without modifying anything.
    Gathers detailed statistics on counts, sample rates, channels, durations, and errors.
    """
    print(f"[Dataset Analysis] Inspecting raw dataset at: {dataset_root}")
    
    # Class folder mappings (including alias handling like 'noice' -> 'noise')
    class_folder_map = {
        "keyword": ["keyword"],
        "noise": ["noise", "noice"],
        "other_speech": ["other_speech", "speech", "unknown"]
    }

    report = {
        "dataset_root": str(dataset_root),
        "total_files": 0,
        "classes": {},
        "file_extensions": {},
        "durations_sec": {"min": float("inf"), "max": 0.0, "avg": 0.0, "total": 0.0},
        "sample_rates": {},
        "channel_counts": {},
        "corrupted_files": [],
        "unreadable_files": [],
        "speaker_info": {
            "speaker_detected": False,
            "unique_speakers_count": 0,
            "speakers_per_class": {},
            "limitation_note": ""
        }
    }

    all_durations = []
    all_speakers = set()

    for standard_class, aliases in class_folder_map.items():
        class_files = []
        found_dir = None
        for alias in aliases:
            candidate = dataset_root / alias
            if candidate.exists() and candidate.is_dir():
                found_dir = candidate
                class_files = find_audio_files(candidate, supported_exts)
                break

        class_stat = {
            "folder_used": str(found_dir) if found_dir else "NOT_FOUND",
            "file_count": len(class_files),
            "extensions": {},
            "durations": [],
            "speakers": set()
        }

        for fpath in class_files:
            ext = fpath.suffix.lower()
            report["file_extensions"][ext] = report["file_extensions"].get(ext, 0) + 1
            class_stat["extensions"][ext] = class_stat["extensions"].get(ext, 0) + 1

            # Speaker extraction check
            spk = extract_speaker_id(fpath)
            if spk:
                class_stat["speakers"].add(spk)
                all_speakers.add(spk)

            # Read file audio metadata
            try:
                info = sf.info(str(fpath))
                sr = info.samplerate
                channels = info.channels
                dur = info.duration

                report["sample_rates"][str(sr)] = report["sample_rates"].get(str(sr), 0) + 1
                report["channel_counts"][str(channels)] = report["channel_counts"].get(str(channels), 0) + 1
                
                class_stat["durations"].append(dur)
                all_durations.append(dur)
            except Exception as e:
                # Fallback to torchaudio info if soundfile fails
                try:
                    info = torchaudio.info(str(fpath))
                    sr = info.sample_rate
                    channels = info.num_channels
                    dur = info.num_frames / sr

                    report["sample_rates"][str(sr)] = report["sample_rates"].get(str(sr), 0) + 1
                    report["channel_counts"][str(channels)] = report["channel_counts"].get(str(channels), 0) + 1

                    class_stat["durations"].append(dur)
                    all_durations.append(dur)
                except Exception as ex_err:
                    report["corrupted_files"].append(str(fpath))
                    report["unreadable_files"].append(f"{fpath}: {str(ex_err)}")

        class_durations = class_stat.pop("durations")
        class_stat["duration_min_sec"] = float(np.min(class_durations)) if class_durations else 0.0
        class_stat["duration_max_sec"] = float(np.max(class_durations)) if class_durations else 0.0
        class_stat["duration_avg_sec"] = float(np.mean(class_durations)) if class_durations else 0.0
        class_stat["duration_total_sec"] = float(np.sum(class_durations)) if class_durations else 0.0
        class_stat["speakers_count"] = len(class_stat["speakers"])
        class_stat["speakers"] = sorted(list(class_stat["speakers"]))

        report["classes"][standard_class] = class_stat
        report["total_files"] += len(class_files)

    if all_durations:
        report["durations_sec"]["min"] = float(np.min(all_durations))
        report["durations_sec"]["max"] = float(np.max(all_durations))
        report["durations_sec"]["avg"] = float(np.mean(all_durations))
        report["durations_sec"]["total"] = float(np.sum(all_durations))
    else:
        report["durations_sec"]["min"] = 0.0

    if len(all_speakers) > 0:
        report["speaker_info"]["speaker_detected"] = True
        report["speaker_info"]["unique_speakers_count"] = len(all_speakers)
        report["speaker_info"]["limitation_note"] = "Speaker IDs detected from file/directory metadata. Speaker-independent splitting will be applied."
    else:
        report["speaker_info"]["speaker_detected"] = False
        report["speaker_info"]["unique_speakers_count"] = 0
        report["speaker_info"]["limitation_note"] = (
            "Speaker identity could not be deterministically resolved from filenames or folder structures. "
            "A deterministic, stratified random split with fixed seed is used to prevent data leakage and maintain class balance."
        )

    return report


def load_and_standardize_audio(
    filepath: Path,
    target_sr: int = AUDIO_CFG.sample_rate
) -> np.ndarray:
    """
    Load an audio file, convert to mono, resample to target_sr, and normalize.
    Returns 1D float32 numpy array in range [-1.0, 1.0].
    """
    try:
        data, sr = sf.read(str(filepath), dtype="float32", always_2d=True)
        # Convert stereo/multi-channel to mono by averaging channels
        if data.shape[1] > 1:
            audio = np.mean(data, axis=1)
        else:
            audio = data[:, 0]
    except Exception:
        # Fallback to torchaudio
        waveform, sr = torchaudio.load(str(filepath))
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        audio = waveform.squeeze(0).numpy()

    # Resample if needed
    if sr != target_sr:
        tensor_audio = torch.from_numpy(audio).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        tensor_audio = resampler(tensor_audio)
        audio = tensor_audio.squeeze(0).numpy()

    # Remove DC offset
    audio = audio - np.mean(audio)

    # Peak normalization with safety headroom (-0.5 dB ~= 0.94)
    peak = np.max(np.abs(audio))
    if peak > 1e-6:
        audio = audio * (0.94 / peak)

    return audio.astype(np.float32)


def split_dataset(
    file_list: List[Path],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Deterministic reproducible split.
    Checks for speaker IDs to avoid cross-split contamination if possible.
    """
    rng = random.Random(seed)
    
    # Group by speaker if detectable
    speaker_groups: Dict[str, List[Path]] = {}
    ungrouped: List[Path] = []

    for fpath in file_list:
        spk = extract_speaker_id(fpath)
        if spk:
            speaker_groups.setdefault(spk, []).append(fpath)
        else:
            ungrouped.append(fpath)

    train_files, val_files, test_files = [], [], []

    if speaker_groups:
        speakers = sorted(list(speaker_groups.keys()))
        rng.shuffle(speakers)
        n_spk = len(speakers)
        n_train = max(1, int(n_spk * train_ratio))
        n_val = max(1, int(n_spk * val_ratio))

        train_spks = set(speakers[:n_train])
        val_spks = set(speakers[n_train:n_train + n_val])
        test_spks = set(speakers[n_train + n_val:])

        for spk, files in speaker_groups.items():
            if spk in train_spks:
                train_files.extend(files)
            elif spk in val_spks:
                val_files.extend(files)
            else:
                test_files.extend(files)

    # Process ungrouped files deterministically
    if ungrouped:
        ungrouped_sorted = sorted(ungrouped)
        rng.shuffle(ungrouped_sorted)
        n = len(ungrouped_sorted)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_files.extend(ungrouped_sorted[:n_train])
        val_files.extend(ungrouped_sorted[n_train:n_train + n_val])
        test_files.extend(ungrouped_sorted[n_train + n_val:])

    return train_files, val_files, test_files


def prepare_and_standardize_dataset(
    dataset_cfg: DatasetConfig = DATASET_CFG,
    audio_cfg: AudioConfig = AUDIO_CFG
):
    """
    Full preparation pipeline:
    1. Inspect raw dataset (READ-ONLY).
    2. Write reports to results/dataset/.
    3. Standardize and write copies to processed_dataset/{train, validation, test}/{class}/.
    """
    report = inspect_dataset(dataset_cfg.dataset_root, dataset_cfg.supported_extensions)

    # Create results directory
    dataset_cfg.results_dir.mkdir(parents=True, exist_ok=True)
    (dataset_cfg.results_dir / "dataset").mkdir(parents=True, exist_ok=True)

    # Save JSON report
    json_path = dataset_cfg.results_dir / "dataset" / "dataset_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Save formatted text report
    txt_path = dataset_cfg.results_dir / "dataset" / "dataset_report.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("============================================================\n")
        f.write("KHUSKHUS-KWS DATASET ANALYSIS REPORT\n")
        f.write("============================================================\n\n")
        f.write(f"Source Dataset: {report['dataset_root']}\n")
        f.write(f"Total Audio Files Found: {report['total_files']}\n")
        f.write(f"File Extensions: {report['file_extensions']}\n")
        f.write(f"Sample Rates Detected: {report['sample_rates']}\n")
        f.write(f"Channel Distributions: {report['channel_counts']}\n\n")
        f.write("Duration Statistics (All Audio):\n")
        f.write(f"  Min Duration:   {report['durations_sec']['min']:.2f} s\n")
        f.write(f"  Max Duration:   {report['durations_sec']['max']:.2f} s\n")
        f.write(f"  Avg Duration:   {report['durations_sec']['avg']:.2f} s\n")
        f.write(f"  Total Duration: {report['durations_sec']['total']:.2f} s ({report['durations_sec']['total']/60:.2f} min)\n\n")
        f.write("Class Breakdown:\n")
        for cls_name, cls_stat in report["classes"].items():
            f.write(f"  [{cls_name}]\n")
            f.write(f"    Folder Used:    {cls_stat['folder_used']}\n")
            f.write(f"    File Count:     {cls_stat['file_count']}\n")
            f.write(f"    Duration (Avg): {cls_stat['duration_avg_sec']:.2f} s (Min: {cls_stat['duration_min_sec']:.2f} s, Max: {cls_stat['duration_max_sec']:.2f} s)\n")
            f.write(f"    Total Duration: {cls_stat['duration_total_sec']:.2f} s\n")
            f.write(f"    Extensions:     {cls_stat['extensions']}\n\n")
        f.write("Speaker Independence Analysis:\n")
        f.write(f"  Speaker IDs Detected: {report['speaker_info']['speaker_detected']}\n")
        f.write(f"  Unique Speakers:      {report['speaker_info']['unique_speakers_count']}\n")
        f.write(f"  Notes: {report['speaker_info']['limitation_note']}\n\n")
        f.write("Audio Standardization Target:\n")
        f.write(f"  Sample Rate: {audio_cfg.sample_rate} Hz\n")
        f.write(f"  Channels:    {audio_cfg.channels} (Mono)\n")
        f.write(f"  Encoding:    PCM 16-bit WAV\n")
        f.write(f"  Splits:      70% Train / 15% Validation / 15% Test (Deterministic Seed: {dataset_cfg.random_seed})\n")
        f.write("============================================================\n")

    print(f"[Dataset Analysis] Report generated:\n  - {txt_path}\n  - {json_path}")

    # Process and write standardized splits
    print(f"\n[Standardization] Preparing processed dataset in: {dataset_cfg.processed_root}")
    class_folder_map = {
        "keyword": ["keyword"],
        "noise": ["noise", "noice"],
        "other_speech": ["other_speech", "speech", "unknown"]
    }

    split_counts = {"train": {}, "validation": {}, "test": {}}

    for standard_class, aliases in class_folder_map.items():
        class_files = []
        for alias in aliases:
            candidate = dataset_cfg.dataset_root / alias
            if candidate.exists() and candidate.is_dir():
                class_files = find_audio_files(candidate, dataset_cfg.supported_extensions)
                break

        train_files, val_files, test_files = split_dataset(
            class_files,
            train_ratio=dataset_cfg.train_ratio,
            val_ratio=dataset_cfg.val_ratio,
            test_ratio=dataset_cfg.test_ratio,
            seed=dataset_cfg.random_seed
        )

        splits_map = {
            "train": train_files,
            "validation": val_files,
            "test": test_files
        }

        for split_name, files in splits_map.items():
            dest_dir = dataset_cfg.processed_root / split_name / standard_class
            dest_dir.mkdir(parents=True, exist_ok=True)
            split_counts[split_name][standard_class] = len(files)

            for idx, src_file in enumerate(files):
                try:
                    audio_mono = load_and_standardize_audio(src_file, audio_cfg.sample_rate)
                    out_filename = f"{standard_class}_{idx:05d}_{src_file.stem}.wav"
                    out_path = dest_dir / out_filename

                    # Write standardized PCM 16-bit WAV
                    sf.write(str(out_path), audio_mono, audio_cfg.sample_rate, subtype="PCM_16")
                except Exception as err:
                    print(f"  [Warning] Failed to standardize {src_file}: {err}")

    print("\n[Standardization Complete] Summary of splits created:")
    for split_name, counts in split_counts.items():
        print(f"  Split '{split_name}': {counts} (Total: {sum(counts.values())})")


if __name__ == "__main__":
    prepare_and_standardize_dataset()
