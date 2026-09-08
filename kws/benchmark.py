"""
Laptop Latency Benchmarking for KhusKhus-KWS
Measures high-resolution stage-by-stage processing times:
1. Audio buffer preparation / window slicing
2. Log-Mel feature extraction
3. Neural network inference (FP32 CPU / GPU)
4. Temporal decision logic
5. Total end-to-end KWS processing cycle

DISCLAIMER: This measures development laptop execution latency for profiling and optimization.
Physical ESP32 hardware latency must be measured separately on the physical microcontroller.
"""

import argparse
import datetime
import json
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple

# Ensure project root is in Python sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import psutil
import torch
import torch.nn as nn

from kws.config import (
    AUDIO_CFG,
    DATASET_CFG,
    DECISION_CFG,
    MODEL_CFG,
    AudioConfig,
    DatasetConfig,
    DecisionConfig,
    ModelConfig,
)
from kws.features import LogMelFeatureExtractor
from kws.metrics import TemporalDecisionEngine
from kws.model import KhusKhusKWS, compute_model_statistics


def calculate_latency_stats(latencies_ms: List[float]) -> Dict[str, float]:
    """Calculate statistical distribution for latency measurements in milliseconds."""
    arr = np.array(latencies_ms)
    return {
        "mean_ms": float(np.mean(arr)),
        "median_p50_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "std_ms": float(np.std(arr)),
    }


def run_benchmark(
    num_iterations: int = 500,
    warmup_runs: int = 50,
    model_path: Optional[Path] = None,
    device_name: str = "cpu",
    dataset_cfg: DatasetConfig = DATASET_CFG,
    model_cfg: ModelConfig = MODEL_CFG,
    audio_cfg: AudioConfig = AUDIO_CFG,
    decision_cfg: DecisionConfig = DECISION_CFG
) -> Dict:
    """Run comprehensive multi-stage latency and resource profiling."""
    device = torch.device(device_name)
    process = psutil.Process(os.getpid())

    # 1. Initialize Pipeline Modules
    feature_extractor = LogMelFeatureExtractor(audio_cfg).to(device)
    feature_extractor.eval()

    model = KhusKhusKWS(model_cfg).to(device)
    model.eval()

    # Load weights if trained model exists
    if model_path is None:
        candidate_paths = [
            dataset_cfg.models_dir / "best_model.pth",
            dataset_cfg.checkpoints_dir / "best_model.pth",
            dataset_cfg.models_dir / "final_model.pth"
        ]
        for p in candidate_paths:
            if p.exists():
                model_path = p
                break

    if model_path and model_path.exists():
        ckpt = torch.load(str(model_path), map_location=device)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
        print(f"[Benchmark] Loaded model weights from: {model_path}")
    else:
        print("[Benchmark] Running with initialized (untrained/fresh) architecture weights.")

    decision_engine = TemporalDecisionEngine(decision_cfg, hop_sec=audio_cfg.hop_sec)
    decision_engine.reset()

    # Pre-generate synthetic 1-second audio buffer (16000 samples)
    raw_audio_buffer = np.random.uniform(-0.8, 0.8, audio_cfg.window_samples).astype(np.float32)

    # 2. Warmup Runs (Exclude from timing)
    print(f"[Benchmark] Warming up pipeline for {warmup_runs} iterations...")
    for _ in range(warmup_runs):
        tensor_audio = torch.from_numpy(raw_audio_buffer).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = feature_extractor(tensor_audio)
            logits = model(feat)
            prob = torch.softmax(logits, dim=1)[0, 0].item()
            _ = decision_engine.process_frame(prob)

    decision_engine.reset()

    # 3. High-Resolution Multi-Stage Measurement
    print(f"[Benchmark] Profiling {num_iterations} consecutive KWS cycles on {device}...")
    stage1_latencies = []  # Audio buffer / tensor prep
    stage2_latencies = []  # Log-Mel feature extraction
    stage3_latencies = []  # Model forward pass
    stage4_latencies = []  # Temporal decision logic
    total_latencies = []   # End-to-end processing

    cpu_usages = []
    ram_usage_start_mb = process.memory_info().rss / (1024.0 * 1024.0)

    for i in range(num_iterations):
        t_total_start = time.perf_counter_ns()

        # Stage 1: Audio buffer preparation
        t0 = time.perf_counter_ns()
        # Simulate ring buffer slice
        audio_slice = raw_audio_buffer[-audio_cfg.window_samples:]
        tensor_audio = torch.from_numpy(audio_slice).unsqueeze(0).to(device)
        t1 = time.perf_counter_ns()
        stage1_latencies.append((t1 - t0) / 1_000_000.0)

        # Stage 2: Feature Extraction (Log-Mel)
        t2 = time.perf_counter_ns()
        with torch.no_grad():
            features = feature_extractor(tensor_audio)
        t3 = time.perf_counter_ns()
        stage2_latencies.append((t3 - t2) / 1_000_000.0)

        # Stage 3: Neural Network Inference
        t4 = time.perf_counter_ns()
        with torch.no_grad():
            logits = model(features)
            kw_prob = torch.softmax(logits, dim=1)[0, 0].item()
        t5 = time.perf_counter_ns()
        stage3_latencies.append((t5 - t4) / 1_000_000.0)

        # Stage 4: Temporal Decision Logic
        t6 = time.perf_counter_ns()
        triggered, _ = decision_engine.process_frame(kw_prob)
        t7 = time.perf_counter_ns()
        stage4_latencies.append((t7 - t6) / 1_000_000.0)

        t_total_end = time.perf_counter_ns()
        total_latencies.append((t_total_end - t_total_start) / 1_000_000.0)

        if i % 100 == 0:
            cpu_usages.append(psutil.cpu_percent(interval=None))

    ram_usage_end_mb = process.memory_info().rss / (1024.0 * 1024.0)
    avg_cpu_percent = float(np.mean(cpu_usages)) if cpu_usages else psutil.cpu_percent(interval=None)

    # Compute Statistics
    stats_stage1 = calculate_latency_stats(stage1_latencies)
    stats_stage2 = calculate_latency_stats(stage2_latencies)
    stats_stage3 = calculate_latency_stats(stage3_latencies)
    stats_stage4 = calculate_latency_stats(stage4_latencies)
    stats_total = calculate_latency_stats(total_latencies)

    model_stats = compute_model_statistics(model, (1, model_cfg.in_channels, 98))

    # Compile Benchmark Results
    benchmark_report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "hardware": {
            "device": str(device),
            "device_name": torch.cuda.get_device_name(0) if "cuda" in str(device) else "CPU (Laptop)",
            "cpu_cores_physical": psutil.cpu_count(logical=False),
            "cpu_cores_logical": psutil.cpu_count(logical=True),
        },
        "model_profile": {
            "name": model_cfg.name,
            "total_parameters": model_stats["total_parameters"],
            "estimated_macs": model_stats["estimated_macs"],
            "fp32_size_kb": model_stats["estimated_fp32_size_kb"],
            "int8_size_kb": model_stats["estimated_int8_size_kb"],
            "max_activation_kb": model_stats["max_activation_tensor_kb"]
        },
        "system_resources": {
            "ram_rss_mb": ram_usage_end_mb,
            "ram_delta_mb": ram_usage_end_mb - ram_usage_start_mb,
            "cpu_utilization_percent": avg_cpu_percent
        },
        "iterations_evaluated": num_iterations,
        "latency_breakdown_ms": {
            "stage1_audio_prep": stats_stage1,
            "stage2_feature_extraction": stats_stage2,
            "stage3_model_inference": stats_stage3,
            "stage4_temporal_decision": stats_stage4,
            "total_end_to_end_cycle": stats_total
        },
        "esp32_disclaimer": (
            "Laptop latency benchmark is strictly for development, bottleneck discovery, and relative model comparison. "
            "Microcontroller execution speeds, flash bandwidth, I2S microphone DMA, and ESP-NN INT8 SIMD performance "
            "must be verified on physical ESP32 hardware."
        )
    }

    # Save outputs
    (dataset_cfg.results_dir / "latency").mkdir(parents=True, exist_ok=True)
    report_json_path = dataset_cfg.results_dir / "latency" / "latency_report.json"
    report_txt_path = dataset_cfg.results_dir / "latency" / "latency_report.txt"

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, indent=2)

    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write("==========================================================================================\n")
        f.write("KHUSKHUS-KWS LAPTOP LATENCY BENCHMARK & SYSTEM PROFILING REPORT\n")
        f.write("==========================================================================================\n\n")
        f.write(f"Timestamp:              {benchmark_report['timestamp']}\n")
        f.write(f"Benchmark Device:       {benchmark_report['hardware']['device_name']}\n")
        f.write(f"Model Architecture:     {model_cfg.name} (Parameters: {model_stats['total_parameters']:,} | MACs: {model_stats['estimated_macs']:,})\n")
        f.write(f"Number of Iterations:   {num_iterations} runs (Warmup: {warmup_runs})\n")
        f.write(f"RAM Usage:              {ram_usage_end_mb:.2f} MB\n")
        f.write(f"CPU Utilization:        {avg_cpu_percent:.1f}%\n\n")
        f.write("Stage-by-Stage Latency Breakdown (Milliseconds):\n")
        f.write(f"{'-'*92}\n")
        f.write(f"{'Processing Stage':<30} | {'Mean (ms)':<10} | {'P50 (ms)':<10} | {'P95 (ms)':<10} | {'Min (ms)':<10} | {'Max (ms)':<10}\n")
        f.write(f"{'-'*92}\n")
        f.write(f"{'1. Audio Window Prep':<30} | {stats_stage1['mean_ms']:<10.4f} | {stats_stage1['median_p50_ms']:<10.4f} | {stats_stage1['p95_ms']:<10.4f} | {stats_stage1['min_ms']:<10.4f} | {stats_stage1['max_ms']:<10.4f}\n")
        f.write(f"{'2. Log-Mel Feature Extract':<30} | {stats_stage2['mean_ms']:<10.4f} | {stats_stage2['median_p50_ms']:<10.4f} | {stats_stage2['p95_ms']:<10.4f} | {stats_stage2['min_ms']:<10.4f} | {stats_stage2['max_ms']:<10.4f}\n")
        f.write(f"{'3. Model Inference (FP32)':<30} | {stats_stage3['mean_ms']:<10.4f} | {stats_stage3['median_p50_ms']:<10.4f} | {stats_stage3['p95_ms']:<10.4f} | {stats_stage3['min_ms']:<10.4f} | {stats_stage3['max_ms']:<10.4f}\n")
        f.write(f"{'4. Temporal Decision':<30} | {stats_stage4['mean_ms']:<10.4f} | {stats_stage4['median_p50_ms']:<10.4f} | {stats_stage4['p95_ms']:<10.4f} | {stats_stage4['min_ms']:<10.4f} | {stats_stage4['max_ms']:<10.4f}\n")
        f.write(f"{'-'*92}\n")
        f.write(f"{'TOTAL END-TO-END CYCLE':<30} | {stats_total['mean_ms']:<10.4f} | {stats_total['median_p50_ms']:<10.4f} | {stats_total['p95_ms']:<10.4f} | {stats_total['min_ms']:<10.4f} | {stats_total['max_ms']:<10.4f}\n")
        f.write(f"{'-'*92}\n\n")
        f.write("ESP32 Deployment Interpretation Notes:\n")
        f.write(f"  {benchmark_report['esp32_disclaimer']}\n")
        f.write("==========================================================================================\n")

    print("\n" + "="*80)
    print("Benchmark Results Summary (Laptop):")
    print("="*80)
    print(f"  Stage 1 (Audio Windowing):   {stats_stage1['mean_ms']:.3f} ms (P95: {stats_stage1['p95_ms']:.3f} ms)")
    print(f"  Stage 2 (Feature Extraction): {stats_stage2['mean_ms']:.3f} ms (P95: {stats_stage2['p95_ms']:.3f} ms)")
    print(f"  Stage 3 (Model Forward Pass): {stats_stage3['mean_ms']:.3f} ms (P95: {stats_stage3['p95_ms']:.3f} ms)")
    print(f"  Stage 4 (Temporal Decision):  {stats_stage4['mean_ms']:.4f} ms (P95: {stats_stage4['p95_ms']:.4f} ms)")
    print(f"  -----------------------------------------------------------------")
    print(f"  TOTAL Processing Latency:    {stats_total['mean_ms']:.3f} ms (P50: {stats_total['median_p50_ms']:.3f} ms, P95: {stats_total['p95_ms']:.3f} ms)")
    print("="*80)
    print(f"Artifacts saved:")
    print(f"  - Report: {report_txt_path}")
    print(f"  - JSON:   {report_json_path}")

    return benchmark_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark KhusKhus-KWS Latency on Laptop")
    parser.add_argument("--iterations", type=int, default=500, help="Number of benchmark iterations")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to benchmark on")
    parser.add_argument("--model-path", type=str, default=None, help="Optional model checkpoint path")
    args = parser.parse_args()

    run_benchmark(
        num_iterations=args.iterations,
        device_name=args.device,
        model_path=Path(args.model_path) if args.model_path else None
    )
