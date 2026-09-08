"""
TFLite and TensorFlow Inference Testing and Parity Benchmark
Evaluates and benchmarks PyTorch vs TensorFlow Keras vs TFLite FP32 vs TFLite INT8
on keyword spotting audio features.
"""

import argparse
import datetime
import json
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple

# Ensure project root is in Python sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import tensorflow as tf
import torch

from kws.config import AUDIO_CFG, DATASET_CFG, MODEL_CFG
from kws.dataset import create_dataloaders
from kws.model import KhusKhusKWS


class TFLiteKWSInference:
    """Helper wrapper for running inference with a TFLite flatbuffer model."""

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_shape = self.input_details[0]["shape"]
        self.input_dtype = self.input_details[0]["dtype"]
        self.output_dtype = self.output_details[0]["dtype"]

        # Quantization parameters (scale, zero_point)
        self.in_scale, self.in_zp = self.input_details[0].get("quantization", (0.0, 0))
        self.out_scale, self.out_zp = self.output_details[0].get("quantization", (0.0, 0))
        self.is_quantized = self.input_dtype in (np.int8, np.uint8)

    def predict(self, x_mel_tf: np.ndarray) -> np.ndarray:
        """
        Runs inference on input batch of shape (Batch, 98, 40) or single (1, 98, 40).
        Returns logits of shape (Batch, 3).
        """
        if x_mel_tf.ndim == 2:
            x_mel_tf = np.expand_dims(x_mel_tf, axis=0)

        batch_size = x_mel_tf.shape[0]
        logits_list = []

        for i in range(batch_size):
            sample = x_mel_tf[i:i + 1]

            if self.is_quantized:
                # Quantize float32 -> int8: q = clamp(round(x / scale) + zero_point)
                if self.in_scale > 0:
                    sample_quant = np.clip(
                        np.round(sample / self.in_scale) + self.in_zp,
                        -128, 127
                    ).astype(np.int8)
                else:
                    sample_quant = sample.astype(np.int8)

                self.interpreter.set_tensor(self.input_details[0]["index"], sample_quant)
                self.interpreter.invoke()
                raw_out = self.interpreter.get_tensor(self.output_details[0]["index"])

                # Dequantize int8 -> float32: x = (q - zero_point) * scale
                if self.out_scale > 0:
                    out_float = (raw_out.astype(np.float32) - self.out_zp) * self.out_scale
                else:
                    out_float = raw_out.astype(np.float32)
                logits_list.append(out_float[0])
            else:
                sample_float = sample.astype(np.float32)
                self.interpreter.set_tensor(self.input_details[0]["index"], sample_float)
                self.interpreter.invoke()
                out_float = self.interpreter.get_tensor(self.output_details[0]["index"])
                logits_list.append(out_float[0])

        return np.array(logits_list)


def run_full_tflite_test(
    models_dir: Path = DATASET_CFG.models_dir,
    num_benchmark_runs: int = 100
) -> Dict:
    """
    Evaluates and benchmarks all available runtime formats:
      1. PyTorch FP32
      2. TensorFlow Keras
      3. TFLite FP32
      4. TFLite INT8
    """
    pt_path = models_dir / "best_model.pth"
    keras_path = models_dir / "kws_model.keras"
    tflite_fp32_path = models_dir / "kws_model_fp32.tflite"
    tflite_int8_path = models_dir / "kws_model_int8.tflite"

    print("==================================================================")
    print("  KhusKhus-KWS : Comprehensive Multi-Runtime Verification")
    print("  Models Directory: ", models_dir)
    print("==================================================================")

    # 1. Load PyTorch model
    print("\n[Loading] PyTorch model...")
    pt_model = KhusKhusKWS()
    ckpt = torch.load(str(pt_path), map_location="cpu")
    pt_model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    pt_model.eval()

    # 2. Load Keras model
    print("[Loading] TensorFlow Keras model...")
    tf_model = tf.keras.models.load_model(str(keras_path))

    # 3. Load TFLite models
    print("[Loading] TFLite FP32 & INT8 models...")
    tflite_fp32 = TFLiteKWSInference(tflite_fp32_path)
    tflite_int8 = TFLiteKWSInference(tflite_int8_path)

    # Prepare Test Data
    _, _, test_loader = create_dataloaders(
        dataset_cfg=DATASET_CFG,
        audio_cfg=AUDIO_CFG
    )

    labels = ["keyword ('khus khus')", "other_speech", "noise"]

    if test_loader is not None and len(test_loader.dataset) > 0:
        print(f"[Dataset] Evaluating on {len(test_loader.dataset)} real test samples...")
        all_x, all_y = [], []
        for bx, by in test_loader:
            all_x.append(bx.numpy())
            all_y.append(by.numpy())
        test_x_pt = np.concatenate(all_x, axis=0)  # (N, 40, 98)
        test_y = np.concatenate(all_y, axis=0)
    else:
        print("[Dataset] Test dataset directory not populated; generating synthetic speech test batch...")
        np.random.seed(42)
        test_x_pt = np.random.randn(20, 40, 98).astype(np.float32)
        test_y = np.random.randint(0, 3, size=(20,))

    # Convert to TF layout: (N, 98, 40)
    test_x_tf = np.transpose(test_x_pt, (0, 2, 1))

    # 4. Predictions & Probability Differences
    print("\n[Inference] Running predictions across runtimes...")
    with torch.no_grad():
        out_pt = pt_model(torch.from_numpy(test_x_pt)).numpy()
    out_tf = tf_model(test_x_tf, training=False).numpy()
    out_tflite_fp32 = tflite_fp32.predict(test_x_tf)
    out_tflite_int8 = tflite_int8.predict(test_x_tf)

    probs_pt = torch.softmax(torch.from_numpy(out_pt), dim=-1).numpy()
    probs_tf = tf.nn.softmax(out_tf).numpy()
    probs_fp32 = tf.nn.softmax(out_tflite_fp32).numpy()
    probs_int8 = tf.nn.softmax(out_tflite_int8).numpy()

    diff_tf = float(np.max(np.abs(probs_pt - probs_tf)))
    diff_fp32 = float(np.max(np.abs(probs_pt - probs_fp32)))
    diff_int8 = float(np.max(np.abs(probs_pt - probs_int8)))

    # Classification Agreement
    pred_pt = np.argmax(probs_pt, axis=1)
    pred_tf = np.argmax(probs_tf, axis=1)
    pred_fp32 = np.argmax(probs_fp32, axis=1)
    pred_int8 = np.argmax(probs_int8, axis=1)

    agree_tf = float(np.mean(pred_pt == pred_tf) * 100.0)
    agree_fp32 = float(np.mean(pred_pt == pred_fp32) * 100.0)
    agree_int8 = float(np.mean(pred_pt == pred_int8) * 100.0)

    # 5. Latency Benchmarks
    print(f"[Benchmark] Measuring single-sample inference latency ({num_benchmark_runs} iterations)...")
    single_pt = torch.from_numpy(test_x_pt[0:1])
    single_tf = test_x_tf[0:1]

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = pt_model(single_pt)
        _ = tf_model(single_tf, training=False)
        _ = tflite_fp32.predict(single_tf)
        _ = tflite_int8.predict(single_tf)

    # PyTorch timing
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_benchmark_runs):
            _ = pt_model(single_pt)
    lat_pt = (time.perf_counter() - t0) / num_benchmark_runs * 1000.0

    # TF timing
    t0 = time.perf_counter()
    for _ in range(num_benchmark_runs):
        _ = tf_model(single_tf, training=False)
    lat_tf = (time.perf_counter() - t0) / num_benchmark_runs * 1000.0

    # TFLite FP32 timing
    t0 = time.perf_counter()
    for _ in range(num_benchmark_runs):
        _ = tflite_fp32.predict(single_tf)
    lat_fp32 = (time.perf_counter() - t0) / num_benchmark_runs * 1000.0

    # TFLite INT8 timing
    t0 = time.perf_counter()
    for _ in range(num_benchmark_runs):
        _ = tflite_int8.predict(single_tf)
    lat_int8 = (time.perf_counter() - t0) / num_benchmark_runs * 1000.0

    # Sizes
    size_pt = pt_path.stat().st_size / 1024.0
    size_keras = keras_path.stat().st_size / 1024.0
    size_fp32 = tflite_fp32_path.stat().st_size / 1024.0
    size_int8 = tflite_int8_path.stat().st_size / 1024.0

    # Report
    print("\n" + "=" * 80)
    print("  RUNTIME COMPARISON & BENCHMARK RESULTS")
    print("=" * 80)
    print(f"  {'Runtime':<20} | {'File Size':<10} | {'Latency (ms)':<13} | {'Max Prob Diff':<14} | {'Agreement':<10}")
    print("-" * 80)
    print(f"  {'PyTorch (.pth)':<20} | {size_pt:7.2f} KB | {lat_pt:10.3f} ms | {'Reference':<14} | {'100.0%':<10}")
    print(f"  {'TF Keras (.keras)':<20} | {size_keras:7.2f} KB | {lat_tf:10.3f} ms | {diff_tf:14.2e} | {agree_tf:8.1f} %")
    print(f"  {'TFLite FP32':<20} | {size_fp32:7.2f} KB | {lat_fp32:10.3f} ms | {diff_fp32:14.2e} | {agree_fp32:8.1f} %")
    print(f"  {'TFLite INT8 (ESP32)':<20} | {size_int8:7.2f} KB | {lat_int8:10.3f} ms | {diff_int8:14.2e} | {agree_int8:8.1f} %")
    print("=" * 80)

    # Sample Predictions Display
    print("\n[Sample Output Comparison (Sample #0)]:")
    print(f"  {'Class':<22} | {'PyTorch':<10} | {'TF Keras':<10} | {'TFLite FP32':<12} | {'TFLite INT8':<12}")
    print("-" * 75)
    for c_idx, c_name in enumerate(labels):
        print(f"  {c_name:<22} | {probs_pt[0, c_idx]*100:8.2f}% | {probs_tf[0, c_idx]*100:8.2f}% | {probs_fp32[0, c_idx]*100:10.2f}% | {probs_int8[0, c_idx]*100:10.2f}%")
    print("=" * 75)

    results_dir = DATASET_CFG.results_dir / "testing"
    results_dir.mkdir(parents=True, exist_ok=True)
    report_file = results_dir / "tflite_evaluation_report.json"
    report_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "num_samples_evaluated": int(test_x_pt.shape[0]),
        "agreement": {
            "keras_vs_pytorch": agree_tf,
            "tflite_fp32_vs_pytorch": agree_fp32,
            "tflite_int8_vs_pytorch": agree_int8,
        },
        "max_prob_difference": {
            "keras": diff_tf,
            "tflite_fp32": diff_fp32,
            "tflite_int8": diff_int8,
        },
        "latency_ms": {
            "pytorch": lat_pt,
            "keras": lat_tf,
            "tflite_fp32": lat_fp32,
            "tflite_int8": lat_int8,
        },
        "file_size_kb": {
            "pytorch": size_pt,
            "keras": size_keras,
            "tflite_fp32": size_fp32,
            "tflite_int8": size_int8,
        }
    }
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"\nReport written to: {report_file}")
    return report_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test TFLite Models for KhusKhus-KWS")
    parser.add_argument("--models_dir", type=str, default="models/KhusKhus-KWS")
    parser.add_argument("--benchmark_runs", type=int, default=100)
    args = parser.parse_args()

    run_full_tflite_test(models_dir=Path(args.models_dir), num_benchmark_runs=args.benchmark_runs)
