"""
KhusKhus-KWS Live Microphone Keyword Spotting & Latency Profiler
Streams audio from laptop microphone (or audio file for testing),
detects wake phrase "khus khus", stops upon detection (or runs continuously),
and outputs detailed results with granular latency.
"""

import argparse
from datetime import datetime
import os
from pathlib import Path
import queue
import sys
import time
from typing import Dict, Optional, Tuple

# Fix Windows console encoding if needed
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure project root is in Python sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
import torch.nn as nn

from kws.config import (
    AUDIO_CFG,
    DATASET_CFG,
    DECISION_CFG,
    MODEL_CFG,
    AudioConfig,
    DecisionConfig,
    ModelConfig,
)
from kws.features import LogMelFeatureExtractor
from kws.metrics import TemporalDecisionEngine
from kws.model import KhusKhusKWS


def list_audio_devices():
    """List all available input audio devices on the system."""
    print("============================================================")
    print("Available Audio Input Devices (Microphones)")
    print("============================================================")
    devices = sd.query_devices()
    default_in = sd.default.device[0]

    input_count = 0
    for idx, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            input_count += 1
            is_default = (idx == default_in)
            tag = " [DEFAULT]" if is_default else ""
            try:
                api_name = sd.query_hostapis(dev["hostapi"])["name"]
            except Exception:
                api_name = "HostAPI"
            print(f"  [{idx:2d}] {dev['name']} (Host: {api_name}){tag}")
            print(f"       Max Inputs: {dev['max_input_channels']}, Default Sample Rate: {int(dev['default_samplerate'])} Hz")

    if input_count == 0:
        print("  [WARNING] No audio input devices detected!")
    print("============================================================\n")


def format_vu_meter(rms: float, max_bars: int = 10) -> str:
    """Generate ASCII VU meter bar based on RMS amplitude."""
    db = 20.0 * np.log10(max(rms, 1e-6))
    # Map dB range [-60dB, 0dB] to [0, max_bars]
    normalized = np.clip((db + 60.0) / 60.0, 0.0, 1.0)
    filled_bars = int(round(normalized * max_bars))
    empty_bars = max_bars - filled_bars
    meter = "#" * filled_bars + "-" * empty_bars
    return f"[{meter}] {db:5.1f} dB"


def preprocess_audio_window(
    audio: np.ndarray,
    target_peak: float = 0.94,
    silence_threshold: float = 0.003
) -> np.ndarray:
    """
    Standardize live streaming window to match the exact training distribution:
    1. Remove DC bias offset.
    2. Adaptive Peak Normalization (scales active speech to target peak ~0.94).
    """
    # 1. DC offset removal
    audio = audio - np.mean(audio)

    # 2. Peak normalization
    peak = np.max(np.abs(audio))
    if peak > silence_threshold:
        scale = target_peak / peak
        audio = audio * scale
    else:
        # Faint background noise/silence: avoid over-amplifying hiss
        audio = audio * 1.0

    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def load_kws_model(
    model_path: Optional[Path] = None,
    device: torch.device = torch.device("cpu"),
    model_cfg: ModelConfig = MODEL_CFG
) -> Tuple[KhusKhusKWS, Path]:
    """Load the trained KhusKhus-KWS neural network weights."""
    if model_path is None:
        candidate_paths = [
            DATASET_CFG.models_dir / "best_model.pth",
            DATASET_CFG.checkpoints_dir / "best_model.pth",
            DATASET_CFG.models_dir / "final_model.pth",
        ]
        for p in candidate_paths:
            if p.exists():
                model_path = p
                break

    if model_path is None or not model_path.exists():
        raise FileNotFoundError(
            f"No trained model checkpoint found in {DATASET_CFG.models_dir}. "
            f"Please run scripts/train_kws.bat first!"
        )

    model = KhusKhusKWS(model_cfg).to(device)
    ckpt = torch.load(str(model_path), map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    return model, model_path


def run_live_kws(
    model_path: Optional[Path] = None,
    device_index: Optional[int] = None,
    test_file: Optional[Path] = None,
    keyword_threshold: float = DECISION_CFG.keyword_threshold,
    consecutive_frames: int = DECISION_CFG.consecutive_frames,
    cooldown_ms: float = DECISION_CFG.cooldown_ms,
    continuous: bool = False,
    timeout_sec: Optional[float] = None,
    save_audio: bool = True,
    save_dir: Path = DATASET_CFG.results_dir / "live_captures",
    device_str: str = "cpu",
    audio_cfg: AudioConfig = AUDIO_CFG,
    model_cfg: ModelConfig = MODEL_CFG,
) -> Optional[Dict]:
    """
    Run real-time microphone keyword spotting with live latency analysis.
    Stops searching when keyword is detected (or runs continuously if configured).
    """
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")

    # 1. Load Model & Feature Extractor
    print("\n" + "=" * 70)
    print("  KHUSKHUS-KWS: REAL-TIME KEYWORD SPOTTING & LATENCY PROFILER")
    print("=" * 70)
    print(f"Wake Phrase:        'khus khus'")
    print(f"Inference Device:   {device}")
    print(f"Sample Rate:        {audio_cfg.sample_rate} Hz (Mono)")
    print(f"Window Duration:    {audio_cfg.duration_sec * 1000.0:.0f} ms ({audio_cfg.window_samples} samples)")
    print(f"Streaming Hop Step: {audio_cfg.hop_sec * 1000.0:.0f} ms ({audio_cfg.hop_samples} samples)")
    print(f"Keyword Threshold:  {keyword_threshold:.2f} ({keyword_threshold * 100:.0f}%)")
    print(f"Consecutive Frames: {consecutive_frames}")
    print(f"Mode:               {'Continuous Monitoring' if continuous else 'Stop on First Detection (One-Shot)'}")

    model, loaded_path = load_kws_model(model_path, device, model_cfg)
    print(f"Loaded Weights:     {loaded_path}")

    feature_extractor = LogMelFeatureExtractor(audio_cfg).to(device)
    feature_extractor.eval()

    decision_cfg = DecisionConfig(
        keyword_threshold=keyword_threshold,
        consecutive_frames=consecutive_frames,
        cooldown_ms=cooldown_ms,
    )
    decision_engine = TemporalDecisionEngine(decision_cfg, hop_sec=audio_cfg.hop_sec)

    # 2. Setup Audio Recording Queue & Buffer
    audio_queue: queue.Queue = queue.Queue()
    window_samples = audio_cfg.window_samples
    hop_samples = audio_cfg.hop_samples
    rolling_buffer = np.zeros(window_samples, dtype=np.float32)

    def audio_callback(indata, frames, time_info, status):
        if status:
            pass  # Buffer overrun or underrun flag
        audio_queue.put(indata[:, 0].copy())

    if test_file:
        print(f"Audio Source:       File Simulation ({test_file})")
        if not Path(test_file).exists():
            raise FileNotFoundError(f"Test audio file not found: {test_file}")
        test_audio, sr = sf.read(str(test_file), dtype="float32")
        if test_audio.ndim > 1:
            test_audio = test_audio.mean(axis=1)
        if sr != audio_cfg.sample_rate:
            import scipy.signal
            num_target_samples = int(len(test_audio) * audio_cfg.sample_rate / sr)
            test_audio = scipy.signal.resample(test_audio, num_target_samples).astype(np.float32)
    else:
        try:
            device_info = sd.query_devices(device_index, "input")
            print(f"Audio Source:       Microphone: {device_info['name']}")
        except Exception as e:
            print(f"[ERROR] Could not open audio device: {e}")
            return None

    if save_audio:
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Capture Output Dir: {save_dir}")

    print("=" * 70)
    if test_file:
        print(">>> STREAMING AUDIO FROM TEST FILE...")
    else:
        print(">>> STARTING LISTENING... Speak 'khus khus' into your microphone.")
        print("    (Press Ctrl+C anytime to stop manually)\n")

    softmax = nn.Softmax(dim=1)
    detection_result = None
    start_time = time.time()
    frames_processed = 0

    # Latency tracking history
    feat_latencies = []
    infer_latencies = []
    dec_latencies = []
    total_latencies = []

    def process_chunk(chunk: np.ndarray) -> Tuple[bool, Optional[Dict]]:
        nonlocal rolling_buffer, frames_processed
        rolling_buffer = np.roll(rolling_buffer, -len(chunk))
        rolling_buffer[-len(chunk):] = chunk
        frames_processed += 1

        if frames_processed < int(0.3 / audio_cfg.hop_sec):
            return False, None

        # Standardize 1.0s audio window to match training distribution
        processed_window = preprocess_audio_window(rolling_buffer)

        # ============================================================
        # STAGE-BY-STAGE LATENCY MEASUREMENT
        # ============================================================
        t_start = time.perf_counter_ns()

        # Stage 1: Feature Extraction (Audio Window -> Log-Mel Spectrogram)
        t0_feat = time.perf_counter_ns()
        audio_tensor = torch.from_numpy(processed_window).unsqueeze(0).to(device)
        with torch.no_grad():
            mel_features = feature_extractor(audio_tensor)
        t1_feat = time.perf_counter_ns()
        feat_ms = (t1_feat - t0_feat) / 1_000_000.0

        # Stage 2: Neural Network Inference (Log-Mel -> Logits -> Probs)
        t0_infer = time.perf_counter_ns()
        with torch.no_grad():
            logits = model(mel_features)
            probs = softmax(logits)[0].cpu().numpy()
        t1_infer = time.perf_counter_ns()
        infer_ms = (t1_infer - t0_infer) / 1_000_000.0

        # Extract class probabilities [0: keyword, 1: other_speech, 2: noise]
        kw_prob = float(probs[0])
        other_prob = float(probs[1])
        noise_prob = float(probs[2])

        # Stage 3: Temporal Decision Engine (Smoothing & Consecutive Verification)
        t0_dec = time.perf_counter_ns()
        triggered, debug_info = decision_engine.process_frame(kw_prob)
        t1_dec = time.perf_counter_ns()
        dec_ms = (t1_dec - t0_dec) / 1_000_000.0

        t_end = time.perf_counter_ns()
        total_ms = (t_end - t_start) / 1_000_000.0

        feat_latencies.append(feat_ms)
        infer_latencies.append(infer_ms)
        dec_latencies.append(dec_ms)
        total_latencies.append(total_ms)

        # Measure raw RMS for live visual meter
        raw_rms = float(np.sqrt(np.mean(rolling_buffer[-hop_samples:] ** 2)))
        vu_str = format_vu_meter(raw_rms, max_bars=8)

        # Real-time console update
        status_symbol = "[ACTIVE]" if kw_prob >= keyword_threshold else "[LISTEN]"
        sys.stdout.write(
            f"\r{status_symbol} Mic: {vu_str} | "
            f"Keyword: {kw_prob * 100:5.1f}% | "
            f"Speech: {other_prob * 100:5.1f}% | "
            f"Noise: {noise_prob * 100:5.1f}% | "
            f"Latency: {total_ms:4.2f}ms "
        )
        sys.stdout.flush()

        # ============================================================
        # KEYWORD TRIGGER DETECTED!
        # ============================================================
        if triggered:
            timestamp_now = datetime.now()
            elapsed_detect = time.time() - start_time
            sys.stdout.write("\n\n")

            # Save triggering audio window if enabled
            saved_audio_path = None
            if save_audio:
                filename = f"detect_{timestamp_now.strftime('%Y%m%d_%H%M%S_%f')[:-3]}.wav"
                saved_audio_path = save_dir / filename
                sf.write(str(saved_audio_path), rolling_buffer, audio_cfg.sample_rate)

            # Real-time Factor (RTF) = processing time / audio chunk step
            hop_ms = audio_cfg.hop_sec * 1000.0
            rtf = total_ms / hop_ms

            res = {
                "timestamp": timestamp_now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                "elapsed_seconds": elapsed_detect,
                "probabilities": {
                    "keyword": kw_prob,
                    "other_speech": other_prob,
                    "noise": noise_prob,
                },
                "prediction": "keyword",
                "wake_phrase": "khus khus",
                "latency_breakdown_ms": {
                    "feature_extraction_ms": feat_ms,
                    "neural_network_inference_ms": infer_ms,
                    "temporal_decision_ms": dec_ms,
                    "total_pipeline_latency_ms": total_ms,
                },
                "latency_stats_ms": {
                    "mean_total_ms": float(np.mean(total_latencies[-20:])),
                    "p95_total_ms": float(np.percentile(total_latencies[-20:], 95)),
                },
                "audio_window_ms": audio_cfg.duration_sec * 1000.0,
                "hop_step_ms": hop_ms,
                "real_time_factor_rtf": rtf,
                "saved_audio_path": str(saved_audio_path) if saved_audio_path else None,
            }

            # Print formatted detection banner
            print("======================================================================")
            print("[TARGET DETECTED] KEYWORD TRIGGERED! Wake Phrase: \"khus khus\"")
            print("======================================================================")
            print(f"Timestamp:              {res['timestamp']}")
            print(f"Search Time to Detect:  {elapsed_detect:.2f} seconds")
            print(f"Consecutive Hits:       {consecutive_frames} frames confirmed")
            print("-" * 70)
            print("Classification Probabilities:")
            print(f"  [+] Keyword ('khus khus'):  {kw_prob * 100:6.2f}%  [TRIGGERED]")
            print(f"  [-] Other Speech:           {other_prob * 100:6.2f}%")
            print(f"  [-] Background Noise:       {noise_prob * 100:6.2f}%")
            print("-" * 70)
            print("Latency Breakdown (This Frame):")
            print(f"  * Feature Extraction (Log-Mel):   {feat_ms:6.3f} ms")
            print(f"  * Neural Network Inference:       {infer_ms:6.3f} ms")
            print(f"  * Temporal Decision Logic:        {dec_ms:6.3f} ms ({dec_ms * 1000.0:.1f} us)")
            print(f"  * Total End-to-End Latency:       {total_ms:6.3f} ms")
            print("-" * 70)
            print("Streaming Performance:")
            print(f"  * Audio Analysis Window:          {audio_cfg.duration_sec * 1000.0:.0f} ms")
            print(f"  * Streaming Hop Step:             {hop_ms:.1f} ms")
            print(f"  * Real-Time Factor (RTF):         {rtf:.4f}x (1.0x is real-time limit, <0.1x is ultra-fast)")
            if saved_audio_path:
                print(f"  * Audio Clip Saved:               {saved_audio_path}")
            print("======================================================================\n")

            return True, res

        return False, None

    try:
        if test_file:
            # Simulated real-time file stream
            num_chunks = int(np.ceil(len(test_audio) / hop_samples))
            for i in range(num_chunks):
                chunk = test_audio[i * hop_samples : (i + 1) * hop_samples]
                if len(chunk) < hop_samples:
                    chunk = np.pad(chunk, (0, hop_samples - len(chunk)))
                
                trig, res = process_chunk(chunk)
                if trig:
                    detection_result = res
                    if not continuous:
                        print("[SUCCESS] Keyword detected! Stopping search as requested.\n")
                        break
                time.sleep(audio_cfg.hop_sec * 0.2)  # Fast simulated streaming
        else:
            # Real-time microphone input stream
            with sd.InputStream(
                samplerate=audio_cfg.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=hop_samples,
                device=device_index,
                callback=audio_callback,
            ):
                while True:
                    # Check timeout
                    elapsed_total = time.time() - start_time
                    if timeout_sec and elapsed_total >= timeout_sec:
                        print(f"\n[INFO] Timeout reached ({timeout_sec:.1f}s). Stopping search.")
                        break

                    # Get new audio chunk from queue
                    try:
                        chunk = audio_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    trig, res = process_chunk(chunk)
                    if trig:
                        detection_result = res
                        if not continuous:
                            print("[SUCCESS] Keyword detected! Stopping search as requested.\n")
                            break
                        else:
                            print("[INFO] Continuing real-time monitoring...\n")

    except KeyboardInterrupt:
        print("\n\n[INFO] Live microphone monitoring stopped by user (Ctrl+C).")

    return detection_result


def main():
    parser = argparse.ArgumentParser(
        description="KhusKhus-KWS Real-Time Microphone Keyword Detection & Latency Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--list_devices",
        action="store_true",
        help="List all available audio input devices and exit",
    )
    parser.add_argument(
        "--device_index",
        type=int,
        default=None,
        help="Audio input device index (default: system default microphone)",
    )
    parser.add_argument(
        "--test_file",
        type=Path,
        default=None,
        help="Simulate streaming from a recorded audio file instead of live microphone",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=None,
        help="Path to trained PyTorch model .pth weights",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DECISION_CFG.keyword_threshold,
        help="Keyword confidence threshold [0.0 - 1.0]",
    )
    parser.add_argument(
        "--consecutive",
        type=int,
        default=DECISION_CFG.consecutive_frames,
        help="Number of consecutive frames required for trigger",
    )
    parser.add_argument(
        "--cooldown_ms",
        type=float,
        default=DECISION_CFG.cooldown_ms,
        help="Cooldown refractory period in ms after trigger",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Keep listening continuously after detection (default: stop on first detection)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Maximum search timeout in seconds (default: unlimited until detection or Ctrl+C)",
    )
    parser.add_argument(
        "--no_save_audio",
        action="store_true",
        help="Disable saving the triggering audio snippet to disk",
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=DATASET_CFG.results_dir / "live_captures",
        help="Directory to save triggered audio recordings",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Inference compute device",
    )

    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        return

    run_live_kws(
        model_path=args.model_path,
        device_index=args.device_index,
        test_file=args.test_file,
        keyword_threshold=args.threshold,
        consecutive_frames=args.consecutive,
        cooldown_ms=args.cooldown_ms,
        continuous=args.continuous,
        timeout_sec=args.timeout,
        save_audio=not args.no_save_audio,
        save_dir=args.save_dir,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
