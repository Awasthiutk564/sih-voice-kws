"""
KhusKhus-KWS Architecture
Custom Ultra-Low-Latency 1D Depthwise-Separable CNN for ESP32 Keyword Spotting.
Optimized for wake phrase: "khus khus".
"""

import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

# Ensure project root is in Python sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from kws.config import MODEL_CFG, ModelConfig


class DepthwiseSeparableConv1d(nn.Module):
    """
    1D Depthwise-Separable Convolution Block.
    Performs depthwise convolution (groups=in_channels) followed by pointwise 1x1 convolution.
    Applies residual shortcut when input and output shapes match and stride == 1.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        dilation: int = 1,
        use_relu6: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.has_residual = (in_channels == out_channels) and (stride == 1)

        # Padding for same output length when stride=1
        padding = ((kernel_size - 1) * dilation) // 2

        # 1. Depthwise Convolution
        self.depthwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False
        )
        self.bn_dw = nn.BatchNorm1d(in_channels)

        # 2. Pointwise Convolution (1x1)
        self.pointwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn_pw = nn.BatchNorm1d(out_channels)

        # Activation: ReLU6 is standard & friendly for INT8 quantization in TFLite Micro
        self.act1 = nn.ReLU6(inplace=True) if use_relu6 else nn.ReLU(inplace=True)
        self.act2 = nn.ReLU6(inplace=True) if use_relu6 else nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # Depthwise pass
        out = self.depthwise(x)
        out = self.bn_dw(out)
        out = self.act1(out)

        # Pointwise pass
        out = self.pointwise(out)
        out = self.bn_pw(out)

        if self.has_residual:
            out = out + residual

        out = self.act2(out)
        return out


class KhusKhusKWS(nn.Module):
    """
    KhusKhus-KWS: Custom 1D Depthwise-Separable CNN Architecture for ESP32.
    
    Structure:
      Input (Batch, 40 Mel, Time=98)
        ↓
      Lightweight Conv1D Stem (40 -> 32, k=5, s=1)
        ↓
      DS Block 1 (32 -> 48, k=5, s=2)  [Time subsampling /2]
        ↓
      DS Block 2 (48 -> 64, k=5, s=2)  [Time subsampling /2]
        ↓
      DS Block 3 (64 -> 64, k=3, s=1)  [Residual connection]
        ↓
      Global Average Pooling (64 -> 64)
        ↓
      Linear Classifier (64 -> 3 classes)
        ↓
      Output: [0=keyword, 1=other_speech, 2=noise]
    """

    def __init__(self, cfg: ModelConfig = MODEL_CFG):
        super().__init__()
        self.cfg = cfg
        self.name = cfg.name

        # Stem Conv1D: processes 40 Mel bins into 32 feature channels
        stem_pad = (cfg.stem_kernel - 1) // 2
        self.stem_conv = nn.Conv1d(
            in_channels=cfg.in_channels,
            out_channels=cfg.stem_channels,
            kernel_size=cfg.stem_kernel,
            stride=cfg.stem_stride,
            padding=stem_pad,
            bias=False
        )
        self.stem_bn = nn.BatchNorm1d(cfg.stem_channels)
        self.stem_act = nn.ReLU6(inplace=True) if cfg.use_relu6 else nn.ReLU(inplace=True)

        # Depthwise-Separable Block 1: 32 -> 48 channels, stride=2
        self.ds_block1 = DepthwiseSeparableConv1d(
            in_channels=cfg.stem_channels,
            out_channels=cfg.ds1_channels,
            kernel_size=cfg.ds1_kernel,
            stride=cfg.ds1_stride,
            use_relu6=cfg.use_relu6
        )

        # Depthwise-Separable Block 2: 48 -> 64 channels, stride=2
        self.ds_block2 = DepthwiseSeparableConv1d(
            in_channels=cfg.ds1_channels,
            out_channels=cfg.ds2_channels,
            kernel_size=cfg.ds2_kernel,
            stride=cfg.ds2_stride,
            use_relu6=cfg.use_relu6
        )

        # Depthwise-Separable Block 3: 64 -> 64 channels, stride=1 (Residual)
        self.ds_block3 = DepthwiseSeparableConv1d(
            in_channels=cfg.ds2_channels,
            out_channels=cfg.ds3_channels,
            kernel_size=cfg.ds3_kernel,
            stride=cfg.ds3_stride,
            use_relu6=cfg.use_relu6
        )

        # Global Average Pooling across time
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Classification Head
        self.dropout = nn.Dropout(p=cfg.dropout)
        self.fc = nn.Linear(cfg.ds3_channels, cfg.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (batch, in_channels, time_frames) -> (B, 40, T)
        Returns:
            logits: Output tensor of shape (batch, num_classes) -> (B, 3)
        """
        # Stem
        x = self.stem_conv(x)
        x = self.stem_bn(x)
        x = self.stem_act(x)

        # Depthwise Separable Blocks
        x = self.ds_block1(x)
        x = self.ds_block2(x)
        x = self.ds_block3(x)

        # Pooling & Head
        x = self.global_pool(x)       # Shape: (B, 64, 1)
        x = torch.flatten(x, 1)        # Shape: (B, 64)
        x = self.dropout(x)
        logits = self.fc(x)           # Shape: (B, 3)

        return logits

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract bottleneck embedding vector before classification."""
        x = self.stem_act(self.stem_bn(self.stem_conv(x)))
        x = self.ds_block1(x)
        x = self.ds_block2(x)
        x = self.ds_block3(x)
        x = self.global_pool(x)
        return torch.flatten(x, 1)


def compute_model_statistics(
    model: nn.Module,
    input_shape: Tuple[int, int, int] = (1, 40, 98)
) -> Dict:
    """
    Computes accurate parameter counts, MACs, memory requirements, and layer details.
    """
    device = next(model.parameters()).device
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    layer_stats = []
    total_macs = 0
    max_activation_bytes = 0

    dummy_input = torch.zeros(input_shape, device=device)
    
    # Layer 1: Stem Conv1D
    # in: (1, 40, 98), out: (1, 32, 98)
    time_len = input_shape[2]
    # Conv1d MACs = out_channels * out_length * (in_channels / groups * kernel_size)
    stem_out_len = (time_len + 2 * model.stem_conv.padding[0] - model.stem_conv.kernel_size[0]) // model.stem_conv.stride[0] + 1
    stem_params = sum(p.numel() for p in model.stem_conv.parameters()) + sum(p.numel() for p in model.stem_bn.parameters())
    stem_macs = model.stem_conv.out_channels * stem_out_len * (model.stem_conv.in_channels * model.stem_conv.kernel_size[0])
    total_macs += stem_macs
    act_stem = 1 * model.stem_conv.out_channels * stem_out_len * 4
    max_activation_bytes = max(max_activation_bytes, act_stem)
    layer_stats.append({
        "layer": "Stem Conv1D + BN + ReLU6",
        "input_shape": f"(1, {model.stem_conv.in_channels}, {time_len})",
        "output_shape": f"(1, {model.stem_conv.out_channels}, {stem_out_len})",
        "kernel": f"{model.stem_conv.kernel_size[0]}",
        "stride": f"{model.stem_conv.stride[0]}",
        "channels": f"{model.stem_conv.out_channels}",
        "params": stem_params,
        "macs": stem_macs
    })

    # DS Block 1
    # Depthwise: in=32, out=32, k=5, s=2
    dw1_len = (stem_out_len + 2 * model.ds_block1.depthwise.padding[0] - model.ds_block1.depthwise.kernel_size[0]) // model.ds_block1.depthwise.stride[0] + 1
    dw1_params = sum(p.numel() for p in model.ds_block1.depthwise.parameters()) + sum(p.numel() for p in model.ds_block1.bn_dw.parameters())
    dw1_macs = model.ds_block1.depthwise.out_channels * dw1_len * (1 * model.ds_block1.depthwise.kernel_size[0]) # groups=in_ch
    pw1_params = sum(p.numel() for p in model.ds_block1.pointwise.parameters()) + sum(p.numel() for p in model.ds_block1.bn_pw.parameters())
    pw1_macs = model.ds_block1.pointwise.out_channels * dw1_len * (model.ds_block1.pointwise.in_channels * 1)
    ds1_params = dw1_params + pw1_params
    ds1_macs = dw1_macs + pw1_macs
    total_macs += ds1_macs
    act_ds1 = 1 * model.ds_block1.pointwise.out_channels * dw1_len * 4
    max_activation_bytes = max(max_activation_bytes, act_ds1)
    layer_stats.append({
        "layer": "DS Block 1 (DW-k5-s2 + PW-k1-s1)",
        "input_shape": f"(1, {model.stem_conv.out_channels}, {stem_out_len})",
        "output_shape": f"(1, {model.ds_block1.pointwise.out_channels}, {dw1_len})",
        "kernel": "DW: 5, PW: 1",
        "stride": "DW: 2, PW: 1",
        "channels": f"{model.ds_block1.pointwise.out_channels}",
        "params": ds1_params,
        "macs": ds1_macs
    })

    # DS Block 2
    # Depthwise: in=48, out=48, k=5, s=2
    dw2_len = (dw1_len + 2 * model.ds_block2.depthwise.padding[0] - model.ds_block2.depthwise.kernel_size[0]) // model.ds_block2.depthwise.stride[0] + 1
    dw2_params = sum(p.numel() for p in model.ds_block2.depthwise.parameters()) + sum(p.numel() for p in model.ds_block2.bn_dw.parameters())
    dw2_macs = model.ds_block2.depthwise.out_channels * dw2_len * (1 * model.ds_block2.depthwise.kernel_size[0])
    pw2_params = sum(p.numel() for p in model.ds_block2.pointwise.parameters()) + sum(p.numel() for p in model.ds_block2.bn_pw.parameters())
    pw2_macs = model.ds_block2.pointwise.out_channels * dw2_len * (model.ds_block2.pointwise.in_channels * 1)
    ds2_params = dw2_params + pw2_params
    ds2_macs = dw2_macs + pw2_macs
    total_macs += ds2_macs
    act_ds2 = 1 * model.ds_block2.pointwise.out_channels * dw2_len * 4
    max_activation_bytes = max(max_activation_bytes, act_ds2)
    layer_stats.append({
        "layer": "DS Block 2 (DW-k5-s2 + PW-k1-s1)",
        "input_shape": f"(1, {model.ds_block1.pointwise.out_channels}, {dw1_len})",
        "output_shape": f"(1, {model.ds_block2.pointwise.out_channels}, {dw2_len})",
        "kernel": "DW: 5, PW: 1",
        "stride": "DW: 2, PW: 1",
        "channels": f"{model.ds_block2.pointwise.out_channels}",
        "params": ds2_params,
        "macs": ds2_macs
    })

    # DS Block 3
    # Depthwise: in=64, out=64, k=3, s=1
    dw3_len = (dw2_len + 2 * model.ds_block3.depthwise.padding[0] - model.ds_block3.depthwise.kernel_size[0]) // model.ds_block3.depthwise.stride[0] + 1
    dw3_params = sum(p.numel() for p in model.ds_block3.depthwise.parameters()) + sum(p.numel() for p in model.ds_block3.bn_dw.parameters())
    dw3_macs = model.ds_block3.depthwise.out_channels * dw3_len * (1 * model.ds_block3.depthwise.kernel_size[0])
    pw3_params = sum(p.numel() for p in model.ds_block3.pointwise.parameters()) + sum(p.numel() for p in model.ds_block3.bn_pw.parameters())
    pw3_macs = model.ds_block3.pointwise.out_channels * dw3_len * (model.ds_block3.pointwise.in_channels * 1)
    ds3_params = dw3_params + pw3_params
    ds3_macs = dw3_macs + pw3_macs
    total_macs += ds3_macs
    act_ds3 = 1 * model.ds_block3.pointwise.out_channels * dw3_len * 4
    max_activation_bytes = max(max_activation_bytes, act_ds3)
    layer_stats.append({
        "layer": "DS Block 3 (DW-k3-s1 + PW-k1-s1 + Res)",
        "input_shape": f"(1, {model.ds_block2.pointwise.out_channels}, {dw2_len})",
        "output_shape": f"(1, {model.ds_block3.pointwise.out_channels}, {dw3_len})",
        "kernel": "DW: 3, PW: 1",
        "stride": "DW: 1, PW: 1",
        "channels": f"{model.ds_block3.pointwise.out_channels}",
        "params": ds3_params,
        "macs": ds3_macs
    })

    # Global Avg Pool
    gap_macs = model.ds_block3.pointwise.out_channels * dw3_len
    total_macs += gap_macs
    layer_stats.append({
        "layer": "Global Adaptive Avg Pool",
        "input_shape": f"(1, {model.ds_block3.pointwise.out_channels}, {dw3_len})",
        "output_shape": f"(1, {model.ds_block3.pointwise.out_channels}, 1)",
        "kernel": f"Time: {dw3_len}",
        "stride": "1",
        "channels": f"{model.ds_block3.pointwise.out_channels}",
        "params": 0,
        "macs": gap_macs
    })

    # FC Layer
    fc_params = sum(p.numel() for p in model.fc.parameters())
    fc_macs = model.fc.in_features * model.fc.out_features
    total_macs += fc_macs
    layer_stats.append({
        "layer": "Dense (Linear Classifier)",
        "input_shape": f"(1, {model.fc.in_features})",
        "output_shape": f"(1, {model.fc.out_features})",
        "kernel": "-",
        "stride": "-",
        "channels": f"{model.fc.out_features}",
        "params": fc_params,
        "macs": fc_macs
    })

    fp32_size_bytes = total_params * 4
    int8_size_bytes = total_params * 1

    return {
        "model_name": model.name,
        "input_shape": list(input_shape),
        "output_classes": model.fc.out_features,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "estimated_macs": total_macs,
        "estimated_fp32_size_kb": fp32_size_bytes / 1024.0,
        "estimated_int8_size_kb": int8_size_bytes / 1024.0,
        "max_activation_tensor_kb": max_activation_bytes / 1024.0,
        "layers": layer_stats
    }


def generate_and_save_model_summary(
    save_dir: Path = MODEL_CFG.models_dir if hasattr(MODEL_CFG, "models_dir") else Path(r"E:\kwas\models\KhusKhus-KWS")
) -> Dict:
    """Generate model summary and configuration files."""
    save_dir.mkdir(parents=True, exist_ok=True)
    model = KhusKhusKWS(MODEL_CFG)
    stats = compute_model_statistics(model, (1, MODEL_CFG.in_channels, 98))

    # Save JSON config
    config_dict = {
        "model_name": MODEL_CFG.name,
        "in_channels": MODEL_CFG.in_channels,
        "stem_channels": MODEL_CFG.stem_channels,
        "stem_kernel": MODEL_CFG.stem_kernel,
        "ds1_channels": MODEL_CFG.ds1_channels,
        "ds1_kernel": MODEL_CFG.ds1_kernel,
        "ds1_stride": MODEL_CFG.ds1_stride,
        "ds2_channels": MODEL_CFG.ds2_channels,
        "ds2_kernel": MODEL_CFG.ds2_kernel,
        "ds2_stride": MODEL_CFG.ds2_stride,
        "ds3_channels": MODEL_CFG.ds3_channels,
        "ds3_kernel": MODEL_CFG.ds3_kernel,
        "ds3_stride": MODEL_CFG.ds3_stride,
        "num_classes": MODEL_CFG.num_classes,
        "use_relu6": MODEL_CFG.use_relu6,
        "dropout": MODEL_CFG.dropout,
        "statistics": {
            "total_parameters": stats["total_parameters"],
            "trainable_parameters": stats["trainable_parameters"],
            "estimated_macs": stats["estimated_macs"],
            "estimated_fp32_size_kb": stats["estimated_fp32_size_kb"],
            "estimated_int8_size_kb": stats["estimated_int8_size_kb"],
            "max_activation_tensor_kb": stats["max_activation_tensor_kb"]
        }
    }

    json_path = save_dir / "model_config.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    # Save formatted TXT summary
    txt_path = save_dir / "model_summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("==========================================================================================\n")
        f.write("KHUSKHUS-KWS ARCHITECTURE SUMMARY & ESP32 RESOURCE AUDIT\n")
        f.write("==========================================================================================\n\n")
        f.write(f"Model Name:              {stats['model_name']}\n")
        f.write(f"Input Shape:             (Batch, Channels={stats['input_shape'][1]}, Time={stats['input_shape'][2]})\n")
        f.write(f"Output Classes:          {stats['output_classes']} [0: keyword ('khus khus'), 1: other_speech, 2: noise]\n")
        f.write(f"Total Parameters:        {stats['total_parameters']:,}\n")
        f.write(f"Trainable Parameters:    {stats['trainable_parameters']:,}\n")
        f.write(f"Estimated MACs:          {stats['estimated_macs']:,}\n")
        f.write(f"FP32 Model Size:         {stats['estimated_fp32_size_kb']:.2f} KB\n")
        f.write(f"Estimated INT8 Size:     {stats['estimated_int8_size_kb']:.2f} KB\n")
        f.write(f"Max Activation Tensor:   {stats['max_activation_tensor_kb']:.2f} KB\n\n")
        f.write("Layer-by-Layer Breakdown Table:\n")
        f.write(f"{'-'*100}\n")
        f.write(f"{'Layer':<35} | {'Input Shape':<16} | {'Output Shape':<16} | {'Params':<8} | {'MACs':<10}\n")
        f.write(f"{'-'*100}\n")
        for layer in stats["layers"]:
            f.write(f"{layer['layer']:<35} | {layer['input_shape']:<16} | {layer['output_shape']:<16} | {layer['params']:<8} | {layer['macs']:<10}\n")
        f.write(f"{'-'*100}\n\n")
        f.write("ESP32 Deployment Compatibility Notes:\n")
        f.write("  - Conv1D, DepthwiseConv1D, Pointwise 1x1 Conv, BatchNorm (fused into conv at export),\n")
        f.write("    ReLU6, GlobalAveragePool, and Dense are 100% natively supported in TFLite Micro / ESP-NN.\n")
        f.write("  - Zero dynamic control flow, zero RNN/Transformer latency penalties.\n")
        f.write("  - Estimated INT8 memory footprint fits comfortably in ESP32 internal SRAM (<30 KB RAM).\n")
        f.write("==========================================================================================\n")

    return stats


if __name__ == "__main__":
    summary_stats = generate_and_save_model_summary(Path(r"E:\kwas\models\KhusKhus-KWS"))
    print(f"Model Summary Generated successfully:")
    print(f"  Total Parameters:    {summary_stats['total_parameters']}")
    print(f"  Estimated MACs:      {summary_stats['estimated_macs']}")
    print(f"  Estimated INT8 Size: {summary_stats['estimated_int8_size_kb']:.2f} KB")
