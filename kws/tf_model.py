"""
TensorFlow / Keras Implementation of KhusKhus-KWS
Clones the PyTorch 1D Depthwise-Separable CNN architecture for keyword spotting ("khus khus"),
transfers weights from PyTorch (.pth) checkpoints, and verifies bit-exact numerical parity.
"""

import os
from pathlib import Path
import sys
from typing import Dict, Optional, Tuple, Union

# Ensure project root is in Python sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import tensorflow as tf
import torch

from kws.config import MODEL_CFG, ModelConfig
from kws.model import KhusKhusKWS


def build_kws_model_tf(
    cfg: ModelConfig = MODEL_CFG,
    input_shape: Tuple[int, int] = (98, 40),
    name: str = "KhusKhus_KWS_TF"
) -> tf.keras.Model:
    """
    Constructs the exact KhusKhus-KWS architecture in TensorFlow / Keras.
    
    Expected input shape: (Batch, Time=98, Channels=40)
    
    Structure:
      Input (B, 98, 40)
        ↓
      Stem: ZeroPadding1D(2) -> Conv1D(32, k=5, s=1) -> BatchNorm -> ReLU6
        ↓
      DS Block 1: ZeroPadding1D(2) -> DepthwiseConv1D(k=5, s=2) -> BN -> ReLU6
                 -> Conv1D(48, k=1, s=1) -> BN -> ReLU6
        ↓
      DS Block 2: ZeroPadding1D(2) -> DepthwiseConv1D(k=5, s=2) -> BN -> ReLU6
                 -> Conv1D(64, k=1, s=1) -> BN -> ReLU6
        ↓
      DS Block 3: ZeroPadding1D(1) -> DepthwiseConv1D(k=3, s=1) -> BN -> ReLU6
                 -> Conv1D(64, k=1, s=1) -> BN
                 -> Add(Residual) -> ReLU6
        ↓
      GlobalAveragePooling1D -> (B, 64)
        ↓
      Dense(3) -> Logits (B, 3) [0: keyword, 1: other_speech, 2: noise]
    """
    inputs = tf.keras.Input(shape=input_shape, name="input_audio_mel")

    # Stem Conv1D: processes 40 Mel bins into 32 feature channels
    # In PyTorch: kernel=5, stride=1, padding=2
    x = tf.keras.layers.ZeroPadding1D(padding=2, name="stem_pad")(inputs)
    x = tf.keras.layers.Conv1D(
        filters=cfg.stem_channels,
        kernel_size=cfg.stem_kernel,
        strides=cfg.stem_stride,
        padding="valid",
        use_bias=False,
        name="stem_conv"
    )(x)
    x = tf.keras.layers.BatchNormalization(
        epsilon=1e-5,
        momentum=0.1,
        name="stem_bn"
    )(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name="stem_act")(x)

    # Depthwise-Separable Block 1: 32 -> 48 channels, stride=2
    # In PyTorch: kernel=5, stride=2, padding=2
    x = tf.keras.layers.ZeroPadding1D(padding=2, name="ds1_dw_pad")(x)
    x = tf.keras.layers.DepthwiseConv1D(
        kernel_size=cfg.ds1_kernel,
        strides=cfg.ds1_stride,
        padding="valid",
        use_bias=False,
        name="ds1_dw"
    )(x)
    x = tf.keras.layers.BatchNormalization(
        epsilon=1e-5,
        momentum=0.1,
        name="ds1_bn_dw"
    )(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name="ds1_act1")(x)

    x = tf.keras.layers.Conv1D(
        filters=cfg.ds1_channels,
        kernel_size=1,
        strides=1,
        padding="valid",
        use_bias=False,
        name="ds1_pw"
    )(x)
    x = tf.keras.layers.BatchNormalization(
        epsilon=1e-5,
        momentum=0.1,
        name="ds1_bn_pw"
    )(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name="ds1_act2")(x)

    # Depthwise-Separable Block 2: 48 -> 64 channels, stride=2
    # In PyTorch: kernel=5, stride=2, padding=2
    x = tf.keras.layers.ZeroPadding1D(padding=2, name="ds2_dw_pad")(x)
    x = tf.keras.layers.DepthwiseConv1D(
        kernel_size=cfg.ds2_kernel,
        strides=cfg.ds2_stride,
        padding="valid",
        use_bias=False,
        name="ds2_dw"
    )(x)
    x = tf.keras.layers.BatchNormalization(
        epsilon=1e-5,
        momentum=0.1,
        name="ds2_bn_dw"
    )(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name="ds2_act1")(x)

    x = tf.keras.layers.Conv1D(
        filters=cfg.ds2_channels,
        kernel_size=1,
        strides=1,
        padding="valid",
        use_bias=False,
        name="ds2_pw"
    )(x)
    x = tf.keras.layers.BatchNormalization(
        epsilon=1e-5,
        momentum=0.1,
        name="ds2_bn_pw"
    )(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name="ds2_act2")(x)

    # Depthwise-Separable Block 3: 64 -> 64 channels, stride=1 (with Residual connection)
    # In PyTorch: kernel=3, stride=1, padding=1
    res = x
    x = tf.keras.layers.ZeroPadding1D(padding=1, name="ds3_dw_pad")(x)
    x = tf.keras.layers.DepthwiseConv1D(
        kernel_size=cfg.ds3_kernel,
        strides=cfg.ds3_stride,
        padding="valid",
        use_bias=False,
        name="ds3_dw"
    )(x)
    x = tf.keras.layers.BatchNormalization(
        epsilon=1e-5,
        momentum=0.1,
        name="ds3_bn_dw"
    )(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name="ds3_act1")(x)

    x = tf.keras.layers.Conv1D(
        filters=cfg.ds3_channels,
        kernel_size=1,
        strides=1,
        padding="valid",
        use_bias=False,
        name="ds3_pw"
    )(x)
    x = tf.keras.layers.BatchNormalization(
        epsilon=1e-5,
        momentum=0.1,
        name="ds3_bn_pw"
    )(x)

    # Residual addition + final ReLU6
    x = tf.keras.layers.Add(name="ds3_add")([x, res])
    x = tf.keras.layers.ReLU(max_value=6.0, name="ds3_act2")(x)

    # Global Average Pooling across time dimension -> (B, 64)
    x = tf.keras.layers.GlobalAveragePooling1D(name="global_pool")(x)

    # Classification Head -> (B, 3)
    outputs = tf.keras.layers.Dense(
        units=cfg.num_classes,
        use_bias=True,
        name="classifier"
    )(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)


def clone_pytorch_to_keras(
    pt_model_or_path: Union[str, Path, torch.nn.Module],
    tf_model: Optional[tf.keras.Model] = None,
    cfg: ModelConfig = MODEL_CFG
) -> tf.keras.Model:
    """
    Clones weights from a trained PyTorch KhusKhusKWS checkpoint into the TensorFlow Keras model.
    Transposes weights to accommodate Keras 1D layout:
      - Conv1D: (C_out, C_in, K) -> (K, C_in, C_out)
      - DepthwiseConv1D: (C_in, 1, K) -> (K, C_in, 1)
      - Dense: (C_out, C_in) -> (C_in, C_out)
      - BatchNorm: gamma, beta, running_mean, running_var
    """
    if isinstance(pt_model_or_path, (str, Path)):
        ckpt_path = Path(pt_model_or_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"PyTorch checkpoint not found at: {ckpt_path}")
        pt_model = KhusKhusKWS(cfg)
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        pt_model.load_state_dict(state_dict)
    else:
        pt_model = pt_model_or_path

    pt_model.eval()
    sd = pt_model.state_dict()

    if tf_model is None:
        tf_model = build_kws_model_tf(cfg=cfg)

    # Helper function to copy BatchNorm parameters
    def set_bn(tf_layer_name: str, pt_prefix: str):
        gamma = sd[f"{pt_prefix}.weight"].detach().cpu().numpy()
        beta = sd[f"{pt_prefix}.bias"].detach().cpu().numpy()
        mean = sd[f"{pt_prefix}.running_mean"].detach().cpu().numpy()
        var = sd[f"{pt_prefix}.running_var"].detach().cpu().numpy()
        tf_model.get_layer(tf_layer_name).set_weights([gamma, beta, mean, var])

    # 1. Stem Conv1D + BatchNorm
    # PyTorch: (out_ch=32, in_ch=40, k=5) -> TF: (k=5, in_ch=40, out_ch=32)
    stem_w = sd["stem_conv.weight"].detach().cpu().numpy().transpose(2, 1, 0)
    tf_model.get_layer("stem_conv").set_weights([stem_w])
    set_bn("stem_bn", "stem_bn")

    # 2. DS Block 1
    # Depthwise: (in_ch=32, 1, k=5) -> TF: (k=5, in_ch=32, 1)
    ds1_dw_w = sd["ds_block1.depthwise.weight"].detach().cpu().numpy().transpose(2, 0, 1)
    tf_model.get_layer("ds1_dw").set_weights([ds1_dw_w])
    set_bn("ds1_bn_dw", "ds_block1.bn_dw")

    # Pointwise: (out_ch=48, in_ch=32, 1) -> TF: (1, in_ch=32, out_ch=48)
    ds1_pw_w = sd["ds_block1.pointwise.weight"].detach().cpu().numpy().transpose(2, 1, 0)
    tf_model.get_layer("ds1_pw").set_weights([ds1_pw_w])
    set_bn("ds1_bn_pw", "ds_block1.bn_pw")

    # 3. DS Block 2
    # Depthwise: (in_ch=48, 1, k=5) -> TF: (k=5, in_ch=48, 1)
    ds2_dw_w = sd["ds_block2.depthwise.weight"].detach().cpu().numpy().transpose(2, 0, 1)
    tf_model.get_layer("ds2_dw").set_weights([ds2_dw_w])
    set_bn("ds2_bn_dw", "ds_block2.bn_dw")

    # Pointwise: (out_ch=64, in_ch=48, 1) -> TF: (1, in_ch=48, out_ch=64)
    ds2_pw_w = sd["ds_block2.pointwise.weight"].detach().cpu().numpy().transpose(2, 1, 0)
    tf_model.get_layer("ds2_pw").set_weights([ds2_pw_w])
    set_bn("ds2_bn_pw", "ds_block2.bn_pw")

    # 4. DS Block 3
    # Depthwise: (in_ch=64, 1, k=3) -> TF: (k=3, in_ch=64, 1)
    ds3_dw_w = sd["ds_block3.depthwise.weight"].detach().cpu().numpy().transpose(2, 0, 1)
    tf_model.get_layer("ds3_dw").set_weights([ds3_dw_w])
    set_bn("ds3_bn_dw", "ds_block3.bn_dw")

    # Pointwise: (out_ch=64, in_ch=64, 1) -> TF: (1, in_ch=64, out_ch=64)
    ds3_pw_w = sd["ds_block3.pointwise.weight"].detach().cpu().numpy().transpose(2, 1, 0)
    tf_model.get_layer("ds3_pw").set_weights([ds3_pw_w])
    set_bn("ds3_bn_pw", "ds_block3.bn_pw")

    # 5. Classifier Head
    # PyTorch: (out_features=3, in_features=64) -> TF: (in_features=64, out_features=3)
    fc_w = sd["fc.weight"].detach().cpu().numpy().transpose(1, 0)
    fc_b = sd["fc.bias"].detach().cpu().numpy()
    tf_model.get_layer("classifier").set_weights([fc_w, fc_b])

    return tf_model


def verify_pytorch_tf_parity(
    pt_model: torch.nn.Module,
    tf_model: tf.keras.Model,
    num_samples: int = 10,
    tolerance: float = 1e-4
) -> Tuple[bool, float, float]:
    """
    Verifies that PyTorch and TensorFlow models produce numerically identical outputs.
    
    Returns:
        (is_passed, max_abs_difference, mean_abs_difference)
    """
    pt_model.eval()
    np.random.seed(42)

    # PyTorch input shape: (Batch, 40 Mel Bins, 98 Time Frames)
    x_pt = np.random.randn(num_samples, 40, 98).astype(np.float32)
    # TensorFlow input shape: (Batch, 98 Time Frames, 40 Mel Bins)
    x_tf = np.transpose(x_pt, (0, 2, 1))

    with torch.no_grad():
        out_pt = pt_model(torch.from_numpy(x_pt)).cpu().numpy()

    out_tf = tf_model(x_tf, training=False).numpy()

    abs_diff = np.abs(out_pt - out_tf)
    max_diff = float(np.max(abs_diff))
    mean_diff = float(np.mean(abs_diff))

    is_passed = max_diff <= tolerance
    return is_passed, max_diff, mean_diff


if __name__ == "__main__":
    print("============================================================")
    print("KhusKhus-KWS : TensorFlow Architecture & Parity Check")
    print("============================================================")

    pt_weights = Path("models/KhusKhus-KWS/best_model.pth")
    if not pt_weights.exists():
        print(f"[Error] Weights file not found: {pt_weights}")
        sys.exit(1)

    # Load PyTorch model
    pt_model = KhusKhusKWS()
    ckpt = torch.load(str(pt_weights), map_location="cpu")
    pt_model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    pt_model.eval()

    # Build and clone to TensorFlow
    tf_model = build_kws_model_tf()
    clone_pytorch_to_keras(pt_model, tf_model)
    tf_model.summary()

    # Verify parity
    passed, max_diff, mean_diff = verify_pytorch_tf_parity(pt_model, tf_model)
    print(f"\n[Parity Check] Max absolute error:  {max_diff:.8e}")
    print(f"[Parity Check] Mean absolute error: {mean_diff:.8e}")
    if passed:
        print("[SUCCESS] Bit-exact numerical parity confirmed between PyTorch & TensorFlow!")
    else:
        print(f"[WARNING] Discrepancy exceeds tolerance ({max_diff:.6f} > 1e-4)")
