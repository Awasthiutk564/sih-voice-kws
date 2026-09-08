"""Streaming ASR Model in TensorFlow & TFLite for Edge & Embedded Devices.

A TensorFlow / Keras implementation of low-latency causal streaming ASR:
- 3-Layer 1D Causal Convolutional Audio Stem (384x downsampling, direct raw waveform processing)
- Causal Multi-Head Self-Attention Transformer Encoder Blocks
- Low-Latency CTC Token Decoder
- TFLite Float32 & Int8 Quantization Exporter + C Header Generator for ESP32
"""

import os
import math
import time
from typing import List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# =====================================================================
# 1. Causal 1D Convolution & ConvStem in Keras
# =====================================================================

class CausalConv1D(layers.Layer):
    """Causal 1D Convolution with explicit left-padding."""

    def __init__(
        self,
        filters: int,
        kernel_size: int,
        strides: int = 1,
        dilation_rate: int = 1,
        use_bias: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.padding = (kernel_size - 1) * dilation_rate
        self.conv = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            dilation_rate=dilation_rate,
            padding="valid",
            use_bias=use_bias,
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        # x shape: (Batch, Time, Channels)
        paddings = tf.constant([[0, 0], [self.padding, 0], [0, 0]])
        x_padded = tf.pad(x, paddings)
        return self.conv(x_padded)


class ConvStemTF(layers.Layer):
    """3-Layer 1D Causal Convolutional Audio Stem with 384x Downsampling.
    
    16kHz Audio -> ~41.67 fps (~24ms per frame).
    """

    def __init__(
        self,
        channels: Optional[List[int]] = None,
        kernel_sizes: Optional[List[int]] = None,
        strides: Optional[List[int]] = None,
        out_dim: int = 192,
        **kwargs
    ):
        super().__init__(**kwargs)
        channels = channels if channels is not None else [32, 64, 128]
        kernel_sizes = kernel_sizes if kernel_sizes is not None else [128, 7, 5]
        strides = strides if strides is not None else [64, 3, 2]

        self.conv1 = CausalConv1D(channels[0], kernel_sizes[0], strides[0], name="stem_conv1")
        self.norm1 = layers.LayerNormalization(epsilon=1e-5, name="stem_norm1")
        self.act1 = layers.Activation("gelu")

        self.conv2 = CausalConv1D(channels[1], kernel_sizes[1], strides[1], name="stem_conv2")
        self.norm2 = layers.LayerNormalization(epsilon=1e-5, name="stem_norm2")
        self.act2 = layers.Activation("gelu")

        self.conv3 = CausalConv1D(channels[2], kernel_sizes[2], strides[2], name="stem_conv3")
        self.norm3 = layers.LayerNormalization(epsilon=1e-5, name="stem_norm3")
        self.act3 = layers.Activation("gelu")

        self.proj = layers.Dense(out_dim, name="stem_proj")
        self.proj_norm = layers.LayerNormalization(epsilon=1e-5, name="stem_proj_norm")

    def call(self, x: tf.Tensor) -> tf.Tensor:
        # Expected input: (Batch, Time, 1) or (Batch, Time)
        if len(x.shape) == 2:
            x = tf.expand_dims(x, axis=-1)

        x = self.act1(self.norm1(self.conv1(x)))
        x = self.act2(self.norm2(self.conv2(x)))
        x = self.act3(self.norm3(self.conv3(x)))
        x = self.proj_norm(self.proj(x))
        return x


# =====================================================================
# 2. Causal Transformer Encoder Block in Keras
# =====================================================================

class CausalTransformerBlock(layers.Layer):
    """Pre-LN Causal Transformer Encoder Layer compatible with TFLite / TFLM."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=self.head_dim,
            dropout=dropout,
        )
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)
        self.ffn = keras.Sequential([
            layers.Dense(d_ff, activation="gelu"),
            layers.Dense(d_model),
        ])

    def call(self, x: tf.Tensor) -> tf.Tensor:
        # Pre-LN Self-Attention with Causal Masking
        normed = self.norm1(x)
        attn_out = self.mha(query=normed, value=normed, key=normed, use_causal_mask=True)
        x = x + attn_out

        # Pre-LN FFN
        x = x + self.ffn(self.norm2(x))
        return x


# =====================================================================
# 3. Streaming ASR Complete Keras Model
# =====================================================================

def build_streaming_asr_model(
    input_samples: int = 6144,
    d_model: int = 192,
    num_heads: int = 3,
    num_layers: int = 4,
    d_ff: int = 384,
    vocab_size: int = 32,
    stem_channels: Optional[List[int]] = None,
) -> keras.Model:
    """Builds an end-to-end Streaming ASR Model in TensorFlow / Keras."""
    stem_channels = stem_channels if stem_channels is not None else [32, 64, 128]

    audio_input = layers.Input(shape=(input_samples,), dtype=tf.float32, name="audio_waveform")
    x = layers.Reshape((input_samples, 1))(audio_input)

    # 1. Causal Conv Stem (384x downsample)
    stem = ConvStemTF(channels=stem_channels, out_dim=d_model, name="conv_stem")
    x = stem(x)

    # 2. Causal Transformer Blocks
    for i in range(num_layers):
        block = CausalTransformerBlock(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            name=f"transformer_block_{i}",
        )
        x = block(x)

    # 3. Final LayerNorm & CTC Projection Head
    x = layers.LayerNormalization(epsilon=1e-5, name="final_norm")(x)
    ctc_logits = layers.Dense(vocab_size, name="ctc_logits")(x)

    model = keras.Model(inputs=audio_input, outputs=ctc_logits, name="StreamingASR_TF")
    return model


# =====================================================================
# 4. TFLite & ESP32 C Header Export Utilities
# =====================================================================

def export_to_tflite(
    model: keras.Model,
    output_tflite_path: str = "models/asr/streaming_asr.tflite",
    quantize_int8: bool = False,
    representative_samples: Optional[np.ndarray] = None,
) -> str:
    """Converts the Keras model to a standalone TFLite model file."""
    os.makedirs(os.path.dirname(output_tflite_path), exist_ok=True)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]

    if quantize_int8:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if representative_samples is not None:
            def representative_data_gen():
                for sample in representative_samples:
                    yield [np.expand_dims(sample.astype(np.float32), axis=0)]
            converter.representative_dataset = representative_data_gen
            converter.target_spec.supported_types = [tf.int8]

    tflite_model = converter.convert()

    with open(output_tflite_path, "wb") as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024.0
    print(f"[+] Exported TFLite Model: {output_tflite_path} ({size_kb:.1f} KB)")
    return output_tflite_path


def export_tflite_to_c_header(
    tflite_path: str,
    output_header_path: str = "esp32_firmware/streaming_asr_tflite_model.h",
    array_name: str = "g_streaming_asr_model_data",
):
    """Converts a .tflite model binary into a C++ header file for ESP32."""
    os.makedirs(os.path.dirname(output_header_path), exist_ok=True)
    with open(tflite_path, "rb") as f:
        tflite_bytes = f.read()

    hex_array = []
    for i in range(0, len(tflite_bytes), 12):
        chunk = tflite_bytes[i : i + 12]
        hex_array.append("    " + ", ".join(f"0x{b:02x}" for b in chunk))

    header_content = f"""// Auto-generated Streaming ASR TensorFlow Lite model for ESP32-S3
#ifndef STREAMING_ASR_TFLITE_MODEL_H_
#define STREAMING_ASR_TFLITE_MODEL_H_

#include <stdint.h>

alignas(16) const unsigned char {array_name}[] = {{
{',\n'.join(hex_array)}
}};
const unsigned int {array_name}_len = {len(tflite_bytes)};

#endif  // STREAMING_ASR_TFLITE_MODEL_H_
"""
    with open(output_header_path, "w", encoding="utf-8") as f:
        f.write(header_content)

    print(f"[+] Exported C Header for ESP32: {output_header_path} ({len(tflite_bytes)} bytes)")


# =====================================================================
# 5. Model Architecture Summary & Test
# =====================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("   Building & Exporting TensorFlow / TFLite Streaming ASR Model")
    print("=" * 65)

    # 1. Build Micro ASR Model (6144 audio samples = ~384ms per chunk at 16kHz)
    CHUNK_SAMPLES = 6144
    model = build_streaming_asr_model(
        input_samples=CHUNK_SAMPLES,
        d_model=192,
        num_heads=3,
        num_layers=4,
        d_ff=384,
        vocab_size=32,
    )

    model.summary()

    # 2. Test forward pass with dummy audio chunk
    dummy_audio = np.random.randn(1, CHUNK_SAMPLES).astype(np.float32)
    t0 = time.perf_counter()
    logits = model(dummy_audio)
    t1 = time.perf_counter()
    print(f"\n[+] TF Forward Pass Output Shape: {logits.shape} (Time: {(t1 - t0)*1000:.2f} ms)")

    # 3. Export to TFLite
    tflite_path = "models/asr/streaming_asr.tflite"
    export_to_tflite(model, tflite_path)

    # 4. Export C Header for ESP32
    header_path = "esp32_firmware/streaming_asr_tflite_model.h"
    export_tflite_to_c_header(tflite_path, header_path)

    print("\n[+] TensorFlow & TFLite Model Build & Export Complete!\n")
