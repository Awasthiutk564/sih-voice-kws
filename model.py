"""Moonshine Streaming ASR Model for Edge & Embedded Devices.

A standalone, self-contained implementation replicating Useful Sensors' Moonshine Streaming architecture:
- 3-Layer 1D Causal Convolutional Audio Stem (384x downsampling, direct raw waveform processing)
- Rotary Position Embeddings (RoPE)
- Pre-LN Streaming Causal Transformer Encoder with Stateful KV-Caching
- Low-Latency CTC Token Decoder & Streaming Audio Session Engine
"""

import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# 1. 1D Causal Convolutional Stem (384x Downsampling)
# =====================================================================

class CausalConv1d(nn.Module):
    """Causal 1D Convolution with history buffer for streaming."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
        )

    def forward(
        self, x: torch.Tensor, state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor (B, C, T)
            state: Previous audio history tensor (B, C, padding) or None

        Returns:
            out: Convolved tensor (B, C_out, T_out)
            new_state: Updated history buffer
        """
        if state is not None:
            x_padded = torch.cat([state, x], dim=-1)
        else:
            x_padded = F.pad(x, (self.padding, 0))

        new_state = (
            x_padded[:, :, -self.padding :]
            if self.padding > 0
            else torch.empty(0, device=x.device)
        )
        out = self.conv(x_padded)
        return out, new_state


class ConvStem(nn.Module):
    """3-Layer 1D Convolutional Audio Stem providing 384x downsampling.

    Configuration:
    - Layer 1: Conv1D(1 -> 64, kernel=128, stride=64)
    - Layer 2: Conv1D(64 -> 128, kernel=7, stride=3)
    - Layer 3: Conv1D(128 -> 256, kernel=5, stride=2)
    Total Downsampling = 64 * 3 * 2 = 384x (16,000 Hz -> 41.67 fps, ~24ms/frame).
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: Optional[List[int]] = None,
        kernel_sizes: Optional[List[int]] = None,
        strides: Optional[List[int]] = None,
        out_dim: int = 416,
        activation: str = "gelu",
    ):
        super().__init__()
        channels = channels if channels is not None else [64, 128, 256]
        kernel_sizes = kernel_sizes if kernel_sizes is not None else [128, 7, 5]
        strides = strides if strides is not None else [64, 3, 2]

        self.channels = channels
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.total_stride = strides[0] * strides[1] * strides[2]

        self.conv1 = CausalConv1d(in_channels, channels[0], kernel_sizes[0], strides[0])
        self.norm1 = nn.GroupNorm(1, channels[0])

        self.conv2 = CausalConv1d(channels[0], channels[1], kernel_sizes[1], strides[1])
        self.norm2 = nn.GroupNorm(1, channels[1])

        self.conv3 = CausalConv1d(channels[1], channels[2], kernel_sizes[2], strides[2])
        self.norm3 = nn.GroupNorm(1, channels[2])

        self.proj = nn.Linear(channels[2], out_dim)
        self.proj_norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU() if activation == "gelu" else nn.ReLU()

    def forward(
        self,
        audio_waveform: torch.Tensor,
        conv_states: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass.

        Args:
            audio_waveform: Raw audio tensor (B, T_samples) or (B, 1, T_samples)
            conv_states: List of 3 cached conv state tensors
        """
        if audio_waveform.dim() == 2:
            x = audio_waveform.unsqueeze(1)
        else:
            x = audio_waveform

        s1 = conv_states[0] if conv_states is not None else None
        s2 = conv_states[1] if conv_states is not None else None
        s3 = conv_states[2] if conv_states is not None else None

        x, ns1 = self.conv1(x, s1)
        x = self.act(self.norm1(x))

        x, ns2 = self.conv2(x, s2)
        x = self.act(self.norm2(x))

        x, ns3 = self.conv3(x, s3)
        x = self.act(self.norm3(x))

        x = x.transpose(1, 2)
        features = self.proj_norm(self.proj(x))

        return features, [ns1, ns2, ns3]


# =====================================================================
# 2. Rotary Position Embeddings (RoPE)
# =====================================================================

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) supporting streaming offsets."""

    def __init__(self, dim: int, max_position_embeddings: int = 4096, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len: int):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        required = offset + seq_len
        if required > self.max_seq_len_cached or self.cos_cached.device != x.device:
            self._set_cos_sin_cache(max(required, self.max_position_embeddings))
            self.cos_cached = self.cos_cached.to(x.device)
            self.sin_cached = self.sin_cached.to(x.device)

        cos = self.cos_cached[offset : offset + seq_len].to(dtype=x.dtype)
        sin = self.sin_cached[offset : offset + seq_len].to(dtype=x.dtype)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


# =====================================================================
# 3. Causal Multi-Head Attention & Transformer Blocks
# =====================================================================

class StreamingMultiHeadAttention(nn.Module):
    """Causal Attention with RoPE and streaming KV-cache."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        max_position_embeddings: int = 4096,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.rotary_emb = RotaryEmbedding(
            self.head_dim, max_position_embeddings=max_position_embeddings, base=rope_base
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(q, seq_len=T, offset=position_offset)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_kv_cache = (k, v)
        T_k = k.shape[2]

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if mask is not None:
            attn_scores = attn_scores + mask
        elif kv_cache is None and T > 1:
            causal_mask = torch.triu(
                torch.full((T, T), float("-inf"), device=x.device, dtype=x.dtype), diagonal=1
            )
            attn_scores = attn_scores + causal_mask.unsqueeze(0).unsqueeze(0)
        elif kv_cache is not None and T > 1:
            causal_mask = torch.zeros((T, T_k), device=x.device, dtype=x.dtype)
            for i in range(T):
                causal_mask[i, position_offset + i + 1 :] = float("-inf")
            attn_scores = attn_scores + causal_mask.unsqueeze(0).unsqueeze(0)

        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(x.dtype)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.out_proj(out)
        return out, new_kv_cache


class TransformerBlock(nn.Module):
    """Pre-LN Transformer Encoder Block."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        max_position_embeddings: int = 4096,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = StreamingMultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            max_position_embeddings=max_position_embeddings,
            rope_base=rope_base,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        normed_x = self.norm1(x)
        attn_out, new_kv_cache = self.attn(
            normed_x, mask=mask, kv_cache=kv_cache, position_offset=position_offset
        )
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, new_kv_cache


# =====================================================================
# 4. Moonshine Streaming Model & Session State
# =====================================================================

class StreamingSessionState:
    """State container for streaming audio inference."""

    def __init__(self, num_layers: int):
        self.conv_states: Optional[List[torch.Tensor]] = None
        self.kv_caches: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * num_layers
        self.frame_offset: int = 0
        self.total_audio_samples: int = 0

    def reset(self):
        self.conv_states = None
        self.kv_caches = [None] * len(self.kv_caches)
        self.frame_offset = 0
        self.total_audio_samples = 0


class MoonshineStreamingASR(nn.Module):
    """Moonshine Streaming Low-Latency ASR Model."""

    def __init__(
        self,
        stem_channels: Optional[List[int]] = None,
        stem_kernel_sizes: Optional[List[int]] = None,
        stem_strides: Optional[List[int]] = None,
        d_model: int = 416,
        num_heads: int = 8,
        num_layers: int = 8,
        d_ff: int = 1664,
        dropout: float = 0.1,
        max_position_embeddings: int = 4096,
        rope_base: float = 10000.0,
        vocab_size: int = 32,
        blank_id: int = 0,
        sample_rate: int = 16000,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.blank_id = blank_id
        self.sample_rate = sample_rate

        self.conv_stem = ConvStem(
            in_channels=1,
            channels=stem_channels if stem_channels is not None else [64, 128, 256],
            kernel_sizes=stem_kernel_sizes if stem_kernel_sizes is not None else [128, 7, 5],
            strides=stem_strides if stem_strides is not None else [64, 3, 2],
            out_dim=d_model,
        )
        self.downsample_factor = self.conv_stem.total_stride

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    max_position_embeddings=max_position_embeddings,
                    rope_base=rope_base,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(d_model)
        self.ctc_head = nn.Linear(d_model, vocab_size)

    def create_session_state(self) -> StreamingSessionState:
        return StreamingSessionState(self.num_layers)

    def forward(self, audio_waveform: torch.Tensor) -> torch.Tensor:
        """Full sequence forward pass."""
        features, _ = self.conv_stem(audio_waveform)
        x = features
        for layer in self.layers:
            x, _ = layer(x, mask=None, kv_cache=None, position_offset=0)
        x = self.final_norm(x)
        return self.ctc_head(x)

    def streaming_forward(
        self,
        audio_chunk: torch.Tensor,
        session_state: Optional[StreamingSessionState] = None,
    ) -> Tuple[torch.Tensor, StreamingSessionState]:
        """Stateful streaming chunk forward pass."""
        if session_state is None:
            session_state = self.create_session_state()

        features, new_conv_states = self.conv_stem(
            audio_chunk, conv_states=session_state.conv_states
        )
        session_state.conv_states = new_conv_states

        B, T_chunk, _ = features.shape
        x = features
        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            x, updated_cache = layer(
                x,
                mask=None,
                kv_cache=session_state.kv_caches[i],
                position_offset=session_state.frame_offset,
            )
            new_kv_caches.append(updated_cache)

        session_state.kv_caches = new_kv_caches
        session_state.frame_offset += T_chunk
        session_state.total_audio_samples += audio_chunk.shape[-1]

        x = self.final_norm(x)
        chunk_logits = self.ctc_head(x)
        return chunk_logits, session_state

    @classmethod
    def create_esp32_micro(cls) -> "MoonshineStreamingASR":
        """Factory for Ultra-Compact ESP32-S3 Micro ASR (~1.5M params, sub-ms latency)."""
        return cls(
            stem_channels=[32, 64, 128],
            stem_kernel_sizes=[128, 7, 5],
            stem_strides=[64, 3, 2],
            d_model=192,
            num_heads=3,
            num_layers=4,
            d_ff=384,
            dropout=0.0,
            vocab_size=32,
        )

    @classmethod
    def create_tiny(cls) -> "MoonshineStreamingASR":
        """Factory for Moonshine Streaming Tiny (~17M params)."""
        return cls(
            stem_channels=[64, 128, 256],
            stem_kernel_sizes=[128, 7, 5],
            stem_strides=[64, 3, 2],
            d_model=416,
            num_heads=8,
            num_layers=8,
            d_ff=1664,
            dropout=0.1,
            vocab_size=32,
        )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def model_size_mb(self, bytes_per_elem: int = 4) -> float:
        total_params = sum(p.numel() for p in self.parameters())
        return (total_params * bytes_per_elem) / (1024 * 1024)


# =====================================================================
# 5. Fast Tokenizer & Streaming CTC Decoder
# =====================================================================

class CTCTokenizer:
    """Character Tokenizer for CTC Decoding."""

    DEFAULT_VOCAB = [
        "<blank>", " ",
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
        "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
        "u", "v", "w", "x", "y", "z",
        "'", ".", ",", "?",
    ]

    def __init__(self, vocab: Union[List[str], None] = None):
        self.vocab = vocab if vocab is not None else self.DEFAULT_VOCAB
        self.blank_id = 0
        self.char_to_id = {ch: idx for idx, ch in enumerate(self.vocab)}
        self.id_to_char = {idx: ch for idx, ch in enumerate(self.vocab)}

    def encode(self, text: str) -> List[int]:
        return [self.char_to_id[ch] for ch in text.lower() if ch in self.char_to_id]

    def decode(self, token_ids: List[int]) -> str:
        return "".join(
            self.id_to_char[tid]
            for tid in token_ids
            if tid != self.blank_id and tid in self.id_to_char
        )

    def ctc_decode(self, token_ids: List[int]) -> str:
        collapsed = []
        prev = None
        for tid in token_ids:
            if tid != prev:
                if tid != self.blank_id:
                    collapsed.append(tid)
                prev = tid
        return self.decode(collapsed)


class StreamingCTCDecoder:
    """Stateful Streaming CTC Decoder."""

    def __init__(self, tokenizer: CTCTokenizer, blank_id: int = 0):
        self.tokenizer = tokenizer
        self.blank_id = blank_id
        self.prev_token_id: Optional[int] = None
        self.emitted_token_ids: List[int] = []
        self.full_transcript: str = ""

    def reset(self):
        self.prev_token_id = None
        self.emitted_token_ids.clear()
        self.full_transcript = ""

    def decode_chunk_logits(self, logits: torch.Tensor) -> Tuple[str, str]:
        if logits.dim() == 3:
            logits = logits.squeeze(0)
        pred_token_ids = torch.argmax(logits, dim=-1).tolist()
        new_chars = []
        for tid in pred_token_ids:
            if tid != self.prev_token_id:
                if tid != self.blank_id:
                    self.emitted_token_ids.append(tid)
                    new_chars.append(self.tokenizer.id_to_char.get(tid, ""))
                self.prev_token_id = tid
        delta_text = "".join(new_chars)
        self.full_transcript += delta_text
        return delta_text, self.full_transcript


class StreamingAudioSession:
    """End-to-End Streaming Audio Session."""

    def __init__(
        self,
        model: MoonshineStreamingASR,
        tokenizer: Optional[CTCTokenizer] = None,
        chunk_samples: int = 6144,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.tokenizer = tokenizer if tokenizer is not None else CTCTokenizer()
        self.decoder = StreamingCTCDecoder(self.tokenizer, blank_id=model.blank_id)
        self.session_state = model.create_session_state()

        downsample = model.downsample_factor
        if chunk_samples % downsample != 0:
            chunk_samples = ((chunk_samples // downsample) + 1) * downsample
        self.chunk_samples = chunk_samples

        self.audio_buffer: List[float] = []
        self.chunk_latencies_ms: List[float] = []
        self.total_processed_audio_sec: float = 0.0

    def reset(self):
        self.session_state.reset()
        self.decoder.reset()
        self.audio_buffer.clear()
        self.chunk_latencies_ms.clear()
        self.total_processed_audio_sec = 0.0

    def feed_samples(self, samples: Union[List[float], np.ndarray, torch.Tensor]) -> List[Tuple[str, float]]:
        if isinstance(samples, torch.Tensor):
            samples = samples.detach().cpu().flatten().tolist()
        elif isinstance(samples, np.ndarray):
            samples = samples.flatten().tolist()

        self.audio_buffer.extend(samples)
        emissions = []

        while len(self.audio_buffer) >= self.chunk_samples:
            chunk = self.audio_buffer[: self.chunk_samples]
            self.audio_buffer = self.audio_buffer[self.chunk_samples :]
            chunk_tensor = torch.tensor(chunk, dtype=torch.float32, device=self.device).unsqueeze(0)

            t0 = time.perf_counter()
            with torch.no_grad():
                logits, self.session_state = self.model.streaming_forward(
                    chunk_tensor, self.session_state
                )
            t1 = time.perf_counter()
            lat_ms = (t1 - t0) * 1000.0

            delta, _ = self.decoder.decode_chunk_logits(logits)
            self.chunk_latencies_ms.append(lat_ms)
            self.total_processed_audio_sec += self.chunk_samples / self.model.sample_rate

            if delta:
                emissions.append((delta, lat_ms))

        return emissions

    def flush(self) -> List[Tuple[str, float]]:
        if not self.audio_buffer:
            return []
        remainder = len(self.audio_buffer)
        downsample = self.model.downsample_factor
        needed = ((remainder + downsample - 1) // downsample) * downsample
        padded_chunk = self.audio_buffer + [0.0] * (needed - remainder)
        self.audio_buffer.clear()

        chunk_tensor = torch.tensor(padded_chunk, dtype=torch.float32, device=self.device).unsqueeze(0)
        t0 = time.perf_counter()
        with torch.no_grad():
            logits, self.session_state = self.model.streaming_forward(
                chunk_tensor, self.session_state
            )
        t1 = time.perf_counter()
        lat_ms = (t1 - t0) * 1000.0

        delta, _ = self.decoder.decode_chunk_logits(logits)
        self.chunk_latencies_ms.append(lat_ms)
        self.total_processed_audio_sec += len(padded_chunk) / self.model.sample_rate

        emissions = []
        if delta:
            emissions.append((delta, lat_ms))
        return emissions

    @property
    def current_transcript(self) -> str:
        return self.decoder.full_transcript
