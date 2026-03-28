"""KV cache compression using TurboQuant.

Wraps two TurboQuant instances to compress keys (with unbiased inner
product estimation for attention logits) and values (with lower MSE
for reconstruction quality).

Supports:
- Sliding window (residual buffer): keeps recent tokens in fp16 for
  higher accuracy on local context.
- GQA/MQA-aware configuration: auto-adjusts key bit allocation based
  on the GQA fan-out ratio to reduce amplified quantization error.

Reference: Section 4 of TurboQuant (arXiv:2504.19874)
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F

from .core import TurboQuant, TurboQuantOutput


class CompressedKV(NamedTuple):
    """Compressed key-value cache."""

    keys: TurboQuantOutput | None  # quantized old tokens, or None if all residual
    values: TurboQuantOutput | None  # quantized old tokens, or None if all residual
    residual_keys: torch.Tensor  # recent tokens kept in fp16
    residual_values: torch.Tensor  # recent tokens kept in fp16
    batch_size: int
    num_heads: int
    seq_len: int
    head_dim: int
    split_point: int  # where quantized ends and residual begins


class TurboQuantKVCache:
    """KV cache compression for transformer attention.

    Keys use unbiased=True for accurate attention logit estimation.
    Values use unbiased=False for lower reconstruction MSE.

    Args:
        head_dim: Dimension of each attention head.
        bit_width: Total bits per coordinate (used if key/value overrides not set).
        residual_length: Number of recent tokens to keep in fp16 (sliding window).
        key_bit_width: Override bit width for keys. Defaults to ``bit_width``.
        value_bit_width: Override bit width for values. Defaults to ``bit_width``.
        seed: Random seed.
    """

    def __init__(
        self,
        head_dim: int,
        bit_width: int = 3,
        residual_length: int = 128,
        key_bit_width: int | None = None,
        value_bit_width: int | None = None,
        seed: int = 0,
    ):
        self.head_dim = head_dim
        self.bit_width = bit_width
        self.residual_length = residual_length

        k_bits = key_bit_width if key_bit_width is not None else bit_width
        v_bits = value_bit_width if value_bit_width is not None else bit_width

        self.key_quantizer = TurboQuant(head_dim, k_bits, unbiased=True, seed=seed)
        self.value_quantizer = TurboQuant(head_dim, v_bits, unbiased=False, seed=seed + 100)

    @staticmethod
    def for_gqa(
        head_dim: int,
        num_kv_heads: int,
        num_query_heads: int,
        bit_width: int = 3,
        residual_length: int = 128,
        seed: int = 0,
    ) -> TurboQuantKVCache:
        """Factory for GQA models. Auto-adjusts key bits based on fan-out ratio.

        Higher GQA ratio means more query heads share each KV head, which
        amplifies quantization error. When the ratio exceeds 2, key precision
        is automatically bumped by 1 bit (up to 4-bit).

        Args:
            head_dim: Attention head dimension.
            num_kv_heads: Number of KV heads (e.g., 8 for Llama-3-8B).
            num_query_heads: Number of query heads (e.g., 32 for Llama-3-8B).
            bit_width: Base bit width.
            residual_length: Sliding window size.
            seed: Random seed.

        Returns:
            Configured TurboQuantKVCache instance.
        """
        gqa_ratio = num_query_heads // num_kv_heads

        # If GQA ratio > 2, bump key bits by 1 to reduce amplified error
        if gqa_ratio > 2:
            key_bits = min(bit_width + 1, 4)
            value_bits = bit_width
        else:
            key_bits = bit_width
            value_bits = bit_width

        return TurboQuantKVCache(
            head_dim=head_dim,
            bit_width=bit_width,
            key_bit_width=key_bits,
            value_bit_width=value_bits,
            residual_length=residual_length,
            seed=seed,
        )

    def to(self, device: torch.device) -> TurboQuantKVCache:
        """Move to device."""
        self.key_quantizer = self.key_quantizer.to(device)
        self.value_quantizer = self.value_quantizer.to(device)
        return self

    def compress(self, keys: torch.Tensor, values: torch.Tensor) -> CompressedKV:
        """Compress key and value tensors.

        Tokens beyond the residual window are quantized; the most recent
        ``residual_length`` tokens are kept in their original precision.

        Args:
            keys: Key tensor of shape (batch, heads, seq, head_dim).
            values: Value tensor of shape (batch, heads, seq, head_dim).

        Returns:
            CompressedKV with quantized old tokens and fp16/fp32 residual.
        """
        B, H, S, D = keys.shape
        if self.head_dim != D:
            raise ValueError(f"Expected head_dim {self.head_dim}, got {D}")

        if self.residual_length >= S:
            # All tokens fit in the residual buffer — no quantization needed
            return CompressedKV(
                keys=None,
                values=None,
                residual_keys=keys,
                residual_values=values,
                batch_size=B,
                num_heads=H,
                seq_len=S,
                head_dim=D,
                split_point=0,
            )

        split_point = S - self.residual_length

        # Older tokens → quantized
        old_keys = keys[:, :, :split_point, :]
        old_values = values[:, :, :split_point, :]
        k_compressed = self.key_quantizer.quantize(old_keys.reshape(-1, D))
        v_compressed = self.value_quantizer.quantize(old_values.reshape(-1, D))

        # Recent tokens → kept in original precision
        if self.residual_length > 0:
            recent_keys = keys[:, :, split_point:, :]
            recent_values = values[:, :, split_point:, :]
        else:
            recent_keys = keys.new_empty(B, H, 0, D)
            recent_values = values.new_empty(B, H, 0, D)

        return CompressedKV(
            keys=k_compressed,
            values=v_compressed,
            residual_keys=recent_keys,
            residual_values=recent_values,
            batch_size=B,
            num_heads=H,
            seq_len=S,
            head_dim=D,
            split_point=split_point,
        )

    def decompress_keys(self, compressed: CompressedKV) -> torch.Tensor:
        """Decompress keys back to full tensors.

        Args:
            compressed: CompressedKV output from compress().

        Returns:
            Keys tensor of shape (batch, heads, seq, head_dim).
        """
        B, H, D = compressed.batch_size, compressed.num_heads, compressed.head_dim
        sp = compressed.split_point

        if compressed.keys is not None:
            old_keys = self.key_quantizer.dequantize(compressed.keys).view(B, H, sp, D)
            if compressed.residual_keys.shape[2] > 0:
                return torch.cat([old_keys, compressed.residual_keys], dim=2)
            return old_keys
        return compressed.residual_keys

    def decompress_values(self, compressed: CompressedKV) -> torch.Tensor:
        """Decompress values back to full tensors.

        Args:
            compressed: CompressedKV output from compress().

        Returns:
            Values tensor of shape (batch, heads, seq, head_dim).
        """
        B, H, D = compressed.batch_size, compressed.num_heads, compressed.head_dim
        sp = compressed.split_point

        if compressed.values is not None:
            old_values = self.value_quantizer.dequantize(compressed.values).view(B, H, sp, D)
            if compressed.residual_values.shape[2] > 0:
                return torch.cat([old_values, compressed.residual_values], dim=2)
            return old_values
        return compressed.residual_values

    def attention(
        self,
        query: torch.Tensor,
        compressed: CompressedKV,
        scale: float | None = None,
    ) -> torch.Tensor:
        """Compute scaled dot-product attention with compressed KV cache.

        Args:
            query: Query tensor of shape (batch, heads, q_len, head_dim).
            compressed: CompressedKV from compress().
            scale: Attention scale factor. Defaults to 1/sqrt(head_dim).

        Returns:
            Attention output of shape (batch, heads, q_len, head_dim).
        """
        if scale is None:
            scale = self.head_dim**-0.5

        keys = self.decompress_keys(compressed)
        values = self.decompress_values(compressed)

        # Standard scaled dot-product attention
        # (B, H, Q, D) @ (B, H, D, S) -> (B, H, Q, S)
        attn_weights = torch.matmul(query, keys.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        # (B, H, Q, S) @ (B, H, S, D) -> (B, H, Q, D)
        return torch.matmul(attn_weights, values)

    def memory_savings(
        self, batch_size: int, num_heads: int, seq_len: int
    ) -> tuple[float, float, float]:
        """Report memory usage comparison.

        Accounts for the residual buffer: quantized tokens use compressed
        storage, while residual tokens use full float32.

        Args:
            batch_size: Batch size.
            num_heads: Number of attention heads.
            seq_len: Sequence length.

        Returns:
            Tuple of (original_mb, compressed_mb, ratio).
        """
        n_total = batch_size * num_heads * seq_len
        original_bytes = n_total * self.head_dim * 4 * 2  # K + V, float32

        residual_len = min(self.residual_length, seq_len)
        quantized_len = seq_len - residual_len

        n_quantized = batch_size * num_heads * quantized_len
        n_residual = batch_size * num_heads * residual_len

        compressed_bytes = (
            self.key_quantizer.memory_bytes(n_quantized)
            + self.value_quantizer.memory_bytes(n_quantized)
            + n_residual * self.head_dim * 4 * 2  # residual K + V in float32
        )

        original_mb = original_bytes / (1024 * 1024)
        compressed_mb = compressed_bytes / (1024 * 1024)
        ratio = original_bytes / max(compressed_bytes, 1)
        return original_mb, compressed_mb, ratio
