"""KV cache compression using TurboQuant.

Wraps two TurboQuant instances to compress keys (with unbiased inner
product estimation for attention logits) and values (with lower MSE
for reconstruction quality).

Reference: Section 4 of TurboQuant (arXiv:2504.19874)
"""

from typing import NamedTuple

import torch
import torch.nn.functional as F

from .core import TurboQuant, TurboQuantOutput


class CompressedKV(NamedTuple):
    """Compressed key-value cache."""

    keys: TurboQuantOutput
    values: TurboQuantOutput
    batch_size: int
    num_heads: int
    seq_len: int
    head_dim: int


class TurboQuantKVCache:
    """KV cache compression for transformer attention.

    Keys use unbiased=True for accurate attention logit estimation.
    Values use unbiased=False for lower reconstruction MSE.

    Args:
        head_dim: Dimension of each attention head.
        bit_width: Total bits per coordinate.
        seed: Random seed.
    """

    def __init__(self, head_dim: int, bit_width: int = 3, seed: int = 0):
        self.head_dim = head_dim
        self.bit_width = bit_width
        self.key_quantizer = TurboQuant(head_dim, bit_width, unbiased=True, seed=seed)
        self.value_quantizer = TurboQuant(head_dim, bit_width, unbiased=False, seed=seed + 100)

    def to(self, device: torch.device) -> "TurboQuantKVCache":
        """Move to device."""
        self.key_quantizer = self.key_quantizer.to(device)
        self.value_quantizer = self.value_quantizer.to(device)
        return self

    def compress(self, keys: torch.Tensor, values: torch.Tensor) -> CompressedKV:
        """Compress key and value tensors.

        Args:
            keys: Key tensor of shape (batch, heads, seq, head_dim).
            values: Value tensor of shape (batch, heads, seq, head_dim).

        Returns:
            CompressedKV with quantized keys and values.
        """
        B, H, S, D = keys.shape
        if self.head_dim != D:
            raise ValueError(f"Expected head_dim {self.head_dim}, got {D}")

        # Flatten to (B*H*S, D) for quantization
        k_flat = keys.reshape(-1, D)
        v_flat = values.reshape(-1, D)

        k_compressed = self.key_quantizer.quantize(k_flat)
        v_compressed = self.value_quantizer.quantize(v_flat)

        return CompressedKV(
            keys=k_compressed,
            values=v_compressed,
            batch_size=B,
            num_heads=H,
            seq_len=S,
            head_dim=D,
        )

    def decompress_keys(self, compressed: CompressedKV) -> torch.Tensor:
        """Decompress keys back to full tensors.

        Args:
            compressed: CompressedKV output from compress().

        Returns:
            Keys tensor of shape (batch, heads, seq, head_dim).
        """
        k_flat = self.key_quantizer.dequantize(compressed.keys)
        return k_flat.view(
            compressed.batch_size,
            compressed.num_heads,
            compressed.seq_len,
            compressed.head_dim,
        )

    def decompress_values(self, compressed: CompressedKV) -> torch.Tensor:
        """Decompress values back to full tensors.

        Args:
            compressed: CompressedKV output from compress().

        Returns:
            Values tensor of shape (batch, heads, seq, head_dim).
        """
        v_flat = self.value_quantizer.dequantize(compressed.values)
        return v_flat.view(
            compressed.batch_size,
            compressed.num_heads,
            compressed.seq_len,
            compressed.head_dim,
        )

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

        Args:
            batch_size: Batch size.
            num_heads: Number of attention heads.
            seq_len: Sequence length.

        Returns:
            Tuple of (original_mb, compressed_mb, ratio).
        """
        n_vectors = batch_size * num_heads * seq_len
        original_bytes = n_vectors * self.head_dim * 4 * 2  # K + V, float32
        compressed_bytes = self.key_quantizer.memory_bytes(
            n_vectors
        ) + self.value_quantizer.memory_bytes(n_vectors)
        original_mb = original_bytes / (1024 * 1024)
        compressed_mb = compressed_bytes / (1024 * 1024)
        ratio = original_bytes / max(compressed_bytes, 1)
        return original_mb, compressed_mb, ratio
