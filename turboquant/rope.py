"""Rotary Position Embedding utilities for Pre-RoPE quantization.

When pre_rope=True, keys are quantized BEFORE RoPE is applied.
RoPE is then applied at attention time to both query and dequantized key.

This preserves the geometric structure that TurboQuant relies on,
since RoPE rotations can distort the Beta distribution of coordinates.

Reference: KVQuant (arXiv:2401.18079) Section 3.1
"""

from __future__ import annotations

import torch


def compute_rope_frequencies(
    head_dim: int,
    max_seq_len: int = 8192,
    base: float = 10000.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Compute RoPE frequency tensor.

    Args:
        head_dim: Dimension of attention head (must be even).
        max_seq_len: Maximum sequence length to precompute.
        base: RoPE base frequency.
        device: Torch device.

    Returns:
        Frequency tensor of shape (max_seq_len, head_dim // 2).
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    device = device or torch.device("cpu")
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq_len, device=device).float()
    return torch.outer(t, inv_freq)


def apply_rope(
    x: torch.Tensor,
    freqs: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply Rotary Position Embedding to input tensor.

    Args:
        x: Input tensor of shape (batch, heads, seq_len, head_dim).
        freqs: Frequency tensor from :func:`compute_rope_frequencies`.
        positions: Optional position indices of shape ``(seq_len,)`` or
            ``(batch, seq_len)``. If *None*, uses ``0 .. seq_len-1``.

    Returns:
        Tensor with RoPE applied, same shape as input.
    """
    seq_len = x.shape[2]
    dim = x.shape[3]

    f = freqs[positions] if positions is not None else freqs[:seq_len]

    # Reshape for broadcasting: (..., 1, seq_len, dim//2)
    while f.dim() < x.dim():
        f = f.unsqueeze(0)

    cos_f = torch.cos(f)
    sin_f = torch.sin(f)

    x1 = x[..., : dim // 2]
    x2 = x[..., dim // 2 :]

    out1 = x1 * cos_f - x2 * sin_f
    out2 = x1 * sin_f + x2 * cos_f

    return torch.cat([out1, out2], dim=-1)
