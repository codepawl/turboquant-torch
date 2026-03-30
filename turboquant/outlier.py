"""Outlier channel detection and routing.

Identifies high-magnitude channels in KV cache tensors and routes
them to a sparse high-precision store, bypassing quantization.
The remaining channels are quantized normally with TurboQuant.

This hybrid approach combines KVQuant's Dense-and-Sparse strategy
with TurboQuant's optimal quantization on the bulk data.

Reference:
  - KVQuant Section 3.2 (Dense-and-Sparse Quantization)
  - GEAR Section 3 (Low-rank + Sparse correction)
  - TurboQuant Section 5 (Outlier channel splitting)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class OutlierSplit:
    """Result of splitting a tensor into bulk + outlier channels.

    Args:
        bulk: Tensor with outlier channels removed (..., dim - n_outliers).
        outlier_values: Outlier channel values in original precision (..., n_outliers).
        outlier_indices: Channel indices that were extracted (n_outliers,).
        original_dim: Original last-dimension size before splitting.
    """

    bulk: torch.Tensor
    outlier_values: torch.Tensor
    outlier_indices: torch.Tensor
    original_dim: int


def detect_outlier_channels(
    x: torch.Tensor,
    n_outliers: int = 8,
    method: str = "magnitude",
) -> torch.Tensor:
    """Detect outlier channels by statistical analysis.

    Args:
        x: Input tensor of shape (..., dim). Common shapes:
            (batch, heads, seq, dim) for KV tensors, or (n, dim).
        n_outliers: Number of outlier channels to identify.
        method: Detection method.
            "magnitude" -- channels with largest average absolute value.
            "variance" -- channels with largest variance.
            "range" -- channels with largest max-min range.

    Returns:
        Sorted indices of outlier channels, shape (n_outliers,).

    Raises:
        ValueError: If method is not recognized.
    """
    flat = x.reshape(-1, x.shape[-1])
    dim = flat.shape[-1]
    n_outliers = min(n_outliers, dim // 4)  # cap at 25% of channels

    if method == "magnitude":
        channel_scores = flat.abs().mean(dim=0)
    elif method == "variance":
        channel_scores = flat.var(dim=0)
    elif method == "range":
        channel_scores = flat.max(dim=0).values - flat.min(dim=0).values
    else:
        raise ValueError(f"Unknown outlier detection method: {method!r}")

    _, indices = torch.topk(channel_scores, n_outliers)
    return indices.sort().values


def split_outliers(
    x: torch.Tensor,
    outlier_indices: torch.Tensor,
) -> OutlierSplit:
    """Split tensor into bulk channels and outlier channels.

    Args:
        x: Input tensor (..., dim).
        outlier_indices: Channel indices to extract as outliers.

    Returns:
        OutlierSplit with bulk and outlier tensors.
    """
    dim = x.shape[-1]
    outlier_values = x[..., outlier_indices]

    mask = torch.ones(dim, dtype=torch.bool, device=x.device)
    mask[outlier_indices] = False
    bulk_indices = torch.arange(dim, device=x.device)[mask]
    bulk = x[..., bulk_indices]

    return OutlierSplit(
        bulk=bulk,
        outlier_values=outlier_values,
        outlier_indices=outlier_indices,
        original_dim=dim,
    )


def merge_outliers(
    bulk: torch.Tensor,
    outlier_split: OutlierSplit,
) -> torch.Tensor:
    """Merge bulk (possibly dequantized) tensor with outlier channels.

    Args:
        bulk: Reconstructed bulk tensor (..., dim - n_outliers).
        outlier_split: Original split containing outlier values and indices.

    Returns:
        Merged tensor (..., original_dim) with outliers restored.
    """
    device = bulk.device
    dtype = bulk.dtype
    dim = outlier_split.original_dim

    shape = list(bulk.shape)
    shape[-1] = dim
    merged = torch.zeros(shape, dtype=dtype, device=device)

    mask = torch.ones(dim, dtype=torch.bool, device=device)
    mask[outlier_split.outlier_indices] = False
    bulk_indices = torch.arange(dim, device=device)[mask]

    merged[..., bulk_indices] = bulk
    merged[..., outlier_split.outlier_indices] = outlier_split.outlier_values.to(dtype)

    return merged
