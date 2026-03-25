"""Fast Walsh-Hadamard Transform and randomized rotation.

The Randomized Hadamard Transform (RHT) is a data-oblivious rotation that
makes any unit vector's coordinates approximately i.i.d. Gaussian. This is
the key enabling primitive for TurboQuant's MSE-optimal scalar quantizer.

Reference: Section 3.1 of TurboQuant (arXiv:2504.19874)
"""

import math

import torch
import torch.nn as nn


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


def fwht(x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Fast Walsh-Hadamard Transform (in-place butterfly algorithm).

    Operates on the last dimension. O(d log d) complexity.

    Args:
        x: Input tensor of shape (..., d) where d must be a power of 2.
        normalize: If True, multiply by 1/sqrt(d) for orthonormal transform.

    Returns:
        Transformed tensor of the same shape.
    """
    d = x.shape[-1]
    if d <= 0 or (d & (d - 1)) != 0:
        raise ValueError(f"Last dim must be power of 2, got {d}")

    result = x.clone()
    h = 1
    while h < d:
        # Reshape to isolate pairs at distance h
        result = result.view(*result.shape[:-1], -1, 2 * h)
        left = result[..., :h]
        right = result[..., h:]
        result = torch.cat([left + right, left - right], dim=-1)
        result = result.view(*x.shape)
        h *= 2

    if normalize:
        result = result / math.sqrt(d)
    return result


class RandomizedHadamardTransform(nn.Module):
    signs: torch.Tensor

    """Randomized Hadamard Transform: random sign flips followed by FWHT.

    After applying RHT to a unit vector of dimension d, each coordinate
    is approximately distributed as N(0, 1/d) for large d, or exactly
    Beta(d/2, d/2) scaled to [-1, 1].

    Args:
        dim: Input dimension. Will be padded to next power of 2 if needed.
        seed: Random seed for reproducible sign generation.
    """

    def __init__(self, dim: int, seed: int = 0):
        super().__init__()
        self.original_dim = dim
        self.padded_dim = _next_power_of_2(dim)

        # Generate random Rademacher signs (±1)
        gen = torch.Generator()
        gen.manual_seed(seed)
        signs = torch.randint(0, 2, (self.padded_dim,), generator=gen) * 2 - 1
        self.register_buffer("signs", signs.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply randomized Hadamard transform.

        Args:
            x: Input tensor of shape (..., dim).

        Returns:
            Rotated tensor of shape (..., padded_dim).
        """
        if x.shape[-1] < self.padded_dim:
            padding = torch.zeros(
                *x.shape[:-1],
                self.padded_dim - x.shape[-1],
                device=x.device,
                dtype=x.dtype,
            )
            x = torch.cat([x, padding], dim=-1)

        # Sign flip then Hadamard
        x = x * self.signs
        return fwht(x, normalize=True)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Apply inverse randomized Hadamard transform.

        Since H is symmetric and orthogonal (H = H^T = H^{-1} with
        normalization), and sign flips are self-inverse, the inverse
        is: FWHT then sign flip, then truncate to original dim.

        Args:
            x: Rotated tensor of shape (..., padded_dim).

        Returns:
            Reconstructed tensor of shape (..., original_dim).
        """
        x = fwht(x, normalize=True)
        x = x * self.signs
        return x[..., : self.original_dim]
