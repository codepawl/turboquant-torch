"""Lloyd-Max optimal scalar quantizer codebooks.

Precomputes MSE-optimal (Lloyd-Max) quantizer boundaries and centroids for
the standard normal distribution N(0,1). At runtime, these are scaled by
1/sqrt(d) for the Gaussian approximation of Beta(d/2, d/2) coordinates
after randomized Hadamard rotation.

For low dimensions (d < 64), exact codebooks for Beta(d/2, d/2) are
computed using scipy.

Reference: Section 3.1, Lemma 3.2 of TurboQuant (arXiv:2504.19874)
"""

from typing import NamedTuple

import numpy as np
import torch
from scipy import integrate
from scipy.stats import beta as beta_dist
from scipy.stats import norm


class Codebook(NamedTuple):
    """Quantizer codebook with decision boundaries and reconstruction centroids."""

    boundaries: np.ndarray  # shape (2^b - 1,)
    centroids: np.ndarray  # shape (2^b,)


def _lloyd_max_normal(bits: int, max_iter: int = 200, tol: float = 1e-12) -> Codebook:
    """Compute Lloyd-Max optimal quantizer for N(0,1).

    Args:
        bits: Number of quantization bits (1-8).
        max_iter: Maximum Lloyd-Max iterations.
        tol: Convergence tolerance.

    Returns:
        Codebook with boundaries and centroids.
    """
    n_levels = 1 << bits

    # Initialize centroids uniformly in [-3, 3]
    centroids = np.linspace(-3, 3, n_levels)

    for _ in range(max_iter):
        # Boundaries are midpoints of adjacent centroids
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0

        # Update centroids: conditional expectation in each bin
        new_centroids = np.zeros(n_levels)
        edges = np.concatenate([[-np.inf], boundaries, [np.inf]])

        for i in range(n_levels):
            lo, hi = edges[i], edges[i + 1]
            # E[X | lo < X < hi] for X ~ N(0,1)
            prob = norm.cdf(hi) - norm.cdf(lo)
            if prob < 1e-15:
                new_centroids[i] = (
                    (lo + hi) / 2 if np.isfinite(lo) and np.isfinite(hi) else centroids[i]
                )
            else:
                # E[X | lo < X < hi] = (phi(lo) - phi(hi)) / (Phi(hi) - Phi(lo))
                new_centroids[i] = (norm.pdf(lo) - norm.pdf(hi)) / prob

        if np.max(np.abs(new_centroids - centroids)) < tol:
            centroids = new_centroids
            break
        centroids = new_centroids

    boundaries = (centroids[:-1] + centroids[1:]) / 2.0
    return Codebook(boundaries=boundaries, centroids=centroids)


def _lloyd_max_beta(bits: int, dim: int, max_iter: int = 200, tol: float = 1e-12) -> Codebook:
    """Compute Lloyd-Max optimal quantizer for Beta(d/2, d/2) on [-1, 1].

    Used for low dimensions where Gaussian approximation is inaccurate.

    Args:
        bits: Number of quantization bits.
        dim: Vector dimension (determines Beta shape parameters).
        max_iter: Maximum Lloyd-Max iterations.
        tol: Convergence tolerance.

    Returns:
        Codebook with boundaries and centroids.
    """
    n_levels = 1 << bits
    a = dim / 2.0

    # Beta(a, a) on [0, 1], then shift to [-1, 1]: X = 2*B - 1
    rv = beta_dist(a, a)

    # Initialize centroids uniformly in [-1, 1]
    centroids = np.linspace(-0.9, 0.9, n_levels)

    for _ in range(max_iter):
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0
        new_centroids = np.zeros(n_levels)
        edges = np.concatenate([[-1.0], boundaries, [1.0]])

        for i in range(n_levels):
            lo, hi = edges[i], edges[i + 1]
            # Transform to [0, 1] for Beta CDF
            lo_01, hi_01 = (lo + 1) / 2, (hi + 1) / 2
            lo_01 = max(0.0, min(1.0, lo_01))
            hi_01 = max(0.0, min(1.0, hi_01))

            prob = rv.cdf(hi_01) - rv.cdf(lo_01)
            if prob < 1e-15:
                new_centroids[i] = (lo + hi) / 2
            else:
                # E[X | lo < X < hi] where X = 2*B - 1, B ~ Beta(a, a)
                # = 2 * E[B | lo_01 < B < hi_01] - 1
                def integrand(b: float) -> float:
                    return float(b * rv.pdf(b))

                e_b, _ = integrate.quad(integrand, lo_01, hi_01)
                new_centroids[i] = 2 * (e_b / prob) - 1

        if np.max(np.abs(new_centroids - centroids)) < tol:
            centroids = new_centroids
            break
        centroids = new_centroids

    boundaries = (centroids[:-1] + centroids[1:]) / 2.0
    return Codebook(boundaries=boundaries, centroids=centroids)


# Precompute N(0,1) codebooks for bit widths 1-4
_NORMAL_CODEBOOKS: dict[int, Codebook] = {}
for _b in range(1, 5):
    _NORMAL_CODEBOOKS[_b] = _lloyd_max_normal(_b)


# Cache for Beta codebooks (keyed by (bits, dim))
_BETA_CODEBOOK_CACHE: dict[tuple[int, int], Codebook] = {}


def get_codebook(bits: int, dim: int) -> Codebook:
    """Get the Lloyd-Max codebook for a given bit width and dimension.

    For dim >= 64, uses the Gaussian approximation (precomputed N(0,1)
    codebook scaled by 1/sqrt(d)). For dim < 64, computes exact
    codebook for Beta(d/2, d/2).

    Args:
        bits: Quantization bit width (1-4).
        dim: Vector dimension.

    Returns:
        Codebook with appropriately scaled boundaries and centroids.
    """
    if bits < 1 or bits > 8:
        raise ValueError(f"Bit width must be 1-8, got {bits}")

    if dim >= 64:
        if bits not in _NORMAL_CODEBOOKS:
            _NORMAL_CODEBOOKS[bits] = _lloyd_max_normal(bits)
        cb = _NORMAL_CODEBOOKS[bits]
        scale = 1.0 / np.sqrt(dim)
        return Codebook(
            boundaries=cb.boundaries * scale,
            centroids=cb.centroids * scale,
        )
    key = (bits, dim)
    if key not in _BETA_CODEBOOK_CACHE:
        _BETA_CODEBOOK_CACHE[key] = _lloyd_max_beta(bits, dim)
    return _BETA_CODEBOOK_CACHE[key]


class LloydMaxCodebook:
    """Scalar quantizer using precomputed Lloyd-Max codebooks.

    Args:
        bits: Number of quantization bits per coordinate.
        dim: Vector dimension (used to select Gaussian vs Beta codebook).
    """

    def __init__(self, bits: int, dim: int):
        self.bits = bits
        self.dim = dim
        cb = get_codebook(bits, dim)
        self._boundaries = torch.from_numpy(cb.boundaries).float()
        self._centroids = torch.from_numpy(cb.centroids).float()

    def to(self, device: torch.device) -> "LloydMaxCodebook":
        """Move codebook tensors to device."""
        self._boundaries = self._boundaries.to(device)
        self._centroids = self._centroids.to(device)
        return self

    @property
    def boundaries(self) -> torch.Tensor:
        return self._boundaries

    @property
    def centroids(self) -> torch.Tensor:
        return self._centroids

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize each coordinate to its nearest codebook index.

        Args:
            x: Input tensor of shape (..., d).

        Returns:
            Integer codes of shape (..., d) in range [0, 2^bits).
        """
        # searchsorted expects sorted boundaries; returns index i such that
        # boundaries[i-1] <= x < boundaries[i]
        boundaries = self._boundaries.to(x.device)
        return torch.searchsorted(boundaries, x)

    def dequantize(self, codes: torch.Tensor) -> torch.Tensor:
        """Map codes back to centroid values.

        Args:
            codes: Integer codes of shape (..., d).

        Returns:
            Reconstructed values of shape (..., d).
        """
        centroids = self._centroids.to(codes.device)
        return centroids[codes]
