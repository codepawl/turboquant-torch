"""Tests for Lloyd-Max codebook computation and scalar quantization."""

import pytest
import torch

from turboquant.codebook import LloydMaxCodebook, get_codebook


class TestCodebook:
    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_codebook_level_count(self, bits):
        """Codebook has exactly 2^b levels."""
        cb = get_codebook(bits, 128)
        assert len(cb.centroids) == 2**bits
        assert len(cb.boundaries) == 2**bits - 1

    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_boundaries_sorted(self, bits):
        """Boundaries are strictly increasing."""
        cb = get_codebook(bits, 128)
        for i in range(len(cb.boundaries) - 1):
            assert cb.boundaries[i] < cb.boundaries[i + 1]

    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_centroids_sorted(self, bits):
        """Centroids are strictly increasing."""
        cb = get_codebook(bits, 128)
        for i in range(len(cb.centroids) - 1):
            assert cb.centroids[i] < cb.centroids[i + 1]

    def test_1bit_symmetry(self):
        """1-bit codebook is symmetric around 0."""
        cb = get_codebook(1, 128)
        assert abs(cb.centroids[0] + cb.centroids[1]) < 1e-6
        assert len(cb.boundaries) == 1
        assert abs(cb.boundaries[0]) < 1e-6

    def test_scaling_by_dim(self):
        """High-dim codebook is N(0,1) scaled by 1/sqrt(d)."""
        cb256 = get_codebook(2, 256)
        cb1024 = get_codebook(2, 1024)
        # Centroids should scale as 1/sqrt(d)
        ratio = cb256.centroids[-1] / cb1024.centroids[-1]
        expected = (1024 / 256) ** 0.5  # 2.0
        assert abs(ratio - expected) < 0.01


class TestLloydMaxCodebook:
    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_code_range(self, bits):
        """Codes are in range [0, 2^b)."""
        lm = LloydMaxCodebook(128, bits)
        x = torch.randn(100, 128) / (128**0.5)
        codes = lm.quantize(x)
        assert codes.min() >= 0
        assert codes.max() < 2**bits

    def test_roundtrip_shape(self):
        """Quantize/dequantize preserves shape."""
        lm = LloydMaxCodebook(128, 3)
        x = torch.randn(10, 128) / (128**0.5)
        codes = lm.quantize(x)
        x_hat = lm.dequantize(codes)
        assert x_hat.shape == x.shape

    def test_dequantize_values_are_centroids(self):
        """Dequantized values are centroid values."""
        lm = LloydMaxCodebook(128, 2)
        x = torch.randn(5, 128) / (128**0.5)
        codes = lm.quantize(x)
        x_hat = lm.dequantize(codes)
        centroids_set = set(lm.centroids.tolist())
        for val in x_hat.flatten().tolist():
            assert val in centroids_set

    def test_low_dim_uses_beta(self):
        """Dimensions < 64 use Beta distribution codebook."""
        cb_low = get_codebook(2, 8)
        cb_high = get_codebook(2, 128)
        # They should differ (different distributions)
        assert not all(
            abs(a - b) < 1e-6 for a, b in zip(cb_low.centroids, cb_high.centroids, strict=True)
        )
