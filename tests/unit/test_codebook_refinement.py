"""Tests for improved codebook accuracy at typical LLM dimensions."""

import time

import numpy as np
import pytest
import torch

from turboquant.codebook import get_codebook
from turboquant.mse_quantizer import TurboQuantMSE


class TestCodebookRefinement:
    def test_common_dims_use_beta(self):
        """Dimensions < 256 should use exact Beta codebooks."""
        for dim in [64, 96, 128]:
            cb = get_codebook(3, dim)
            # Beta codebook centroids should be in [-1, 1]
            assert cb.centroids.max() <= 1.0
            assert cb.centroids.min() >= -1.0

    def test_high_dim_uses_gaussian(self):
        """Dimensions >= 256 should use scaled Gaussian codebooks."""
        cb = get_codebook(3, 512)
        # Gaussian codebook scaled by 1/sqrt(512) ~ 0.044
        assert cb.centroids.max() < 0.5

    def test_precomputed_matches_computed(self):
        """Precomputed codebooks should match freshly computed ones."""
        from turboquant.codebook import _lloyd_max_beta
        from turboquant.codebook_data import PRECOMPUTED_BETA_CODEBOOKS

        # Check a few entries
        items = list(PRECOMPUTED_BETA_CODEBOOKS.items())[:4]
        for (bits, dim), data in items:
            fresh = _lloyd_max_beta(bits, dim)
            assert np.allclose(fresh.boundaries, data["boundaries"], atol=1e-6)
            assert np.allclose(fresh.centroids, data["centroids"], atol=1e-6)

    def test_precomputed_covers_common_dims(self):
        """Precomputed data should cover typical LLM head dimensions."""
        from turboquant.codebook_data import PRECOMPUTED_BETA_CODEBOOKS

        for dim in [64, 96, 128]:
            for bits in [1, 2, 3, 4]:
                assert (bits, dim) in PRECOMPUTED_BETA_CODEBOOKS

    @pytest.mark.parametrize("dim", [64, 128])
    def test_quantization_reasonable_mse(self, dim):
        """Beta codebook should give reasonable MSE at dim < 256."""
        torch.manual_seed(42)
        n = 500
        x = torch.randn(n, dim)
        x = x / torch.norm(x, dim=-1, keepdim=True)

        mse_quant = TurboQuantMSE(dim=dim, bit_width=3, seed=42)
        out = mse_quant.quantize(x)
        x_hat = mse_quant.dequantize(out)
        mse = ((x - x_hat) ** 2).mean().item()

        # MSE should be reasonable (< 0.1 for 3-bit on unit vectors)
        assert mse < 0.1, f"MSE too high at dim={dim}: {mse}"

    def test_codebook_transition_at_256(self):
        """Codebook type should switch at dim=256."""
        # dim=255 → Beta codebook (centroids in [-1, 1])
        cb_low = get_codebook(3, 255)
        assert cb_low.centroids.max() <= 1.0

        # dim=256 → Gaussian codebook (scaled by 1/sqrt(256) = 0.0625)
        cb_high = get_codebook(3, 256)
        # Gaussian centroids scaled down, max should be much less than 1
        assert cb_high.centroids.max() < 0.3

    def test_no_import_slowdown(self):
        """Importing codebook_data should be fast (precomputed, no scipy)."""
        start = time.time()
        from turboquant import codebook_data  # noqa: F811

        _ = codebook_data.PRECOMPUTED_BETA_CODEBOOKS
        elapsed = time.time() - start
        assert elapsed < 0.5, f"Import took {elapsed:.2f}s, too slow"
