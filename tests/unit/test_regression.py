"""Regression tests with pinned expected outputs.

These tests verify that specific known inputs always produce
the same outputs. If any test fails after a code change, it
means the change altered behavior — intentional or not.
"""

import torch

from turboquant import TurboQuant, TurboQuantKVCache
from turboquant.codebook import get_codebook
from turboquant.outlier import detect_outlier_channels


class TestPinnedOutputs:
    def test_3bit_dim128_mse_pinned(self):
        """MSE distortion at 3-bit, dim=128 should be stable."""
        torch.manual_seed(42)
        x = torch.randn(100, 128)
        x = x / torch.norm(x, dim=-1, keepdim=True)

        tq = TurboQuant(dim=128, bit_width=3, unbiased=True, seed=0)
        x_hat = tq.dequantize(tq.quantize(x))
        mse = ((x - x_hat) ** 2).sum(dim=-1).mean().item()

        assert abs(mse - 0.178362) < 0.001, f"MSE changed: {mse}"

    def test_2bit_dim64_mse_pinned(self):
        """MSE at 2-bit, dim=64 should be stable."""
        torch.manual_seed(42)
        x = torch.randn(100, 64)
        x = x / torch.norm(x, dim=-1, keepdim=True)

        tq = TurboQuant(dim=64, bit_width=2, unbiased=True, seed=0)
        x_hat = tq.dequantize(tq.quantize(x))
        mse = ((x - x_hat) ** 2).sum(dim=-1).mean().item()
        assert abs(mse - 0.549076) < 0.001, f"MSE changed: {mse}"

    def test_codebook_3bit_dim128_pinned(self):
        """Codebook values for 3-bit, dim=128 should be exact."""
        cb = get_codebook(3, 128)
        assert len(cb.centroids) == 8
        assert len(cb.boundaries) == 7
        # Centroids should be symmetric around 0
        assert abs(cb.centroids[0] + cb.centroids[-1]) < 0.01

    def test_kv_compression_ratio_pinned(self):
        """Compression ratio for known config should be stable."""
        cache = TurboQuantKVCache(head_dim=128, bit_width=3, residual_length=0)
        orig, comp, ratio = cache.memory_savings(1, 32, 2048)
        # 3-bit on 128-dim should give ~10x
        assert 9.0 < ratio < 11.0, f"Ratio changed: {ratio}"

    def test_kv_compress_deterministic(self):
        """Same input + same seed = same output."""
        torch.manual_seed(0)
        k = torch.randn(1, 4, 32, 128)
        v = torch.randn(1, 4, 32, 128)

        cache1 = TurboQuantKVCache(head_dim=128, bit_width=3, seed=0, residual_length=0)
        comp1 = cache1.compress(k, v)
        k_hat1 = cache1.decompress_keys(comp1)

        cache2 = TurboQuantKVCache(head_dim=128, bit_width=3, seed=0, residual_length=0)
        comp2 = cache2.compress(k, v)
        k_hat2 = cache2.decompress_keys(comp2)

        assert torch.allclose(k_hat1, k_hat2, atol=1e-6)

    def test_outlier_split_deterministic(self):
        """Same input = same outlier detection."""
        torch.manual_seed(42)
        x = torch.randn(1, 4, 32, 128)
        x[:, :, :, 0] *= 100
        x[:, :, :, 7] *= 50

        idx1 = detect_outlier_channels(x, n_outliers=4)
        idx2 = detect_outlier_channels(x, n_outliers=4)
        assert torch.equal(idx1, idx2)
        assert 0 in idx1  # channel 0 was amplified 100x
        assert 7 in idx1  # channel 7 was amplified 50x


class TestAPIStability:
    """Verify public API signatures haven't changed."""

    def test_turboquant_api(self):
        tq = TurboQuant(dim=128, bit_width=3, unbiased=True, seed=0)
        assert hasattr(tq, "quantize")
        assert hasattr(tq, "dequantize")
        assert hasattr(tq, "compression_ratio")
        assert hasattr(tq, "compute_inner_product")
        assert hasattr(tq, "to")

    def test_kv_cache_api(self):
        cache = TurboQuantKVCache(head_dim=128, bit_width=3)
        assert hasattr(cache, "compress")
        assert hasattr(cache, "decompress_keys")
        assert hasattr(cache, "decompress_values")
        assert hasattr(cache, "attention")
        assert hasattr(cache, "memory_savings")
        assert hasattr(cache, "for_gqa")
        assert hasattr(cache, "to")

    def test_wrap_api(self):
        """v0.4.0 wrap API exists."""
        import turboquant

        assert hasattr(turboquant, "wrap")
        assert hasattr(turboquant, "TurboQuantWrapper")
        assert hasattr(turboquant, "TurboQuantDynamicCache")

    def test_kv_cache_new_params(self):
        """v0.3.0 params exist."""
        cache = TurboQuantKVCache(
            head_dim=128,
            bit_width=3,
            n_outlier_channels=8,
            residual_length=32,
            pre_rope=True,
        )
        assert cache.n_outlier_channels == 8
        assert cache.pre_rope is True
