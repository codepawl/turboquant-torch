"""Tests for adaptive per-layer bit allocation."""

import pytest
import torch

from turboquant.adaptive import AdaptiveKVCache, gradient_allocation, uniform_allocation


class TestAllocation:
    def test_uniform(self):
        """Uniform allocation returns identical values."""
        assert uniform_allocation(32, 3) == [3] * 32

    def test_gradient_linear(self):
        """Linear gradient increases bits from early to late layers."""
        bits = gradient_allocation(12, min_bits=2, max_bits=4, strategy="linear")
        assert len(bits) == 12
        assert bits[0] == 2
        assert bits[-1] == 4
        for i in range(len(bits) - 1):
            assert bits[i] <= bits[i + 1]

    def test_gradient_step(self):
        """Step allocation uses 3 tiers."""
        bits = gradient_allocation(12, min_bits=2, max_bits=4, strategy="step")
        assert len(bits) == 12
        assert bits[0] == 2
        assert bits[-1] == 4

    def test_single_layer(self):
        """Single layer gets min_bits."""
        bits = gradient_allocation(1, min_bits=2, max_bits=4)
        assert bits == [2]

    def test_invalid_strategy_raises(self):
        """Unknown strategy raises ValueError."""
        with pytest.raises(ValueError, match="Unknown"):
            gradient_allocation(10, strategy="exotic")  # type: ignore[arg-type]


class TestAdaptiveKVCache:
    def test_create(self):
        """Creates correct number of per-layer caches."""
        cache = AdaptiveKVCache(head_dim=128, layer_bits=[2, 3, 3, 4])
        assert cache.n_layers == 4
        assert len(cache.caches) == 4

    def test_compress_decompress_per_layer(self):
        """Each layer compresses and decompresses independently."""
        cache = AdaptiveKVCache(
            head_dim=64,
            layer_bits=[2, 3, 4],
            residual_length=0,
        )

        keys = torch.randn(1, 2, 16, 64)
        values = torch.randn(1, 2, 16, 64)

        for layer_idx in range(3):
            compressed = cache.compress_layer(layer_idx, keys, values)
            k_hat = cache.decompress_layer_keys(layer_idx, compressed)
            assert k_hat.shape == keys.shape

    def test_higher_bits_lower_mse(self):
        """Layers with more bits should have lower reconstruction MSE."""
        cache = AdaptiveKVCache(
            head_dim=128,
            layer_bits=[2, 4],
            residual_length=0,
        )

        keys = torch.randn(1, 4, 32, 128)
        values = torch.randn(1, 4, 32, 128)

        mses = []
        for layer_idx in range(2):
            compressed = cache.compress_layer(layer_idx, keys, values)
            k_hat = cache.decompress_layer_keys(layer_idx, compressed)
            mse = ((keys - k_hat) ** 2).mean().item()
            mses.append(mse)

        assert mses[0] > mses[1]  # 2-bit > 4-bit MSE

    def test_summary(self):
        """Summary reports correct layer count and average."""
        cache = AdaptiveKVCache(head_dim=128, layer_bits=[2, 3, 3, 4])
        s = cache.summary()
        assert "4 layers" in s
        assert "3.0 bits/layer" in s

    def test_attention_layer(self):
        """Attention through per-layer cache produces valid output."""
        cache = AdaptiveKVCache(
            head_dim=64,
            layer_bits=[3, 3],
            residual_length=0,
        )

        keys = torch.randn(1, 2, 16, 64)
        values = torch.randn(1, 2, 16, 64)
        query = torch.randn(1, 2, 1, 64)

        compressed = cache.compress_layer(0, keys, values)
        out = cache.attention_layer(0, query, compressed)
        assert out.shape == (1, 2, 1, 64)
        assert not torch.isnan(out).any()
