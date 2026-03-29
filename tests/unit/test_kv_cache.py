"""Tests for TurboQuantKVCache."""

import torch

from turboquant.kv_cache import TurboQuantKVCache


class TestKVCache:
    def test_compress_decompress_shape(self):
        """Compress/decompress preserves KV shape."""
        B, H, S, D = 2, 4, 32, 64
        cache = TurboQuantKVCache(head_dim=D, bit_width=3)
        keys = torch.randn(B, H, S, D)
        values = torch.randn(B, H, S, D)
        compressed = cache.compress(keys, values)
        k_hat = cache.decompress_keys(compressed)
        v_hat = cache.decompress_values(compressed)
        assert k_hat.shape == keys.shape
        assert v_hat.shape == values.shape

    def test_attention_output_shape(self):
        """Attention with compressed KV produces correct output shape."""
        B, H, S, D = 1, 2, 16, 64
        cache = TurboQuantKVCache(head_dim=D, bit_width=3)
        keys = torch.randn(B, H, S, D)
        values = torch.randn(B, H, S, D)
        query = torch.randn(B, H, 1, D)
        compressed = cache.compress(keys, values)
        out = cache.attention(query, compressed)
        assert out.shape == (B, H, 1, D)

    def test_attention_reasonable_output(self):
        """Attention output should not be all zeros or NaN."""
        B, H, S, D = 1, 2, 16, 64
        cache = TurboQuantKVCache(head_dim=D, bit_width=3)
        keys = torch.randn(B, H, S, D)
        values = torch.randn(B, H, S, D)
        query = torch.randn(B, H, 1, D)
        compressed = cache.compress(keys, values)
        out = cache.attention(query, compressed)
        assert not torch.isnan(out).any()
        assert not torch.all(out == 0)

    def test_memory_savings(self):
        """Memory savings ratio should be > 1."""
        cache = TurboQuantKVCache(head_dim=128, bit_width=3)
        orig, comp, ratio = cache.memory_savings(2, 32, 2048)
        assert ratio > 1
        assert orig > comp

    def test_key_uses_unbiased(self):
        """Key quantizer should use unbiased mode."""
        cache = TurboQuantKVCache(head_dim=64, bit_width=3)
        assert cache.key_quantizer.unbiased is True

    def test_value_uses_biased(self):
        """Value quantizer should use biased (MSE-only) mode."""
        cache = TurboQuantKVCache(head_dim=64, bit_width=3)
        assert cache.value_quantizer.unbiased is False
