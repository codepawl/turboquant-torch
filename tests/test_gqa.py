"""Tests for GQA/MQA-aware configuration in TurboQuantKVCache."""

import torch

from turboquant.kv_cache import TurboQuantKVCache


class TestGQA:
    def test_for_gqa_bumps_key_bits(self):
        """GQA ratio > 2 should bump key bits."""
        cache = TurboQuantKVCache.for_gqa(
            head_dim=128, num_kv_heads=8, num_query_heads=32, bit_width=3
        )
        assert cache.key_quantizer.bit_width == 4  # bumped from 3
        assert cache.value_quantizer.bit_width == 3  # unchanged

    def test_for_gqa_no_bump_low_ratio(self):
        """GQA ratio <= 2 should not bump."""
        cache = TurboQuantKVCache.for_gqa(
            head_dim=128, num_kv_heads=16, num_query_heads=32, bit_width=3
        )
        assert cache.key_quantizer.bit_width == 3
        assert cache.value_quantizer.bit_width == 3

    def test_for_gqa_mha_no_bump(self):
        """MHA (ratio=1) should not bump."""
        cache = TurboQuantKVCache.for_gqa(
            head_dim=128, num_kv_heads=32, num_query_heads=32, bit_width=3
        )
        assert cache.key_quantizer.bit_width == 3
        assert cache.value_quantizer.bit_width == 3

    def test_for_gqa_caps_at_4_bits(self):
        """Key bits should not exceed 4 even with high base bit_width."""
        cache = TurboQuantKVCache.for_gqa(
            head_dim=64, num_kv_heads=1, num_query_heads=32, bit_width=4
        )
        assert cache.key_quantizer.bit_width == 4  # min(4+1, 4) = 4

    def test_separate_key_value_bits(self):
        """Can set different bits for keys and values directly."""
        cache = TurboQuantKVCache(head_dim=64, key_bit_width=4, value_bit_width=2)
        assert cache.key_quantizer.bit_width == 4
        assert cache.value_quantizer.bit_width == 2

    def test_default_bits_fallback(self):
        """When key/value overrides are None, both use bit_width."""
        cache = TurboQuantKVCache(head_dim=64, bit_width=3)
        assert cache.key_quantizer.bit_width == 3
        assert cache.value_quantizer.bit_width == 3

    def test_gqa_attention_shape(self):
        """GQA cache produces correct attention output shape."""
        cache = TurboQuantKVCache.for_gqa(
            head_dim=64, num_kv_heads=4, num_query_heads=16, bit_width=3
        )
        # KV shape uses num_kv_heads
        keys = torch.randn(1, 4, 32, 64)
        values = torch.randn(1, 4, 32, 64)
        query = torch.randn(1, 4, 1, 64)  # query per KV head group
        compressed = cache.compress(keys, values)
        out = cache.attention(query, compressed)
        assert out.shape == (1, 4, 1, 64)

    def test_gqa_with_residual(self):
        """GQA factory respects residual_length."""
        cache = TurboQuantKVCache.for_gqa(
            head_dim=64,
            num_kv_heads=8,
            num_query_heads=32,
            bit_width=3,
            residual_length=16,
        )
        assert cache.residual_length == 16
        keys = torch.randn(1, 8, 64, 64)
        values = torch.randn(1, 8, 64, 64)
        compressed = cache.compress(keys, values)
        assert compressed.split_point == 48  # 64 - 16
        assert compressed.residual_keys.shape[2] == 16
