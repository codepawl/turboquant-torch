"""Integration test for full KV cache compression flow."""

import torch
import torch.nn.functional as F

from turboquant import TurboQuantKVCache


class TestKVCompressionFlow:
    def test_compress_attention_decompress(self):
        """Full flow: compress KV → run attention → output is reasonable."""
        B, H, S, D = 1, 4, 128, 128
        cache = TurboQuantKVCache(head_dim=D, bit_width=3, residual_length=32)

        keys = torch.randn(B, H, S, D)
        values = torch.randn(B, H, S, D)
        query = torch.randn(B, H, 1, D)

        # Ground truth
        scale = D**-0.5
        true_out = F.softmax(query @ keys.transpose(-2, -1) * scale, dim=-1) @ values

        # Compressed
        compressed = cache.compress(keys, values)
        comp_out = cache.attention(query, compressed)

        # Should be close
        mse = ((true_out - comp_out) ** 2).mean().item()
        assert mse < 0.01, f"Attention output MSE too high: {mse}"

    def test_gqa_plus_sliding_window(self):
        """GQA factory + sliding window work together."""
        cache = TurboQuantKVCache.for_gqa(
            head_dim=128,
            num_kv_heads=8,
            num_query_heads=32,
            bit_width=3,
            residual_length=32,
        )

        keys = torch.randn(1, 8, 128, 128)
        values = torch.randn(1, 8, 128, 128)
        query = torch.randn(1, 8, 1, 128)

        compressed = cache.compress(keys, values)
        out = cache.attention(query, compressed)

        assert out.shape == (1, 8, 1, 128)
        assert not torch.isnan(out).any()

    def test_memory_actually_smaller(self):
        """Compressed memory is actually smaller than original."""
        cache = TurboQuantKVCache(head_dim=128, bit_width=3, residual_length=0)
        orig, comp, ratio = cache.memory_savings(1, 32, 2048)
        assert ratio > 5.0
        assert comp < orig
