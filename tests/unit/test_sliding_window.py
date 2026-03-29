"""Tests for sliding window (residual buffer) in TurboQuantKVCache."""

import torch
import torch.nn.functional as F

from turboquant.kv_cache import TurboQuantKVCache


class TestSlidingWindow:
    def test_all_residual_short_seq(self):
        """Short sequences stay entirely in fp16 residual."""
        cache = TurboQuantKVCache(head_dim=64, bit_width=3, residual_length=128)
        keys = torch.randn(1, 2, 50, 64)  # seq_len=50 < residual=128
        values = torch.randn(1, 2, 50, 64)
        compressed = cache.compress(keys, values)
        assert compressed.keys is None  # nothing quantized
        assert compressed.values is None
        assert compressed.residual_keys.shape == keys.shape
        assert compressed.residual_values.shape == values.shape
        assert compressed.split_point == 0

    def test_split_long_seq(self):
        """Long sequences split into quantized + residual."""
        cache = TurboQuantKVCache(head_dim=64, bit_width=3, residual_length=32)
        keys = torch.randn(1, 2, 100, 64)  # seq_len=100 > residual=32
        values = torch.randn(1, 2, 100, 64)
        compressed = cache.compress(keys, values)
        assert compressed.keys is not None  # old tokens quantized
        assert compressed.values is not None
        assert compressed.split_point == 68  # 100 - 32
        assert compressed.residual_keys.shape[2] == 32
        assert compressed.residual_values.shape[2] == 32

    def test_exact_residual_boundary(self):
        """Sequence length == residual_length → all residual."""
        cache = TurboQuantKVCache(head_dim=64, bit_width=3, residual_length=64)
        keys = torch.randn(1, 2, 64, 64)
        values = torch.randn(1, 2, 64, 64)
        compressed = cache.compress(keys, values)
        assert compressed.keys is None
        assert compressed.split_point == 0

    def test_zero_residual(self):
        """residual_length=0 quantizes everything (backwards compat)."""
        cache = TurboQuantKVCache(head_dim=64, bit_width=3, residual_length=0)
        keys = torch.randn(1, 2, 32, 64)
        values = torch.randn(1, 2, 32, 64)
        compressed = cache.compress(keys, values)
        assert compressed.keys is not None
        assert compressed.split_point == 32
        assert compressed.residual_keys.shape[2] == 0

    def test_attention_with_sliding_window(self):
        """Attention works with mixed quantized + residual KV."""
        cache = TurboQuantKVCache(head_dim=64, bit_width=3, residual_length=16)
        keys = torch.randn(1, 2, 64, 64)
        values = torch.randn(1, 2, 64, 64)
        query = torch.randn(1, 2, 1, 64)
        compressed = cache.compress(keys, values)
        out = cache.attention(query, compressed)
        assert out.shape == (1, 2, 1, 64)
        assert not torch.isnan(out).any()

    def test_attention_all_residual(self):
        """Attention works when everything is in the residual buffer."""
        cache = TurboQuantKVCache(head_dim=64, bit_width=3, residual_length=128)
        keys = torch.randn(1, 2, 32, 64)
        values = torch.randn(1, 2, 32, 64)
        query = torch.randn(1, 2, 1, 64)
        compressed = cache.compress(keys, values)
        out = cache.attention(query, compressed)
        assert out.shape == (1, 2, 1, 64)
        assert not torch.isnan(out).any()

    def test_decompress_roundtrip_shape(self):
        """Decompress returns correct shape with sliding window."""
        cache = TurboQuantKVCache(head_dim=64, bit_width=3, residual_length=16)
        keys = torch.randn(2, 4, 48, 64)
        values = torch.randn(2, 4, 48, 64)
        compressed = cache.compress(keys, values)
        k_hat = cache.decompress_keys(compressed)
        v_hat = cache.decompress_values(compressed)
        assert k_hat.shape == keys.shape
        assert v_hat.shape == values.shape

    def test_residual_improves_accuracy(self):
        """Sliding window should give lower attention MSE than full quantization."""
        torch.manual_seed(42)
        head_dim = 64
        keys = torch.randn(1, 2, 128, head_dim)
        values = torch.randn(1, 2, 128, head_dim)
        query = torch.randn(1, 2, 1, head_dim)

        # Ground truth attention
        scale = head_dim**-0.5
        true_attn = F.softmax(query @ keys.transpose(-2, -1) * scale, dim=-1)
        true_out = true_attn @ values

        # Without sliding window
        cache_no_sw = TurboQuantKVCache(head_dim=head_dim, bit_width=3, residual_length=0)
        comp_no_sw = cache_no_sw.compress(keys, values)
        out_no_sw = cache_no_sw.attention(query, comp_no_sw)

        # With sliding window (last 32 tokens in fp16)
        cache_sw = TurboQuantKVCache(head_dim=head_dim, bit_width=3, residual_length=32)
        comp_sw = cache_sw.compress(keys, values)
        out_sw = cache_sw.attention(query, comp_sw)

        mse_no_sw = ((true_out - out_no_sw) ** 2).mean().item()
        mse_sw = ((true_out - out_sw) ** 2).mean().item()

        assert mse_sw <= mse_no_sw, f"Sliding window should help: {mse_sw} > {mse_no_sw}"

    def test_memory_savings_with_residual(self):
        """Memory savings accounts for residual buffer."""
        cache_no_sw = TurboQuantKVCache(head_dim=128, bit_width=3, residual_length=0)
        cache_sw = TurboQuantKVCache(head_dim=128, bit_width=3, residual_length=128)

        _, comp_no_sw, ratio_no_sw = cache_no_sw.memory_savings(2, 32, 2048)
        _, comp_sw, ratio_sw = cache_sw.memory_savings(2, 32, 2048)

        # With residual, compressed size is larger → ratio is smaller
        assert comp_sw > comp_no_sw
        assert ratio_sw < ratio_no_sw
