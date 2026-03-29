"""Integration test for Pre-RoPE with other features."""

import torch
import torch.nn.functional as F

from turboquant.kv_cache import TurboQuantKVCache
from turboquant.rope import apply_rope, compute_rope_frequencies


class TestPreRoPE:
    def test_pre_rope_flag(self):
        """pre_rope parameter is stored."""
        cache = TurboQuantKVCache(head_dim=64, bit_width=3, pre_rope=True)
        assert cache.pre_rope is True
        cache2 = TurboQuantKVCache(head_dim=64, bit_width=3)
        assert cache2.pre_rope is False

    def test_pre_rope_compress_stores_positions(self):
        """Compressed output stores positions when pre_rope=True."""
        cache = TurboQuantKVCache(head_dim=64, bit_width=3, pre_rope=True, residual_length=0)
        keys = torch.randn(1, 2, 32, 64)
        values = torch.randn(1, 2, 32, 64)
        positions = torch.arange(32)
        freqs = compute_rope_frequencies(64)
        compressed = cache.compress(keys, values, positions=positions, rope_freqs=freqs)
        assert compressed.positions is not None
        assert torch.equal(compressed.positions, positions)

    def test_non_pre_rope_no_positions(self):
        """Compressed output has no positions when pre_rope=False."""
        cache = TurboQuantKVCache(head_dim=64, bit_width=3, pre_rope=False, residual_length=0)
        keys = torch.randn(1, 2, 32, 64)
        values = torch.randn(1, 2, 32, 64)
        compressed = cache.compress(keys, values)
        assert compressed.positions is None

    def test_pre_rope_attention_output_shape(self):
        """Attention with pre_rope produces correct shape."""
        head_dim = 64
        cache = TurboQuantKVCache(head_dim=head_dim, bit_width=3, pre_rope=True, residual_length=0)
        freqs = compute_rope_frequencies(head_dim)

        keys_raw = torch.randn(1, 2, 16, head_dim)
        values = torch.randn(1, 2, 16, head_dim)
        query_raw = torch.randn(1, 2, 1, head_dim)

        positions = torch.arange(16)
        query_pos = torch.tensor([16])

        compressed = cache.compress(keys_raw, values, positions=positions, rope_freqs=freqs)
        out = cache.attention(
            query_raw,
            compressed,
            query_positions=query_pos,
            rope_freqs=freqs,
        )
        assert out.shape == (1, 2, 1, head_dim)
        assert not torch.isnan(out).any()

    def test_pre_rope_produces_reasonable_output(self):
        """Pre-RoPE quantization should produce non-degenerate attention output."""
        torch.manual_seed(42)
        head_dim = 128
        seq_len = 256
        freqs = compute_rope_frequencies(head_dim, max_seq_len=seq_len + 1)
        positions = torch.arange(seq_len)
        query_pos = torch.tensor([seq_len])

        keys_raw = torch.randn(1, 4, seq_len, head_dim)
        values = torch.randn(1, 4, seq_len, head_dim)
        query_raw = torch.randn(1, 4, 1, head_dim)

        # Ground truth: apply RoPE then compute attention
        keys_roped = apply_rope(keys_raw, freqs, positions)
        query_roped = apply_rope(query_raw, freqs, query_pos)
        scale = head_dim**-0.5
        true_attn = F.softmax(query_roped @ keys_roped.transpose(-2, -1) * scale, dim=-1)
        true_out = true_attn @ values

        # Pre-RoPE quantization
        cache_pre = TurboQuantKVCache(
            head_dim=head_dim, bit_width=3, pre_rope=True, residual_length=0
        )
        comp_pre = cache_pre.compress(keys_raw, values, positions=positions, rope_freqs=freqs)
        out_pre = cache_pre.attention(
            query_raw,
            comp_pre,
            query_positions=query_pos,
            rope_freqs=freqs,
        )
        mse_pre = ((true_out - out_pre) ** 2).mean().item()

        # Post-RoPE quantization (baseline)
        cache_post = TurboQuantKVCache(
            head_dim=head_dim, bit_width=3, pre_rope=False, residual_length=0
        )
        comp_post = cache_post.compress(keys_roped, values)
        out_post = cache_post.attention(query_roped, comp_post)
        mse_post = ((true_out - out_post) ** 2).mean().item()

        # Pre-RoPE should be at least comparable
        assert mse_pre < mse_post * 2.0, f"Pre-RoPE much worse: {mse_pre} vs {mse_post}"


class TestPreRoPEWithSlidingWindow:
    def test_pre_rope_plus_sliding_window(self):
        """Pre-RoPE works together with sliding window."""
        head_dim = 64
        cache = TurboQuantKVCache(
            head_dim=head_dim,
            bit_width=3,
            pre_rope=True,
            residual_length=16,
        )
        freqs = compute_rope_frequencies(head_dim, max_seq_len=128)

        keys = torch.randn(1, 2, 64, head_dim)
        values = torch.randn(1, 2, 64, head_dim)
        query = torch.randn(1, 2, 1, head_dim)
        positions = torch.arange(64)
        query_pos = torch.tensor([64])

        compressed = cache.compress(keys, values, positions=positions, rope_freqs=freqs)
        assert compressed.split_point == 48  # 64 - 16
        assert compressed.positions is not None

        out = cache.attention(query, compressed, query_positions=query_pos, rope_freqs=freqs)
        assert out.shape == (1, 2, 1, head_dim)
        assert not torch.isnan(out).any()

    def test_pre_rope_all_residual(self):
        """Pre-RoPE with all tokens in residual buffer."""
        head_dim = 64
        cache = TurboQuantKVCache(
            head_dim=head_dim, bit_width=3, pre_rope=True, residual_length=128
        )
        freqs = compute_rope_frequencies(head_dim)

        keys = torch.randn(1, 2, 32, head_dim)
        values = torch.randn(1, 2, 32, head_dim)
        query = torch.randn(1, 2, 1, head_dim)
        positions = torch.arange(32)
        query_pos = torch.tensor([32])

        compressed = cache.compress(keys, values, positions=positions, rope_freqs=freqs)
        assert compressed.keys is None  # all in residual
        out = cache.attention(query, compressed, query_positions=query_pos, rope_freqs=freqs)
        assert out.shape == (1, 2, 1, head_dim)


class TestPreRoPEWithGQA:
    def test_pre_rope_with_gqa_and_sliding_window(self):
        """Pre-RoPE + GQA + sliding window all work together."""
        cache = TurboQuantKVCache.for_gqa(
            head_dim=128,
            num_kv_heads=8,
            num_query_heads=32,
            bit_width=3,
            residual_length=16,
        )
        cache.pre_rope = True

        freqs = compute_rope_frequencies(128)
        keys = torch.randn(1, 8, 64, 128)
        values = torch.randn(1, 8, 64, 128)
        query = torch.randn(1, 8, 1, 128)
        positions = torch.arange(64)
        query_pos = torch.tensor([64])

        compressed = cache.compress(keys, values, positions=positions, rope_freqs=freqs)
        out = cache.attention(query, compressed, query_positions=query_pos, rope_freqs=freqs)

        assert out.shape == (1, 8, 1, 128)
        assert not torch.isnan(out).any()
