"""Property-based tests using Hypothesis.

These tests generate random inputs to find edge cases that
manual tests might miss. Each test verifies an invariant
that must hold for ALL valid inputs.
"""

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from turboquant import TurboQuant, TurboQuantKVCache
from turboquant.codebook import get_codebook
from turboquant.outlier import detect_outlier_channels, merge_outliers, split_outliers

dims = st.sampled_from([32, 64, 96, 128, 256])
bit_widths = st.sampled_from([2, 3, 4])
small_n = st.integers(min_value=1, max_value=50)
seq_lens = st.sampled_from([1, 4, 16, 64, 128])
head_counts = st.sampled_from([1, 2, 4, 8])
batch_sizes = st.sampled_from([1, 2, 4])
outlier_counts = st.sampled_from([0, 1, 4, 8, 16])


class TestTurboQuantProperties:
    @given(dim=dims, bit_width=bit_widths, n=small_n)
    @settings(max_examples=50, deadline=5000)
    def test_roundtrip_preserves_shape(self, dim, bit_width, n):
        """Quantize then dequantize always preserves shape."""
        tq = TurboQuant(dim=dim, bit_width=bit_width, unbiased=True)
        x = torch.randn(n, dim)
        out = tq.quantize(x)
        x_hat = tq.dequantize(out)
        assert x_hat.shape == x.shape

    @given(dim=dims, bit_width=bit_widths, n=small_n)
    @settings(max_examples=50, deadline=5000)
    def test_no_nan_in_output(self, dim, bit_width, n):
        """Output never contains NaN."""
        tq = TurboQuant(dim=dim, bit_width=bit_width, unbiased=True)
        x = torch.randn(n, dim)
        x_hat = tq.dequantize(tq.quantize(x))
        assert not torch.isnan(x_hat).any()
        assert not torch.isinf(x_hat).any()

    @given(dim=dims, n=small_n)
    @settings(max_examples=30, deadline=5000)
    def test_more_bits_less_distortion(self, dim, n):
        """Higher bit_width always gives lower or equal MSE distortion."""
        x = torch.randn(n, dim)
        distortions = []
        for bw in [2, 3, 4]:
            tq = TurboQuant(dim=dim, bit_width=bw, unbiased=True)
            x_hat = tq.dequantize(tq.quantize(x))
            d = ((x - x_hat) ** 2).sum(dim=-1).mean().item()
            distortions.append(d)
        for i in range(len(distortions) - 1):
            assert distortions[i] >= distortions[i + 1] * 0.9

    @given(dim=dims, bit_width=bit_widths)
    @settings(max_examples=30, deadline=5000)
    def test_compression_ratio_positive(self, dim, bit_width):
        """Compression ratio is always > 1."""
        tq = TurboQuant(dim=dim, bit_width=bit_width, unbiased=True)
        assert tq.compression_ratio() > 1.0

    @given(dim=dims, bit_width=bit_widths, n=small_n)
    @settings(max_examples=30, deadline=5000)
    def test_zero_input(self, dim, bit_width, n):
        """Zero vectors don't crash."""
        tq = TurboQuant(dim=dim, bit_width=bit_width, unbiased=True)
        x = torch.zeros(n, dim)
        x_hat = tq.dequantize(tq.quantize(x))
        assert x_hat.shape == x.shape
        assert not torch.isnan(x_hat).any()

    @given(dim=dims, bit_width=bit_widths, n=small_n)
    @settings(max_examples=30, deadline=5000)
    def test_large_values(self, dim, bit_width, n):
        """Very large values don't crash or produce NaN."""
        tq = TurboQuant(dim=dim, bit_width=bit_width, unbiased=True)
        x = torch.randn(n, dim) * 1000
        x_hat = tq.dequantize(tq.quantize(x))
        assert not torch.isnan(x_hat).any()
        assert not torch.isinf(x_hat).any()

    @given(dim=dims, bit_width=bit_widths, n=small_n)
    @settings(max_examples=30, deadline=5000)
    def test_single_vector(self, dim, bit_width, n):
        """Single vector works (n=1)."""
        tq = TurboQuant(dim=dim, bit_width=bit_width, unbiased=True)
        x = torch.randn(1, dim)
        x_hat = tq.dequantize(tq.quantize(x))
        assert x_hat.shape == (1, dim)


class TestKVCacheProperties:
    @given(
        head_dim=st.sampled_from([64, 128, 256]),
        bit_width=bit_widths,
        seq_len=seq_lens,
        heads=head_counts,
        batch=batch_sizes,
        outliers=st.sampled_from([0, 4, 8]),
        residual=st.sampled_from([0, 16, 32]),
    )
    @settings(max_examples=50, deadline=10000)
    def test_compress_decompress_shape(
        self, head_dim, bit_width, seq_len, heads, batch, outliers, residual
    ):
        """Compress then decompress always preserves original shape."""
        outliers = min(outliers, head_dim // 4)
        cache = TurboQuantKVCache(
            head_dim=head_dim,
            bit_width=bit_width,
            n_outlier_channels=outliers,
            residual_length=residual,
        )
        k = torch.randn(batch, heads, seq_len, head_dim)
        v = torch.randn(batch, heads, seq_len, head_dim)
        comp = cache.compress(k, v)
        k_hat = cache.decompress_keys(comp)
        v_hat = cache.decompress_values(comp)
        assert k_hat.shape == k.shape
        assert v_hat.shape == v.shape

    @given(
        head_dim=st.sampled_from([64, 128]),
        bit_width=bit_widths,
        seq_len=st.sampled_from([16, 64]),
        heads=st.sampled_from([2, 4]),
    )
    @settings(max_examples=30, deadline=10000)
    def test_attention_output_valid(self, head_dim, bit_width, seq_len, heads):
        """Attention output is always valid (no NaN, correct shape)."""
        cache = TurboQuantKVCache(
            head_dim=head_dim,
            bit_width=bit_width,
            residual_length=0,
        )
        k = torch.randn(1, heads, seq_len, head_dim)
        v = torch.randn(1, heads, seq_len, head_dim)
        q = torch.randn(1, heads, 1, head_dim)
        comp = cache.compress(k, v)
        out = cache.attention(q, comp)
        assert out.shape == (1, heads, 1, head_dim)
        assert not torch.isnan(out).any()

    @given(
        head_dim=st.sampled_from([64, 128]),
        seq_len=st.sampled_from([16, 64]),
    )
    @settings(max_examples=20, deadline=10000)
    def test_memory_savings_positive(self, head_dim, seq_len):
        """Compressed memory is always less than original."""
        cache = TurboQuantKVCache(
            head_dim=head_dim,
            bit_width=3,
            residual_length=0,
        )
        orig, comp, ratio = cache.memory_savings(1, 4, seq_len)
        assert ratio > 1.0
        assert comp < orig


class TestOutlierProperties:
    @given(
        dim=dims,
        n_outliers=st.integers(min_value=1, max_value=8),
        n=st.integers(min_value=5, max_value=50),
    )
    @settings(max_examples=40, deadline=5000)
    def test_split_merge_roundtrip(self, dim, n_outliers, n):
        """Split then merge recovers original tensor exactly."""
        n_outliers = min(n_outliers, dim // 4)
        if n_outliers == 0:
            return
        x = torch.randn(n, dim)
        indices = detect_outlier_channels(x, n_outliers=n_outliers)
        split = split_outliers(x, indices)
        merged = merge_outliers(split.bulk, split)
        assert torch.allclose(x, merged, atol=1e-5)

    @given(dim=dims, n_outliers=st.integers(min_value=1, max_value=8))
    @settings(max_examples=30, deadline=5000)
    def test_detect_returns_sorted_unique(self, dim, n_outliers):
        """Detected indices are sorted and unique."""
        n_outliers = min(n_outliers, dim // 4)
        if n_outliers == 0:
            return
        x = torch.randn(20, dim)
        indices = detect_outlier_channels(x, n_outliers=n_outliers)
        assert len(indices) == len(indices.unique())
        assert (indices[1:] >= indices[:-1]).all()


class TestCodebookProperties:
    @given(bit_width=bit_widths, dim=dims)
    @settings(max_examples=30, deadline=5000)
    def test_codebook_boundaries_sorted(self, bit_width, dim):
        """Codebook boundaries are always sorted."""
        cb = get_codebook(bit_width, dim)
        for i in range(len(cb.boundaries) - 1):
            assert cb.boundaries[i] <= cb.boundaries[i + 1]

    @given(bit_width=bit_widths, dim=dims)
    @settings(max_examples=30, deadline=5000)
    def test_codebook_centroids_sorted(self, bit_width, dim):
        """Codebook centroids are always sorted."""
        cb = get_codebook(bit_width, dim)
        for i in range(len(cb.centroids) - 1):
            assert cb.centroids[i] <= cb.centroids[i + 1]

    @given(bit_width=bit_widths, dim=dims)
    @settings(max_examples=30, deadline=5000)
    def test_codebook_level_count(self, bit_width, dim):
        """Codebook has exactly 2^bit_width levels."""
        cb = get_codebook(bit_width, dim)
        assert len(cb.centroids) == 2**bit_width
        assert len(cb.boundaries) == 2**bit_width - 1
