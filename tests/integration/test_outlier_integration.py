"""Integration test: outlier routing with KV cache compression."""

import torch

from turboquant import TurboQuantKVCache


class TestOutlierKVCache:
    def test_outlier_compress_decompress(self):
        """KV cache with outlier routing preserves shapes."""
        cache = TurboQuantKVCache(
            head_dim=128,
            bit_width=3,
            n_outlier_channels=8,
            residual_length=0,
        )
        keys = torch.randn(1, 4, 64, 128)
        values = torch.randn(1, 4, 64, 128)

        compressed = cache.compress(keys, values)
        k_hat = cache.decompress_keys(compressed)
        v_hat = cache.decompress_values(compressed)

        assert k_hat.shape == keys.shape
        assert v_hat.shape == values.shape

    def test_outlier_improves_mse(self):
        """Outlier routing should reduce MSE on data with outlier channels."""
        head_dim = 128
        torch.manual_seed(42)
        keys = torch.randn(1, 4, 64, head_dim)
        values = torch.randn(1, 4, 64, head_dim)
        # Create artificial outliers
        keys[:, :, :, 0] *= 50
        keys[:, :, :, 1] *= 50

        # Without outlier routing
        cache_no = TurboQuantKVCache(
            head_dim=head_dim, bit_width=3, n_outlier_channels=0, residual_length=0
        )
        comp_no = cache_no.compress(keys, values)
        k_no = cache_no.decompress_keys(comp_no)
        mse_no = ((keys - k_no) ** 2).mean().item()

        # With outlier routing
        cache_yes = TurboQuantKVCache(
            head_dim=head_dim, bit_width=3, n_outlier_channels=8, residual_length=0
        )
        comp_yes = cache_yes.compress(keys, values)
        k_yes = cache_yes.decompress_keys(comp_yes)
        mse_yes = ((keys - k_yes) ** 2).mean().item()

        assert mse_yes < mse_no, f"Outlier routing should help: {mse_yes} >= {mse_no}"

    def test_outlier_plus_sliding_window(self):
        """Outlier routing + sliding window work together."""
        cache = TurboQuantKVCache(
            head_dim=128,
            bit_width=3,
            n_outlier_channels=8,
            residual_length=16,
        )
        keys = torch.randn(1, 4, 64, 128)
        values = torch.randn(1, 4, 64, 128)
        query = torch.randn(1, 4, 1, 128)

        compressed = cache.compress(keys, values)
        out = cache.attention(query, compressed)

        assert out.shape == (1, 4, 1, 128)
        assert not torch.isnan(out).any()

    def test_outlier_all_residual(self):
        """Outlier routing with seq_len <= residual_length skips quantization."""
        cache = TurboQuantKVCache(
            head_dim=64,
            bit_width=3,
            n_outlier_channels=4,
            residual_length=128,
        )
        keys = torch.randn(1, 2, 16, 64)
        values = torch.randn(1, 2, 16, 64)

        compressed = cache.compress(keys, values)
        k_hat = cache.decompress_keys(compressed)

        # All tokens in residual, so should be exact
        assert torch.allclose(keys, k_hat)
