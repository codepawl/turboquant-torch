"""Tests for outlier channel detection and routing."""

import pytest
import torch

from turboquant.outlier import detect_outlier_channels, merge_outliers, split_outliers


class TestDetectOutliers:
    def test_finds_large_channels(self):
        """Should identify channels with largest magnitudes."""
        x = torch.randn(10, 64)
        x[:, 5] *= 100
        x[:, 10] *= 100

        indices = detect_outlier_channels(x, n_outliers=2, method="magnitude")
        assert 5 in indices
        assert 10 in indices

    def test_4d_input(self):
        """Works with (batch, heads, seq, dim) tensors."""
        x = torch.randn(1, 4, 32, 128)
        x[:, :, :, 0] *= 100

        indices = detect_outlier_channels(x, n_outliers=4)
        assert 0 in indices
        assert len(indices) == 4

    def test_cap_at_quarter(self):
        """n_outliers capped at dim // 4."""
        x = torch.randn(10, 16)
        indices = detect_outlier_channels(x, n_outliers=100)
        assert len(indices) <= 4

    @pytest.mark.parametrize("method", ["magnitude", "variance", "range"])
    def test_all_methods(self, method):
        """All detection methods produce valid output."""
        x = torch.randn(10, 64)
        indices = detect_outlier_channels(x, n_outliers=4, method=method)
        assert len(indices) == 4
        assert indices.max() < 64

    def test_invalid_method_raises(self):
        """Unknown method raises ValueError."""
        x = torch.randn(10, 64)
        with pytest.raises(ValueError, match="Unknown"):
            detect_outlier_channels(x, n_outliers=4, method="bogus")


class TestSplitMerge:
    def test_split_dimensions(self):
        """Split produces correct dimensions."""
        x = torch.randn(1, 4, 32, 128)
        indices = torch.tensor([0, 5, 10, 15])

        split = split_outliers(x, indices)
        assert split.bulk.shape[-1] == 124  # 128 - 4
        assert split.outlier_values.shape[-1] == 4

    def test_roundtrip(self):
        """Split then merge recovers original tensor."""
        x = torch.randn(1, 4, 32, 64)
        indices = torch.tensor([3, 7, 15, 31])

        split = split_outliers(x, indices)
        merged = merge_outliers(split.bulk, split)

        assert torch.allclose(x, merged, atol=1e-6)

    def test_outlier_values_preserved(self):
        """Outlier channel values are exact (no quantization)."""
        x = torch.randn(10, 64)
        x[:, 5] = 999.0

        indices = torch.tensor([5])
        split = split_outliers(x, indices)

        assert torch.all(split.outlier_values[:, 0] == 999.0)

    def test_original_dim_stored(self):
        """OutlierSplit records original dimension."""
        x = torch.randn(10, 64)
        indices = torch.tensor([0, 1])
        split = split_outliers(x, indices)
        assert split.original_dim == 64
