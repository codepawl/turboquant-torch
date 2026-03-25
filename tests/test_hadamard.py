"""Tests for Fast Walsh-Hadamard Transform and randomized rotation."""

import pytest
import torch

from turboquant.hadamard import RandomizedHadamardTransform, fwht


class TestFWHT:
    def test_inverse_property(self):
        """FWHT applied twice (with normalization) is identity."""
        x = torch.randn(64)
        y = fwht(fwht(x))
        torch.testing.assert_close(y, x, atol=1e-5, rtol=1e-5)

    def test_batch_inverse(self):
        """FWHT inverse works with batched inputs."""
        x = torch.randn(10, 128)
        y = fwht(fwht(x))
        torch.testing.assert_close(y, x, atol=1e-5, rtol=1e-5)

    def test_norm_preservation(self):
        """FWHT (with normalization) preserves L2 norm."""
        x = torch.randn(256)
        y = fwht(x)
        torch.testing.assert_close(torch.norm(y), torch.norm(x), atol=1e-4, rtol=1e-4)

    def test_output_shape(self):
        """Output shape matches input shape."""
        x = torch.randn(5, 3, 32)
        y = fwht(x)
        assert y.shape == x.shape

    def test_non_power_of_2_raises(self):
        """Non-power-of-2 dimension raises assertion error."""
        x = torch.randn(10)
        with pytest.raises(AssertionError):
            fwht(x)


class TestRandomizedHadamardTransform:
    def test_roundtrip(self):
        """inverse(forward(x)) == x."""
        rht = RandomizedHadamardTransform(100, seed=42)
        x = torch.randn(5, 100)
        y = rht.forward(x)
        x_rec = rht.inverse(y)
        torch.testing.assert_close(x_rec, x, atol=1e-5, rtol=1e-5)

    def test_norm_preservation(self):
        """RHT preserves L2 norms of input vectors."""
        rht = RandomizedHadamardTransform(64, seed=0)
        x = torch.randn(20, 64)
        y = rht.forward(x)
        x_norms = torch.norm(x, dim=-1)
        y_norms = torch.norm(y, dim=-1)
        torch.testing.assert_close(x_norms, y_norms, atol=1e-4, rtol=1e-4)

    def test_padding(self):
        """Non-power-of-2 dims are padded correctly."""
        rht = RandomizedHadamardTransform(50, seed=0)
        assert rht.padded_dim == 64
        x = torch.randn(3, 50)
        y = rht.forward(x)
        assert y.shape[-1] == 64
        x_rec = rht.inverse(y)
        assert x_rec.shape[-1] == 50

    def test_deterministic_with_seed(self):
        """Same seed produces same rotation."""
        rht1 = RandomizedHadamardTransform(64, seed=123)
        rht2 = RandomizedHadamardTransform(64, seed=123)
        x = torch.randn(4, 64)
        torch.testing.assert_close(rht1.forward(x), rht2.forward(x))

    def test_different_seeds_differ(self):
        """Different seeds produce different rotations."""
        rht1 = RandomizedHadamardTransform(64, seed=0)
        rht2 = RandomizedHadamardTransform(64, seed=1)
        x = torch.randn(4, 64)
        assert not torch.allclose(rht1.forward(x), rht2.forward(x))
