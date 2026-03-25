"""Tests for MSE-optimal quantizer."""

import pytest
import torch

from turboquant.mse_quantizer import TurboQuantMSE


class TestTurboQuantMSE:
    def test_roundtrip_shape(self):
        """Quantize/dequantize preserves shape."""
        mse = TurboQuantMSE(dim=100, bits=2)
        x = torch.randn(10, 100)
        out = mse.quantize(x)
        x_hat = mse.dequantize(out)
        assert x_hat.shape == x.shape

    def test_norms_stored(self):
        """Norms are correctly stored and restored."""
        mse = TurboQuantMSE(dim=64, bits=3)
        x = torch.randn(5, 64)
        out = mse.quantize(x)
        # Reconstructed norms should be similar (not exact due to quantization)
        orig_norms = torch.norm(x, dim=-1)
        # The norms are stored exactly, but reconstruction changes direction
        assert out.norms.shape == orig_norms.shape
        torch.testing.assert_close(out.norms, orig_norms, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_distortion_reasonable(self, bits):
        """Distortion is finite and non-negative."""
        mse = TurboQuantMSE(dim=128, bits=bits)
        x = torch.randn(50, 128)
        d = mse.distortion(x)
        assert (d >= 0).all()
        assert torch.isfinite(d).all()

    def test_more_bits_less_distortion(self):
        """Higher bit width gives lower distortion."""
        x = torch.randn(100, 128)
        distortions = {}
        for bits in [1, 2, 3, 4]:
            mse = TurboQuantMSE(dim=128, bits=bits, seed=0)
            distortions[bits] = mse.distortion(x).mean().item()
        for b in [1, 2, 3]:
            assert distortions[b] > distortions[b + 1]

    def test_residual_shape(self):
        """Residual has correct shape."""
        mse = TurboQuantMSE(dim=64, bits=2)
        x = torch.randn(5, 64)
        r = mse.get_residual(x)
        assert r.shape == x.shape

    def test_zero_vector(self):
        """Zero vectors don't cause NaN."""
        mse = TurboQuantMSE(dim=64, bits=2)
        x = torch.zeros(3, 64)
        out = mse.quantize(x)
        x_hat = mse.dequantize(out)
        assert torch.isfinite(x_hat).all()
