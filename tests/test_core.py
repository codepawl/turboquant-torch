"""Tests for TurboQuant full pipeline."""

import pytest
import torch

from turboquant.core import TurboQuant


class TestTurboQuant:
    def test_roundtrip_shape(self):
        """Quantize/dequantize preserves shape."""
        tq = TurboQuant(dim=128, bit_width=3, unbiased=True)
        x = torch.randn(10, 128)
        out = tq.quantize(x)
        x_hat = tq.dequantize(out)
        assert x_hat.shape == x.shape

    def test_biased_mode(self):
        """Biased mode (no QJL) works and has no QJL codes."""
        tq = TurboQuant(dim=64, bit_width=3, unbiased=False)
        x = torch.randn(5, 64)
        out = tq.quantize(x)
        assert out.qjl_output is None
        x_hat = tq.dequantize(out)
        assert x_hat.shape == x.shape

    def test_unbiased_has_qjl(self):
        """Unbiased mode produces QJL codes."""
        tq = TurboQuant(dim=64, bit_width=3, unbiased=True)
        x = torch.randn(5, 64)
        out = tq.quantize(x)
        assert out.qjl_output is not None
        assert out.qjl_output.sign_bits.shape == (5, 64)

    def test_relative_mse_bounded(self):
        """Relative MSE is < 0.5 for 3-bit quantization."""
        tq = TurboQuant(dim=128, bit_width=3, unbiased=False)
        x = torch.randn(200, 128)
        out = tq.quantize(x)
        x_hat = tq.dequantize(out)
        mse = ((x - x_hat) ** 2).sum(dim=-1)
        energy = (x**2).sum(dim=-1)
        rel_mse = (mse / energy).mean().item()
        assert rel_mse < 0.5

    def test_compression_ratio(self):
        """3-bit gives > 8x compression ratio."""
        tq = TurboQuant(dim=128, bit_width=3)
        assert tq.compression_ratio() > 8

    def test_compute_inner_product_single_query(self):
        """Inner product computation with single query."""
        tq = TurboQuant(dim=64, bit_width=3, unbiased=True)
        db = torch.randn(20, 64)
        query = torch.randn(64)
        out = tq.quantize(db)
        ip = tq.compute_inner_product(query, out)
        assert ip.shape == (20,)

    def test_compute_inner_product_batch_query(self):
        """Inner product computation with batch queries."""
        tq = TurboQuant(dim=64, bit_width=3, unbiased=True)
        db = torch.randn(20, 64)
        queries = torch.randn(5, 64)
        out = tq.quantize(db)
        ip = tq.compute_inner_product(queries, out)
        assert ip.shape == (5, 20)

    def test_bit_width_2_minimum_unbiased(self):
        """Minimum unbiased bit width is 2."""
        tq = TurboQuant(dim=64, bit_width=2, unbiased=True)
        x = torch.randn(5, 64)
        out = tq.quantize(x)
        assert out.qjl_output is not None

    def test_bit_width_1_unbiased_raises(self):
        """bit_width=1 with unbiased=True should raise."""
        with pytest.raises(ValueError):
            TurboQuant(dim=64, bit_width=1, unbiased=True)

    def test_memory_bytes(self):
        """Memory estimation is positive and scales with n_vectors."""
        tq = TurboQuant(dim=128, bit_width=3)
        m1 = tq.memory_bytes(100)
        m2 = tq.memory_bytes(200)
        assert m1 > 0
        assert m2 > m1
