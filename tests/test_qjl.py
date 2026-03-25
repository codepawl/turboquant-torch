"""Tests for Quantized Johnson-Lindenstrauss transform."""

import torch
import pytest
from turboquant.qjl import QJL, QJLOutput, pack_bits, unpack_bits


class TestPackBits:
    def test_roundtrip(self):
        """pack then unpack is identity."""
        bits = torch.randint(0, 2, (10, 64), dtype=torch.uint8)
        packed = pack_bits(bits)
        unpacked = unpack_bits(packed, 64)
        assert torch.equal(bits, unpacked)

    def test_compression_ratio(self):
        """Packed tensor is 8x smaller."""
        bits = torch.randint(0, 2, (100, 128), dtype=torch.uint8)
        packed = pack_bits(bits)
        assert packed.shape == (100, 16)

    def test_known_values(self):
        """Test specific bit patterns."""
        bits = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.uint8)
        packed = pack_bits(bits)
        assert packed.item() == 255

        bits = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.uint8)
        packed = pack_bits(bits)
        assert packed.item() == 0

        bits = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.uint8)
        packed = pack_bits(bits)
        assert packed.item() == 128


class TestQJL:
    def test_output_shape(self):
        """Sign bits have correct shape."""
        qjl = QJL(64, proj_dim=64, seed=0)
        x = torch.randn(10, 64)
        out = qjl.quantize(x)
        assert out.sign_bits.shape == (10, 64)
        assert out.sign_bits.dtype == torch.uint8
        assert out.norms.shape == (10,)

    def test_binary_values(self):
        """Output contains only 0s and 1s."""
        qjl = QJL(128, seed=0)
        x = torch.randn(5, 128)
        out = qjl.quantize(x)
        assert ((out.sign_bits == 0) | (out.sign_bits == 1)).all()

    def test_dequantize_for_dot_shape(self):
        """Dequantized vector has input dimension."""
        qjl = QJL(64, proj_dim=64, seed=0)
        x = torch.randn(10, 64)
        out = qjl.quantize(x)
        v = qjl.dequantize_for_dot(out)
        assert v.shape == (10, 64)

    def test_approximate_unbiasedness(self):
        """Inner product estimation is approximately unbiased over many trials.

        Average over many random seeds/vectors to check that the
        estimator is centered around the true inner product.
        """
        dim = 64
        n_trials = 200
        estimates = []
        x = torch.randn(dim)
        y = torch.randn(dim)
        true_ip = (x * y).sum().item()

        for seed in range(n_trials):
            qjl = QJL(dim, proj_dim=dim, seed=seed)
            out = qjl.quantize(x.unsqueeze(0))
            est = qjl.estimate_inner_product(y, out)
            estimates.append(est.item())

        mean_est = sum(estimates) / len(estimates)
        # Should be within reasonable range of true value
        assert abs(mean_est - true_ip) < abs(true_ip) * 0.5 + 1.0, (
            f"mean_est={mean_est:.3f}, true_ip={true_ip:.3f}"
        )

    def test_deterministic_with_seed(self):
        """Same seed gives same quantization."""
        x = torch.randn(5, 64)
        qjl1 = QJL(64, seed=42)
        qjl2 = QJL(64, seed=42)
        out1 = qjl1.quantize(x)
        out2 = qjl2.quantize(x)
        assert torch.equal(out1.sign_bits, out2.sign_bits)

    def test_estimate_inner_product_shape(self):
        """Inner product estimation has correct output shape."""
        qjl = QJL(64, proj_dim=64, seed=0)
        db = torch.randn(20, 64)
        query = torch.randn(5, 64)
        out = qjl.quantize(db)
        ip = qjl.estimate_inner_product(query, out)
        assert ip.shape == (5, 20)
