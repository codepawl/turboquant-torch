"""End-to-end pipeline tests combining multiple modules."""

import torch

from turboquant import TurboQuant, TurboQuantMSE


class TestFullPipeline:
    def test_quantize_dequantize_inner_product(self):
        """Full pipeline: quantize → dequantize → inner product is close to original."""
        torch.manual_seed(42)
        dim = 128
        tq = TurboQuant(dim=dim, bit_width=3, unbiased=True)

        x = torch.randn(50, dim)
        y = torch.randn(10, dim)

        true_ip = y @ x.t()

        output = tq.quantize(x)
        estimated_ip = tq.compute_inner_product(y, output)

        rel_error = ((true_ip - estimated_ip) ** 2).mean() / (true_ip**2).mean()
        assert rel_error < 0.2, f"Relative IP error too high: {rel_error:.4f}"

    def test_mse_then_full_pipeline_consistency(self):
        """MSE-only and full pipeline give consistent MSE component."""
        torch.manual_seed(42)
        dim = 128
        x = torch.randn(20, dim)

        mse_only = TurboQuantMSE(dim=dim, bit_width=2)
        full = TurboQuant(dim=dim, bit_width=3, unbiased=True)  # 2-bit MSE + 1-bit QJL

        mse_distortion = mse_only.distortion(x).mean()
        full_out = full.quantize(x)
        full_hat = full.dequantize(full_out)
        full_distortion = ((x - full_hat) ** 2).sum(dim=-1).mean()

        # Full pipeline (with QJL correction) should have similar or lower distortion
        assert full_distortion < mse_distortion * 2.0

    def test_different_bit_widths_ordering(self):
        """Higher bit width → lower distortion, consistently across pipeline."""
        torch.manual_seed(42)
        dim = 128
        x = torch.randn(100, dim)

        distortions = []
        for bw in [2, 3, 4]:
            tq = TurboQuant(dim=dim, bit_width=bw, unbiased=True)
            out = tq.quantize(x)
            x_hat = tq.dequantize(out)
            d = ((x - x_hat) ** 2).sum(dim=-1).mean().item()
            distortions.append(d)

        for i in range(len(distortions) - 1):
            assert distortions[i] > distortions[i + 1]
