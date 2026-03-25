"""Empirical validation of distortion bounds from TurboQuant paper Table 1.

The paper reports normalized MSE distortion D_mse for unit vectors:
  1-bit: ~0.36
  2-bit: ~0.117
  3-bit: ~0.03
  4-bit: ~0.009

We validate that empirical distortion is in the right ballpark (within 5x)
since our codebook values may not be exactly optimal.

Reference: Table 1 of TurboQuant (arXiv:2504.19874)
"""

import torch
import pytest
from turboquant.mse_quantizer import TurboQuantMSE


# Paper Table 1 values (MSE distortion for unit vectors)
PAPER_DISTORTION = {
    1: 0.36,
    2: 0.117,
    3: 0.03,
    4: 0.009,
}


class TestDistortionBounds:
    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_distortion_in_ballpark(self, bits):
        """Empirical distortion is within 5x of paper Table 1 values."""
        dim = 256
        n_vectors = 500
        torch.manual_seed(0)
        # Random unit vectors
        x = torch.randn(n_vectors, dim)
        x = x / torch.norm(x, dim=-1, keepdim=True)

        mse = TurboQuantMSE(dim=dim, bits=bits, seed=42)
        distortion = mse.distortion(x).mean().item()
        paper_val = PAPER_DISTORTION[bits]

        # Should be within 5x of paper value
        assert distortion < paper_val * 5, (
            f"{bits}-bit: empirical={distortion:.4f}, paper={paper_val}, "
            f"ratio={distortion/paper_val:.1f}x"
        )
        # Should not be zero
        assert distortion > paper_val * 0.01

    def test_monotonically_decreasing(self):
        """More bits -> lower distortion, consistently."""
        dim = 256
        torch.manual_seed(0)
        x = torch.randn(200, dim)
        x = x / torch.norm(x, dim=-1, keepdim=True)

        distortions = []
        for bits in [1, 2, 3, 4]:
            mse = TurboQuantMSE(dim=dim, bits=bits, seed=42)
            d = mse.distortion(x).mean().item()
            distortions.append(d)

        for i in range(len(distortions) - 1):
            assert distortions[i] > distortions[i + 1], (
                f"Distortion not monotonically decreasing: "
                f"{distortions[i]:.4f} <= {distortions[i+1]:.4f}"
            )
