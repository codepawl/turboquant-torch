"""MSE-optimal quantizer: randomized Hadamard rotation + Lloyd-Max scalar quantization.

This is Stage 1 of TurboQuant. It normalizes input vectors to unit norm,
applies a randomized Hadamard transform to spread information uniformly
across coordinates, then applies an MSE-optimal scalar quantizer (Lloyd-Max)
independently to each coordinate.

MSE distortion bound for unit vectors: D_mse <= (sqrt(3) * pi / 2) * (1 / 4^b)

Reference: Section 3.1, Theorem 3.3 of TurboQuant (arXiv:2504.19874)
"""

from typing import NamedTuple

import torch

from .codebook import LloydMaxCodebook
from .hadamard import RandomizedHadamardTransform


class MSEQuantizedOutput(NamedTuple):
    """Output of MSE quantization."""

    codes: torch.Tensor  # Integer codes, shape (..., padded_dim)
    norms: torch.Tensor  # L2 norms, shape (...)
    bit_width: int  # Bits per coordinate


class TurboQuantMSE:
    """MSE-optimal vector quantizer using rotation + scalar quantization.

    Args:
        dim: Input vector dimension.
        bits: Bits per coordinate for the scalar quantizer (1-4).
        seed: Random seed for the Hadamard rotation.
    """

    def __init__(self, dim: int, bits: int = 2, seed: int = 0):
        self.dim = dim
        self.bits = bits
        self.rht = RandomizedHadamardTransform(dim, seed=seed)
        self.codebook = LloydMaxCodebook(bits, self.rht.padded_dim)

    def to(self, device: torch.device) -> "TurboQuantMSE":
        """Move to device."""
        self.rht = self.rht.to(device)
        self.codebook = self.codebook.to(device)
        return self

    def quantize(self, x: torch.Tensor) -> MSEQuantizedOutput:
        """Quantize input vectors.

        Args:
            x: Input tensor of shape (..., dim).

        Returns:
            MSEQuantizedOutput with codes, norms, and bit_width.
        """
        # Store L2 norm
        norms = torch.norm(x, dim=-1)

        # Normalize to unit norm (handle zero vectors)
        safe_norms = norms.clamp(min=1e-10)
        x_unit = x / safe_norms.unsqueeze(-1)

        # Rotate
        x_rot = self.rht.forward(x_unit)

        # Scalar quantize each coordinate
        codes = self.codebook.quantize(x_rot)

        return MSEQuantizedOutput(codes=codes, norms=norms, bit_width=self.bits)

    def dequantize(self, output: MSEQuantizedOutput) -> torch.Tensor:
        """Reconstruct vectors from quantized codes.

        Args:
            output: MSEQuantizedOutput from quantize().

        Returns:
            Reconstructed tensor of shape (..., dim).
        """
        # Lookup centroids
        x_rot = self.codebook.dequantize(output.codes)

        # Inverse rotation
        x_unit = self.rht.inverse(x_rot)

        # Rescale by stored norm
        return x_unit * output.norms.unsqueeze(-1)

    def get_residual(self, x: torch.Tensor) -> torch.Tensor:
        """Compute quantization residual: x - dequantize(quantize(x)).

        Args:
            x: Input tensor of shape (..., dim).

        Returns:
            Residual tensor of shape (..., dim).
        """
        output = self.quantize(x)
        x_hat = self.dequantize(output)
        return x - x_hat

    def distortion(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-vector MSE distortion ||x - x_hat||^2.

        Args:
            x: Input tensor of shape (..., dim).

        Returns:
            Distortion values of shape (...).
        """
        residual = self.get_residual(x)
        return (residual**2).sum(dim=-1)
