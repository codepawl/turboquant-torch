"""Quantized Johnson-Lindenstrauss (QJL) 1-bit quantizer.

QJL projects vectors using a random Rademacher matrix and stores only the
sign bits. Combined with an MSE quantizer on the original vector, QJL on the
residual produces an unbiased inner product estimator.

The asymmetric estimator uses full-precision query projections with 1-bit
key codes. For Rademacher projections, the unbiased scaling is:
  <x, y> ≈ ||x|| * sqrt(pi/2) * (1/m) * sum_i sign(r_i . x) * (r_i . y)

Reference: Section 3.2 of TurboQuant (arXiv:2504.19874)
See also: QJL paper (arXiv:2406.03482)
"""

import math
from typing import NamedTuple, Optional

import torch


def pack_bits(bits: torch.Tensor) -> torch.Tensor:
    """Pack a binary {0, 1} tensor into uint8 for 8x memory savings.

    Args:
        bits: Tensor of 0s and 1s with shape (..., d) where d is
            divisible by 8.

    Returns:
        Packed uint8 tensor of shape (..., d // 8).
    """
    *batch, d = bits.shape
    assert d % 8 == 0, f"Last dim must be divisible by 8, got {d}"
    bits = bits.view(*batch, d // 8, 8).to(torch.uint8)
    # Bit 0 is MSB
    powers = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1],
                          dtype=torch.uint8, device=bits.device)
    return (bits * powers).sum(dim=-1).to(torch.uint8)


def unpack_bits(packed: torch.Tensor, num_bits: int) -> torch.Tensor:
    """Unpack uint8 tensor back to binary {0, 1} tensor.

    Args:
        packed: uint8 tensor of shape (..., d // 8).
        num_bits: Total number of bits (original last dimension).

    Returns:
        Binary tensor of shape (..., num_bits).
    """
    *batch, n_bytes = packed.shape
    assert num_bits == n_bytes * 8, f"num_bits={num_bits} != n_bytes*8={n_bytes * 8}"
    packed = packed.to(torch.uint8)
    powers = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1],
                          dtype=torch.uint8, device=packed.device)
    # Expand each byte into 8 bits
    unpacked = (packed.unsqueeze(-1) // powers) % 2
    return unpacked.view(*batch, num_bits)


class QJLOutput(NamedTuple):
    """Output of QJL quantization."""
    sign_bits: torch.Tensor  # Binary {0,1} tensor, shape (..., proj_dim)
    norms: torch.Tensor      # L2 norms of input vectors, shape (...)


class QJL:
    """Quantized Johnson-Lindenstrauss transform for 1-bit quantization.

    Projects input vectors with a random Rademacher matrix and stores
    only the sign bits plus input norms. Supports asymmetric inner product
    estimation where the query is projected in full precision.

    Args:
        input_dim: Dimension of input vectors.
        proj_dim: Projection dimension (number of sign bits stored).
            Defaults to input_dim.
        seed: Random seed for reproducible projection matrix.
    """

    def __init__(self, input_dim: int, proj_dim: Optional[int] = None, seed: int = 42):
        self.input_dim = input_dim
        self.proj_dim = proj_dim or input_dim
        self.seed = seed
        self._proj_matrix: Optional[torch.Tensor] = None

    def _get_proj_matrix(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Lazily generate the random Rademacher projection matrix."""
        if self._proj_matrix is None or self._proj_matrix.device != device:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(self.seed)
            # Rademacher: uniform {-1, +1}
            self._proj_matrix = (
                torch.randint(0, 2, (self.proj_dim, self.input_dim), generator=gen) * 2 - 1
            ).to(dtype=dtype, device=device)
        return self._proj_matrix

    def quantize(self, x: torch.Tensor) -> QJLOutput:
        """Project and quantize to sign bits.

        Args:
            x: Input tensor of shape (..., input_dim).

        Returns:
            QJLOutput with sign_bits (shape (..., proj_dim), values {0, 1})
            and norms (shape (...)).
        """
        R = self._get_proj_matrix(x.device, x.dtype)
        norms = torch.norm(x, dim=-1)
        # z = x @ R^T, shape (..., proj_dim)
        z = x @ R.t()
        # Store sign as {0, 1}: 1 if z >= 0, 0 if z < 0
        sign_bits = (z >= 0).to(torch.uint8)
        return QJLOutput(sign_bits=sign_bits, norms=norms)

    def dequantize_for_dot(self, output: QJLOutput) -> torch.Tensor:
        """Reconstruct a vector suitable for dot-product estimation.

        The returned vector v satisfies: <v, y> ≈ <x, y> for the
        original x that was quantized.

        Args:
            output: QJLOutput from quantize().

        Returns:
            Reconstructed tensor of shape (..., input_dim).
        """
        R = self._get_proj_matrix(output.sign_bits.device, torch.float32)
        # Convert {0, 1} back to {-1, +1}
        signs = output.sign_bits.float() * 2 - 1  # (..., proj_dim)
        # Asymmetric scaling: ||x|| * sqrt(pi/2) / m * R^T @ signs
        scale = output.norms.unsqueeze(-1) * math.sqrt(math.pi / 2) / self.proj_dim
        v = scale * (signs @ R)
        return v

    def estimate_inner_product(
        self, query: torch.Tensor, output: QJLOutput
    ) -> torch.Tensor:
        """Estimate inner products between full-precision query and quantized vectors.

        Uses asymmetric estimation: query is projected in full precision,
        then dotted with sign bits. Scaling includes stored norms for
        unbiased estimation.

        Args:
            query: Query tensor of shape (..., input_dim) or (input_dim,).
            output: QJLOutput with sign_bits shape (n, proj_dim) and norms shape (n,).

        Returns:
            Estimated inner products of shape (..., n).
        """
        R = self._get_proj_matrix(query.device, query.dtype)
        # Project query: shape (..., proj_dim)
        query_proj = query @ R.t()
        # Convert sign bits to {-1, +1}
        signs = output.sign_bits.float() * 2 - 1  # (n, proj_dim)
        # Raw dot: (..., proj_dim) @ (proj_dim, n) -> (..., n)
        raw = query_proj @ signs.t() / self.proj_dim
        # Scale by ||x_i|| * sqrt(pi/2) for each database vector
        ip = raw * output.norms.unsqueeze(0) * math.sqrt(math.pi / 2)
        return ip
