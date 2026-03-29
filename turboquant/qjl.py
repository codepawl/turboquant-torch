"""Quantized Johnson-Lindenstrauss (QJL) 1-bit quantizer.

QJL projects vectors using a random Gaussian matrix and stores only the
sign bits. Combined with an MSE quantizer on the original vector, QJL on the
residual produces an unbiased inner product estimator.

The asymmetric estimator uses full-precision query projections with 1-bit
key codes. The unbiased scaling is:
  <x, y> ≈ ||x|| * sqrt(pi/2) * (1/m) * sum_i sign(r_i . x) * (r_i . y)

Reference: Section 3.2 of TurboQuant (arXiv:2504.19874)
See also: QJL paper (arXiv:2406.03482), official impl: https://github.com/amirzandieh/QJL
"""

import math
from typing import Literal, NamedTuple

import torch
import torch.nn.functional as F


def pack_bits(bits: torch.Tensor) -> torch.Tensor:
    """Pack a binary {0, 1} tensor into uint8 for 8x memory savings.

    Pads the last dimension to the next multiple of 8 if needed.

    Args:
        bits: Tensor of 0s and 1s with shape (..., d).

    Returns:
        Packed uint8 tensor of shape (..., ceil(d / 8)).
    """
    *batch, d = bits.shape
    pad = (8 - d % 8) % 8
    if pad > 0:
        bits = F.pad(bits, (0, pad))
    d_padded = bits.shape[-1]
    bits = bits.view(*batch, d_padded // 8, 8).to(torch.uint8)
    # Bit 0 is MSB
    powers = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8, device=bits.device)
    return (bits * powers).sum(dim=-1).to(torch.uint8)


def unpack_bits(packed: torch.Tensor, num_bits: int) -> torch.Tensor:
    """Unpack uint8 tensor back to binary {0, 1} tensor.

    Args:
        packed: uint8 tensor of shape (..., ceil(num_bits / 8)).
        num_bits: Total number of bits (original last dimension).

    Returns:
        Binary tensor of shape (..., num_bits).
    """
    *batch, n_bytes = packed.shape
    packed = packed.to(torch.uint8)
    powers = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8, device=packed.device)
    # Expand each byte into 8 bits
    unpacked = (packed.unsqueeze(-1) // powers) % 2
    return unpacked.view(*batch, n_bytes * 8)[..., :num_bits]


class QJLOutput(NamedTuple):
    """Output of QJL quantization."""

    sign_bits: torch.Tensor  # Binary {0,1} tensor, shape (..., proj_dim)
    norms: torch.Tensor  # L2 norms of input vectors, shape (...)


class QJL:
    """Quantized Johnson-Lindenstrauss transform for 1-bit quantization.

    Projects input vectors with a random matrix and stores only the sign
    bits plus input norms. Supports asymmetric inner product estimation
    where the query is projected in full precision.

    The paper (Definition 1) and the official QJL implementation use
    Gaussian i.i.d. N(0,1) entries by default.

    Args:
        input_dim: Dimension of input vectors.
        proj_dim: Projection dimension (number of sign bits stored).
            Defaults to input_dim.
        seed: Random seed for reproducible projection matrix.
        proj_type: Type of random projection matrix. "gaussian" (default,
            i.i.d. N(0,1)) or "rademacher" (uniform {-1, +1}).
    """

    def __init__(
        self,
        input_dim: int,
        proj_dim: int | None = None,
        seed: int = 0,
        proj_type: Literal["gaussian", "rademacher"] = "gaussian",
    ):
        self.input_dim = input_dim
        self.proj_dim = proj_dim or input_dim
        self.seed = seed
        self.proj_type = proj_type
        self._proj_matrix: torch.Tensor | None = None

    def to(self, device: torch.device) -> "QJL":
        """Move to device."""
        if self._proj_matrix is not None:
            self._proj_matrix = self._proj_matrix.to(device)
        return self

    def _get_proj_matrix(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Lazily generate the random projection matrix."""
        if self._proj_matrix is None or self._proj_matrix.device != device:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(self.seed)
            if self.proj_type == "rademacher":
                self._proj_matrix = (
                    torch.randint(0, 2, (self.proj_dim, self.input_dim), generator=gen) * 2 - 1
                ).to(dtype=dtype, device=device)
            else:
                self._proj_matrix = torch.randn(self.proj_dim, self.input_dim, generator=gen).to(
                    dtype=dtype, device=device
                )
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
        z = x @ R.t()
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
        signs = output.sign_bits.float() * 2 - 1  # {0,1} -> {-1,+1}
        # Asymmetric scaling: ||x|| * sqrt(pi/2) / m * R^T @ signs
        scale = output.norms.unsqueeze(-1) * math.sqrt(math.pi / 2) / self.proj_dim
        return scale * (signs @ R)

    def estimate_inner_product(self, query: torch.Tensor, output: QJLOutput) -> torch.Tensor:
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
        query_proj = query @ R.t()
        signs = output.sign_bits.float() * 2 - 1  # {0,1} -> {-1,+1}
        raw = query_proj @ signs.t() / self.proj_dim
        # Scale by ||x_i|| * sqrt(pi/2) for each database vector
        return raw * output.norms.unsqueeze(0) * math.sqrt(math.pi / 2)
