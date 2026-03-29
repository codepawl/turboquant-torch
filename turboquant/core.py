"""TurboQuant: two-stage online vector quantizer.

Combines an MSE-optimal quantizer (Stage 1) with a QJL 1-bit quantizer
on the residual (Stage 2) to produce an unbiased inner product estimator.

Total bit budget: b bits per coordinate = (b-1) MSE bits + 1 QJL bit.

Reference: TurboQuant (arXiv:2504.19874)
"""

from typing import NamedTuple

import torch

from .mse_quantizer import MSEQuantizedOutput, TurboQuantMSE
from .qjl import QJL, QJLOutput


class TurboQuantOutput(NamedTuple):
    """Output of TurboQuant quantization."""

    mse_codes: torch.Tensor  # MSE quantizer codes, shape (..., padded_dim)
    mse_norms: torch.Tensor  # L2 norms, shape (...)
    qjl_output: QJLOutput | None  # QJL sign bits + residual norms, or None
    bit_width: int  # Total bits per coordinate
    dim: int  # Original input dimension


class TurboQuant:
    """TurboQuant two-stage vector quantizer.

    When unbiased=True, allocates (b-1) bits to the MSE quantizer and
    1 bit to QJL on the residual, producing an unbiased inner product
    estimator. When unbiased=False, all b bits go to MSE (biased but
    lower MSE distortion).

    Args:
        dim: Input vector dimension.
        bit_width: Total bits per coordinate (>= 2 for unbiased mode).
        unbiased: If True, use QJL correction for unbiased inner products.
        seed: Random seed for rotation and projection matrices.
    """

    def __init__(
        self,
        dim: int,
        bit_width: int = 3,
        unbiased: bool = True,
        seed: int = 0,
    ):
        self.dim = dim
        self.bit_width = bit_width
        self.unbiased = unbiased

        if unbiased:
            if bit_width < 2:
                raise ValueError("Unbiased mode requires bit_width >= 2")
            mse_bits = bit_width - 1
        else:
            mse_bits = bit_width

        self.mse = TurboQuantMSE(dim, bit_width=mse_bits, seed=seed)
        self.qjl: QJL | None = None
        if unbiased:
            self.qjl = QJL(dim, proj_dim=dim, seed=seed + 1)

    def to(self, device: torch.device) -> "TurboQuant":
        """Move to device."""
        self.mse = self.mse.to(device)
        if self.qjl is not None:
            self.qjl = self.qjl.to(device)
        return self

    def quantize(self, x: torch.Tensor) -> TurboQuantOutput:
        """Quantize input vectors.

        Args:
            x: Input tensor of shape (..., dim).

        Returns:
            TurboQuantOutput with MSE codes, norms, and optional QJL output.
        """
        mse_out = self.mse.quantize(x)

        qjl_output = None
        if self.qjl is not None:
            residual = x - self.mse.dequantize(mse_out)
            qjl_output = self.qjl.quantize(residual)

        return TurboQuantOutput(
            mse_codes=mse_out.codes,
            mse_norms=mse_out.norms,
            qjl_output=qjl_output,
            bit_width=self.bit_width,
            dim=self.dim,
        )

    def dequantize(self, output: TurboQuantOutput) -> torch.Tensor:
        """Reconstruct vectors from quantized output.

        If QJL output is present, adds the QJL correction to the
        MSE reconstruction.

        Args:
            output: TurboQuantOutput from quantize().

        Returns:
            Reconstructed tensor of shape (..., dim).
        """
        mse_out = MSEQuantizedOutput(
            codes=output.mse_codes,
            norms=output.mse_norms,
            bit_width=self.bit_width - (1 if self.unbiased else 0),
        )
        x_hat = self.mse.dequantize(mse_out)

        if self.qjl is not None and output.qjl_output is not None:
            qjl_correction = self.qjl.dequantize_for_dot(output.qjl_output)
            x_hat = x_hat + qjl_correction

        return x_hat

    def compute_inner_product(self, query: torch.Tensor, output: TurboQuantOutput) -> torch.Tensor:
        """Compute inner products between query and quantized vectors.

        Uses asymmetric estimation: full-precision query with quantized keys.
        When unbiased=True, combines MSE reconstruction with QJL correction
        for an unbiased estimator.

        Args:
            query: Query tensor of shape (dim,) or (n_queries, dim).
            output: TurboQuantOutput for database vectors.

        Returns:
            Inner product estimates of shape (n_queries, n_vectors) or (n_vectors,).
        """
        mse_out = MSEQuantizedOutput(
            codes=output.mse_codes,
            norms=output.mse_norms,
            bit_width=self.bit_width - (1 if self.unbiased else 0),
        )
        x_hat = self.mse.dequantize(mse_out)

        # MSE component: query @ x_hat^T
        ip = x_hat @ query if query.dim() == 1 else query @ x_hat.t()

        # QJL correction
        if self.qjl is not None and output.qjl_output is not None:
            qjl_ip = self.qjl.estimate_inner_product(query, output.qjl_output)
            ip = ip + qjl_ip.squeeze(0) if query.dim() == 1 else ip + qjl_ip

        return ip

    def compression_ratio(self) -> float:
        """Compression ratio vs 32-bit float storage.

        Returns:
            Ratio of original to compressed size (e.g., 10.67 for 3-bit).
        """
        return 32.0 / self.bit_width

    def memory_bytes(self, n_vectors: int) -> int:
        """Estimated compressed memory in bytes for n_vectors.

        Args:
            n_vectors: Number of vectors stored.

        Returns:
            Approximate memory usage in bytes.
        """
        bits_per_vector = self.bit_width * self.dim
        # Add 32 bits for norm storage
        bits_per_vector += 32
        return (n_vectors * bits_per_vector + 7) // 8
