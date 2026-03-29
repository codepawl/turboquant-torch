"""Codebook refinement benchmark: Beta vs Gaussian approximation.

Compares MSE distortion when using exact Beta(d/2, d/2) codebooks
(new, for dim < 256) vs Gaussian-approximation codebooks (old) at
typical LLM head dimensions.

Outputs:
    benchmarks/codebook_results.json
    stdout table

Usage:
    python benchmarks/bench_codebook.py
"""

import json
from pathlib import Path

import numpy as np
import torch

from turboquant.codebook import Codebook, _lloyd_max_normal
from turboquant.mse_quantizer import TurboQuantMSE

DIMS = [64, 96, 128]
BITS = 3
N_VECTORS = 1000
SEED = 42


def gaussian_codebook(bit_width: int, dim: int) -> Codebook:
    """Get Gaussian-approximation codebook (old behavior for all dim >= 64)."""
    cb = _lloyd_max_normal(bit_width)
    scale = 1.0 / np.sqrt(dim)
    return Codebook(
        boundaries=cb.boundaries * scale,
        centroids=cb.centroids * scale,
    )


def measure_mse(dim: int, codebook: Codebook, seed: int) -> float:
    """Measure quantization MSE using a given codebook on random unit vectors."""
    torch.manual_seed(seed)
    x = torch.randn(N_VECTORS, dim)
    x = x / torch.norm(x, dim=-1, keepdim=True)

    # Use the MSE quantizer pipeline (Hadamard rotation + scalar quantize)
    quant = TurboQuantMSE(dim=dim, bit_width=BITS, seed=seed)
    out = quant.quantize(x)
    x_hat = quant.dequantize(out)
    return ((x - x_hat) ** 2).mean().item()


def main():
    out_dir = Path(__file__).parent
    results = []

    title = f"Codebook Refinement Benchmark ({BITS}-bit, {N_VECTORS} unit vectors)"
    print(f"\n{title}")
    print("-" * len(title))
    header = f"{'dim':>6} | {'Gaussian MSE':>14} | {'Beta MSE':>14} | {'Improvement':>12}"
    print(header)
    print("-" * len(header))

    for dim in DIMS:
        # Gaussian approximation (old behavior)
        torch.manual_seed(SEED)
        x = torch.randn(N_VECTORS, dim)
        x = x / torch.norm(x, dim=-1, keepdim=True)

        # Old: force Gaussian codebook
        quant_gauss = TurboQuantMSE(dim=dim, bit_width=BITS, seed=SEED)
        # Temporarily swap in Gaussian codebook
        gauss_cb = gaussian_codebook(BITS, dim)
        orig_boundaries = quant_gauss.codebook._boundaries.clone()
        orig_centroids = quant_gauss.codebook._centroids.clone()
        quant_gauss.codebook._boundaries = torch.from_numpy(gauss_cb.boundaries).float()
        quant_gauss.codebook._centroids = torch.from_numpy(gauss_cb.centroids).float()
        out_g = quant_gauss.quantize(x)
        x_hat_g = quant_gauss.dequantize(out_g)
        mse_gauss = ((x - x_hat_g) ** 2).mean().item()
        quant_gauss.codebook._boundaries = orig_boundaries
        quant_gauss.codebook._centroids = orig_centroids

        # New: Beta codebook (current behavior via get_codebook)
        quant_beta = TurboQuantMSE(dim=dim, bit_width=BITS, seed=SEED)
        out_b = quant_beta.quantize(x)
        x_hat_b = quant_beta.dequantize(out_b)
        mse_beta = ((x - x_hat_b) ** 2).mean().item()

        improvement = (1 - mse_beta / mse_gauss) * 100 if mse_gauss > 0 else 0

        results.append(
            {
                "dim": dim,
                "bits": BITS,
                "mse_gaussian": mse_gauss,
                "mse_beta": mse_beta,
                "improvement_pct": improvement,
                "n_vectors": N_VECTORS,
            }
        )

        print(f"{dim:>6} | {mse_gauss:>14.8f} | {mse_beta:>14.8f} | {improvement:>+11.1f}%")

    json_path = out_dir / "codebook_results.json"
    with json_path.open("w") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"\nSaved: {json_path}")


if __name__ == "__main__":
    main()
