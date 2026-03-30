"""Outlier channel routing benchmark.

Compares quantization MSE with and without outlier routing across
different n_outlier_channels settings. Tests on synthetic data with
realistic outlier distributions (mimicking attention sink channels).

Outputs:
    benchmarks/outlier_results.json
    stdout table

Usage:
    python benchmarks/bench_outlier.py
"""

import json
from pathlib import Path

import torch

from turboquant import TurboQuantKVCache

HEAD_DIM = 128
SEQ_LEN = 256
BATCH = 1
NUM_HEADS = 4
N_TRIALS = 5
OUTLIER_CONFIGS = [0, 4, 8, 16, 32]
OUTLIER_MAGNITUDES = [10, 50, 100]  # how much larger outlier channels are


def create_outlier_data(
    head_dim: int,
    n_real_outliers: int,
    magnitude: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create KV tensors with synthetic outlier channels."""
    gen = torch.Generator().manual_seed(seed)
    keys = torch.randn(BATCH, NUM_HEADS, SEQ_LEN, head_dim, generator=gen)
    values = torch.randn(BATCH, NUM_HEADS, SEQ_LEN, head_dim, generator=gen)

    # Make first n_real_outliers channels have large magnitude
    for i in range(n_real_outliers):
        keys[:, :, :, i] *= magnitude
        values[:, :, :, i] *= magnitude

    return keys, values


def run_outlier_benchmark() -> list[dict]:
    """Run outlier routing sweep."""
    results = []

    for magnitude in OUTLIER_MAGNITUDES:
        n_real_outliers = 4  # ground truth outlier count

        for n_outlier_channels in OUTLIER_CONFIGS:
            key_mses = []
            val_mses = []

            for trial in range(N_TRIALS):
                keys, values = create_outlier_data(HEAD_DIM, n_real_outliers, magnitude, seed=trial)

                cache = TurboQuantKVCache(
                    head_dim=HEAD_DIM,
                    bit_width=3,
                    n_outlier_channels=n_outlier_channels,
                    residual_length=0,
                )

                compressed = cache.compress(keys, values)
                k_hat = cache.decompress_keys(compressed)
                v_hat = cache.decompress_values(compressed)

                key_mse = ((keys - k_hat) ** 2).mean().item()
                val_mse = ((values - v_hat) ** 2).mean().item()
                key_mses.append(key_mse)
                val_mses.append(val_mse)

            result = {
                "outlier_magnitude": magnitude,
                "n_outlier_channels": n_outlier_channels,
                "key_mse_mean": sum(key_mses) / len(key_mses),
                "val_mse_mean": sum(val_mses) / len(val_mses),
            }
            results.append(result)

    return results


def print_results(results: list[dict]) -> None:
    """Print results as formatted table."""
    print(f"\n{'=' * 70}")
    print("Outlier Channel Routing Benchmark")
    print(f"{'=' * 70}")
    print(f"Config: head_dim={HEAD_DIM}, seq_len={SEQ_LEN}, bit_width=3")
    print(f"        {N_TRIALS} trials averaged")
    print(f"{'=' * 70}")

    for magnitude in OUTLIER_MAGNITUDES:
        print(f"\n  Outlier magnitude: {magnitude}x")
        print(f"  {'n_outliers':>12} {'Key MSE':>12} {'Val MSE':>12} {'Key Δ%':>10}")
        print(f"  {'-' * 48}")

        mag_results = [r for r in results if r["outlier_magnitude"] == magnitude]
        baseline_mse = mag_results[0]["key_mse_mean"]  # n_outliers=0

        for r in mag_results:
            delta = ((r["key_mse_mean"] - baseline_mse) / baseline_mse) * 100
            sign = "+" if delta > 0 else ""
            print(
                f"  {r['n_outlier_channels']:>12} "
                f"{r['key_mse_mean']:>12.6f} "
                f"{r['val_mse_mean']:>12.6f} "
                f"{sign}{delta:>9.1f}%"
            )


def main() -> None:
    results = run_outlier_benchmark()

    out_path = Path(__file__).parent / "outlier_results.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print_results(results)


if __name__ == "__main__":
    main()
