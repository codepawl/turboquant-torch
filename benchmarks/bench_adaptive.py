"""Adaptive per-layer bit allocation benchmark.

Compares uniform 3-bit vs gradient 2-4 bit allocation strategies
across simulated multi-layer KV caches. Measures per-layer MSE
and overall attention quality.

Outputs:
    benchmarks/adaptive_results.json
    stdout table

Usage:
    python benchmarks/bench_adaptive.py
"""

import json
from pathlib import Path

import torch

from turboquant.adaptive import AdaptiveKVCache, gradient_allocation, uniform_allocation

HEAD_DIM = 128
SEQ_LEN = 256
BATCH = 1
NUM_HEADS = 4
N_LAYERS = 32
N_TRIALS = 3


def run_adaptive_benchmark() -> list[dict]:
    """Compare allocation strategies."""
    strategies = {
        "uniform-3bit": uniform_allocation(N_LAYERS, 3),
        "gradient-linear-2to4": gradient_allocation(N_LAYERS, 2, 4, "linear"),
        "gradient-step-2to4": gradient_allocation(N_LAYERS, 2, 4, "step"),
        "uniform-4bit": uniform_allocation(N_LAYERS, 4),
    }

    results = []

    for name, allocation in strategies.items():
        avg_bits = sum(allocation) / len(allocation)
        layer_mses = []

        for trial in range(N_TRIALS):
            gen = torch.Generator().manual_seed(trial)
            cache = AdaptiveKVCache(
                head_dim=HEAD_DIM,
                layer_bits=allocation,
                residual_length=0,
            )

            trial_mses = []
            for layer_idx in range(N_LAYERS):
                keys = torch.randn(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM, generator=gen)
                values = torch.randn(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM, generator=gen)

                compressed = cache.compress_layer(layer_idx, keys, values)
                k_hat = cache.decompress_layer_keys(layer_idx, compressed)
                mse = ((keys - k_hat) ** 2).mean().item()
                trial_mses.append(mse)

            layer_mses.append(trial_mses)

        # Average across trials
        avg_layer_mses = [
            sum(layer_mses[t][layer] for t in range(N_TRIALS)) / N_TRIALS
            for layer in range(N_LAYERS)
        ]

        result = {
            "strategy": name,
            "avg_bits": avg_bits,
            "allocation": allocation,
            "overall_mse": sum(avg_layer_mses) / len(avg_layer_mses),
            "early_layer_mse": sum(avg_layer_mses[: N_LAYERS // 3]) / (N_LAYERS // 3),
            "late_layer_mse": sum(avg_layer_mses[-N_LAYERS // 3 :]) / (N_LAYERS // 3),
            "per_layer_mse": avg_layer_mses,
        }
        results.append(result)

    return results


def print_results(results: list[dict]) -> None:
    """Print results as formatted table."""
    print(f"\n{'=' * 75}")
    print("Adaptive Per-Layer Bit Allocation Benchmark")
    print(f"{'=' * 75}")
    print(f"Config: {N_LAYERS} layers, head_dim={HEAD_DIM}, seq_len={SEQ_LEN}")
    print(f"        {N_TRIALS} trials averaged")
    print(f"{'=' * 75}")

    print(f"\n  {'Strategy':<25} {'Avg bits':>8} {'MSE':>10} {'Early MSE':>10} {'Late MSE':>10}")
    print(f"  {'-' * 65}")

    for r in results:
        print(
            f"  {r['strategy']:<25} {r['avg_bits']:>8.1f} "
            f"{r['overall_mse']:>10.6f} "
            f"{r['early_layer_mse']:>10.6f} "
            f"{r['late_layer_mse']:>10.6f}"
        )


def main() -> None:
    results = run_adaptive_benchmark()

    # Save without per_layer_mse arrays for readability
    save_results = []
    for r in results:
        save_r = {k: v for k, v in r.items() if k != "per_layer_mse"}
        save_results.append(save_r)

    out_path = Path(__file__).parent / "adaptive_results.json"
    with out_path.open("w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print_results(results)


if __name__ == "__main__":
    main()
