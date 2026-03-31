"""Benchmark regression tracker.

Runs standardized benchmarks and compares against stored baselines.
Not run in CI (too slow), but run manually before releases.

Usage:
    python benchmarks/bench_regression.py          # run and compare
    python benchmarks/bench_regression.py --save    # save new baseline
"""

import json
import sys
import time
from pathlib import Path

import torch

from turboquant import TurboQuant, TurboQuantKVCache

BASELINE_PATH = Path("benchmarks/regression_baseline.json")


def bench_quantize_speed(dim=128, bit_width=3, n=1000, repeats=5):
    """Measure quantize throughput (vectors/sec)."""
    tq = TurboQuant(dim=dim, bit_width=bit_width, unbiased=True)
    x = torch.randn(n, dim)

    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        tq.quantize(x)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg = sum(times) / len(times)
    return {"vectors_per_sec": n / avg, "avg_ms": avg * 1000}


def bench_kv_compress_speed(head_dim=128, bit_width=3, seq_len=512, heads=8, repeats=5):
    """Measure KV cache compress throughput."""
    cache = TurboQuantKVCache(head_dim=head_dim, bit_width=bit_width, residual_length=0)
    k = torch.randn(1, heads, seq_len, head_dim)
    v = torch.randn(1, heads, seq_len, head_dim)

    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        cache.compress(k, v)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg = sum(times) / len(times)
    return {"compress_ms": avg * 1000, "tokens_per_sec": seq_len * heads / avg}


def bench_distortion(dim=128, bit_width=3, n=1000):
    """Measure MSE distortion on unit vectors."""
    torch.manual_seed(42)
    x = torch.randn(n, dim)
    x = x / torch.norm(x, dim=-1, keepdim=True)

    tq = TurboQuant(dim=dim, bit_width=bit_width, unbiased=True, seed=0)
    x_hat = tq.dequantize(tq.quantize(x))
    mse = ((x - x_hat) ** 2).sum(dim=-1).mean().item()
    return {"mse": mse}


def run_all():
    results = {
        "quantize_128_3bit": bench_quantize_speed(128, 3),
        "quantize_256_3bit": bench_quantize_speed(256, 3),
        "kv_compress_128_3bit": bench_kv_compress_speed(128, 3),
        "kv_compress_128_3bit_outlier": bench_kv_compress_speed(128, 3),
        "distortion_128_2bit": bench_distortion(128, 2),
        "distortion_128_3bit": bench_distortion(128, 3),
        "distortion_128_4bit": bench_distortion(128, 4),
    }
    return results  # noqa: RET504


def compare(current, baseline, threshold=0.3):
    """Compare current vs baseline. Flag regressions > threshold."""
    regressions = []
    for key in current:
        if key not in baseline:
            continue
        for metric in current[key]:
            if metric not in baseline[key]:
                continue
            cur = current[key][metric]
            base = baseline[key][metric]
            if base == 0:
                continue

            # For speed metrics (*_per_sec), lower is worse
            # For error metrics (mse, *_ms), higher is worse
            if "per_sec" in metric:
                change = (cur - base) / base
                if change < -threshold:
                    regressions.append(
                        f"  {key}.{metric}: {base:.2f} -> {cur:.2f} "
                        f"({change * 100:+.1f}% REGRESSION)"
                    )
            elif "mse" in metric:
                change = (cur - base) / base
                if change > threshold:
                    regressions.append(
                        f"  {key}.{metric}: {base:.6f} -> {cur:.6f} "
                        f"({change * 100:+.1f}% REGRESSION)"
                    )
    return regressions


def main():
    save_mode = "--save" in sys.argv

    print("Running benchmarks...")
    results = run_all()

    if save_mode:
        BASELINE_PATH.write_text(json.dumps(results, indent=2))
        print(f"Baseline saved to {BASELINE_PATH}")
        return

    # Print results
    for key, metrics in results.items():
        print(f"\n  {key}:")
        for metric, value in metrics.items():
            print(f"    {metric}: {value:.4f}")

    # Compare against baseline
    if BASELINE_PATH.exists():
        baseline = json.loads(BASELINE_PATH.read_text())
        regressions = compare(results, baseline)
        if regressions:
            print("\nREGRESSIONS DETECTED:")
            for r in regressions:
                print(r)
            sys.exit(1)
        else:
            print("\nNo regressions detected.")
    else:
        print("\nNo baseline found. Run with --save to create one.")


if __name__ == "__main__":
    main()
