"""GQA error amplification benchmark.

Measures how quantization error amplifies when KV heads are shared across
multiple query heads (Grouped Query Attention), and how bumping key bits
mitigates this.

Outputs:
    benchmarks/gqa_results.json
    stdout table

Usage:
    python benchmarks/bench_gqa.py
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from turboquant import TurboQuantKVCache

HEAD_DIM = 128
SEQ_LEN = 256
BATCH = 1
N_TRIALS = 5  # average over multiple random seeds

# Bit-width configs: (label, key_bits, value_bits)
BIT_CONFIGS = [
    ("3-bit K / 3-bit V", 3, 3),
    ("4-bit K / 3-bit V (GQA-aware)", 4, 3),
    ("4-bit K / 4-bit V", 4, 4),
]

# GQA ratios: (num_kv_heads, num_query_heads, label)
GQA_RATIOS = [
    (32, 32, "1 (MHA)"),
    (16, 32, "2"),
    (8, 32, "4 (Llama-3)"),
    (4, 32, "8"),
]


def run_gqa_benchmark():
    """Run GQA error amplification sweep."""
    results = []

    for num_kv, num_q, ratio_label in GQA_RATIOS:
        gqa_ratio = num_q // num_kv

        for config_label, k_bits, v_bits in BIT_CONFIGS:
            total_attn_mse = 0.0
            total_out_mse = 0.0

            for trial in range(N_TRIALS):
                torch.manual_seed(trial * 1000)

                keys = torch.randn(BATCH, num_kv, SEQ_LEN, HEAD_DIM)
                values = torch.randn(BATCH, num_kv, SEQ_LEN, HEAD_DIM)
                queries = torch.randn(BATCH, num_q, 1, HEAD_DIM)

                # Ground truth: expand KV to match query heads
                keys_exp = keys.repeat_interleave(gqa_ratio, dim=1)
                values_exp = values.repeat_interleave(gqa_ratio, dim=1)

                scale = HEAD_DIM**-0.5
                true_attn = F.softmax(queries @ keys_exp.transpose(-2, -1) * scale, dim=-1)
                true_out = true_attn @ values_exp

                # Compress KV (at KV-head granularity)
                cache = TurboQuantKVCache(
                    head_dim=HEAD_DIM,
                    key_bit_width=k_bits,
                    value_bit_width=v_bits,
                    residual_length=0,  # isolate GQA effect
                    seed=trial,
                )
                compressed = cache.compress(keys, values)
                k_hat = cache.decompress_keys(compressed)
                v_hat = cache.decompress_values(compressed)

                k_hat_exp = k_hat.repeat_interleave(gqa_ratio, dim=1)
                v_hat_exp = v_hat.repeat_interleave(gqa_ratio, dim=1)

                comp_attn = F.softmax(queries @ k_hat_exp.transpose(-2, -1) * scale, dim=-1)
                comp_out = comp_attn @ v_hat_exp

                total_attn_mse += ((true_attn - comp_attn) ** 2).mean().item()
                total_out_mse += ((true_out - comp_out) ** 2).mean().item()

            results.append(
                {
                    "config": config_label,
                    "gqa_ratio": gqa_ratio,
                    "gqa_ratio_label": ratio_label,
                    "num_kv_heads": num_kv,
                    "num_query_heads": num_q,
                    "key_bits": k_bits,
                    "value_bits": v_bits,
                    "attn_mse": total_attn_mse / N_TRIALS,
                    "output_mse": total_out_mse / N_TRIALS,
                    "head_dim": HEAD_DIM,
                    "seq_len": SEQ_LEN,
                }
            )

    return results


def print_table(results):
    """Pretty-print the GQA benchmark table."""
    title = f"GQA Error Amplification Benchmark (head_dim={HEAD_DIM}, seq_len={SEQ_LEN})"
    print(f"\n{title}")
    print("-" * len(title))

    header = (
        f"{'Config':<30} | {'GQA Ratio':>10} | "
        f"{'Attn MSE':>14} | {'Output MSE':>14} | {'vs 3-bit baseline':>18}"
    )
    print(header)
    print("-" * len(header))

    # Build baseline lookup: attn_mse for "3-bit K / 3-bit V" at each ratio
    baselines = {}
    for r in results:
        if r["key_bits"] == 3 and r["value_bits"] == 3:
            baselines[r["gqa_ratio"]] = r["output_mse"]

    for r in results:
        bl = baselines.get(r["gqa_ratio"], 0)
        if r["key_bits"] == 3 and r["value_bits"] == 3:
            delta = "baseline"
        elif bl > 0:
            pct = (r["output_mse"] - bl) / bl * 100
            delta = f"{pct:+.1f}%"
        else:
            delta = "n/a"

        print(
            f"{r['config']:<30} | "
            f"{r['gqa_ratio_label']:>10} | "
            f"{r['attn_mse']:>14.8f} | "
            f"{r['output_mse']:>14.8f} | "
            f"{delta:>18}"
        )


def main():
    out_dir = Path(__file__).parent

    print("Running GQA error amplification benchmark...")
    results = run_gqa_benchmark()
    print_table(results)

    json_path = out_dir / "gqa_results.json"
    with json_path.open("w") as f:
        json.dump(
            {
                "head_dim": HEAD_DIM,
                "seq_len": SEQ_LEN,
                "n_trials": N_TRIALS,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved: {json_path}")


if __name__ == "__main__":
    main()
