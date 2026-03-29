"""Pre-RoPE vs Post-RoPE key quantization benchmark.

Compares attention output MSE against fp32 ground truth at various
sequence lengths, measuring whether quantizing keys before RoPE
preserves more accuracy than quantizing after.

Outputs:
    benchmarks/pre_rope_results.json
    stdout table

Usage:
    python benchmarks/bench_pre_rope.py
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from turboquant import TurboQuantKVCache
from turboquant.rope import apply_rope, compute_rope_frequencies

SEQ_LENGTHS = [128, 512, 1024, 2048]
HEAD_DIM = 128
NUM_HEADS = 8
BIT_WIDTH = 3
BATCH = 1
N_TRIALS = 3


def bench_one(seq_len: int, trial: int) -> dict:
    """Run Pre-RoPE vs Post-RoPE comparison for one config."""
    torch.manual_seed(trial * 1000 + seq_len)
    freqs = compute_rope_frequencies(HEAD_DIM, max_seq_len=seq_len + 1)
    positions = torch.arange(seq_len)
    query_pos = torch.tensor([seq_len])

    keys_raw = torch.randn(BATCH, NUM_HEADS, seq_len, HEAD_DIM)
    values = torch.randn(BATCH, NUM_HEADS, seq_len, HEAD_DIM)
    query_raw = torch.randn(BATCH, NUM_HEADS, 1, HEAD_DIM)

    # Ground truth: apply RoPE then fp32 attention
    keys_roped = apply_rope(keys_raw, freqs, positions)
    query_roped = apply_rope(query_raw, freqs, query_pos)
    scale = HEAD_DIM**-0.5
    true_attn = F.softmax(query_roped @ keys_roped.transpose(-2, -1) * scale, dim=-1)
    true_out = true_attn @ values

    # Post-RoPE: quantize AFTER RoPE
    cache_post = TurboQuantKVCache(
        head_dim=HEAD_DIM, bit_width=BIT_WIDTH, residual_length=0, seed=trial
    )
    comp_post = cache_post.compress(keys_roped, values)
    out_post = cache_post.attention(query_roped, comp_post)
    mse_post = ((true_out - out_post) ** 2).mean().item()

    # Pre-RoPE: quantize BEFORE RoPE
    cache_pre = TurboQuantKVCache(
        head_dim=HEAD_DIM, bit_width=BIT_WIDTH, pre_rope=True, residual_length=0, seed=trial
    )
    comp_pre = cache_pre.compress(keys_raw, values, positions=positions, rope_freqs=freqs)
    out_pre = cache_pre.attention(query_raw, comp_pre, query_positions=query_pos, rope_freqs=freqs)
    mse_pre = ((true_out - out_pre) ** 2).mean().item()

    return {"mse_post": mse_post, "mse_pre": mse_pre}


def main():
    out_dir = Path(__file__).parent
    results = []

    print(
        f"Pre-RoPE vs Post-RoPE Benchmark (head_dim={HEAD_DIM}, {BIT_WIDTH}-bit, {NUM_HEADS} heads)"
    )
    print("-" * 80)
    header = f"{'seq_len':>8} | {'Post-RoPE MSE':>14} | {'Pre-RoPE MSE':>14} | {'Δ':>10}"
    print(header)
    print("-" * len(header))

    for seq_len in SEQ_LENGTHS:
        print(f"  Running seq_len={seq_len}...", end="", flush=True)
        total_post = 0.0
        total_pre = 0.0
        for trial in range(N_TRIALS):
            r = bench_one(seq_len, trial)
            total_post += r["mse_post"]
            total_pre += r["mse_pre"]

        avg_post = total_post / N_TRIALS
        avg_pre = total_pre / N_TRIALS
        delta_pct = (avg_pre - avg_post) / avg_post * 100 if avg_post > 0 else 0

        results.append(
            {
                "seq_len": seq_len,
                "mse_post_rope": avg_post,
                "mse_pre_rope": avg_pre,
                "delta_pct": delta_pct,
                "head_dim": HEAD_DIM,
                "bit_width": BIT_WIDTH,
                "num_heads": NUM_HEADS,
                "n_trials": N_TRIALS,
            }
        )

        print(f"\r{seq_len:>8} | {avg_post:>14.8f} | {avg_pre:>14.8f} | {delta_pct:>+9.1f}%")

    json_path = out_dir / "pre_rope_results.json"
    with json_path.open("w") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"\nSaved: {json_path}")


if __name__ == "__main__":
    main()
