"""Sliding window (residual buffer) benchmark.

Measures attention accuracy vs fp32 ground truth at different residual_length
values: 0, 32, 64, 128, 256.

Outputs:
    benchmarks/sliding_window_results.json
    stdout table

Usage:
    python benchmarks/bench_sliding_window.py
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from turboquant import TurboQuantKVCache

RESIDUAL_LENGTHS = [0, 32, 64, 128, 256]
BIT_WIDTH = 3
BATCH_SIZE = 1


def _bench_layers(config_name, keys_values, head_dim, num_heads, seq_len, n_layers):
    """Run sliding-window sweep on a list of (keys, values) layer tensors."""
    results = []

    for res_len in RESIDUAL_LENGTHS:
        cache = TurboQuantKVCache(
            head_dim=head_dim,
            bit_width=BIT_WIDTH,
            residual_length=res_len,
        )

        total_attn_mse = 0.0
        total_out_mse = 0.0

        for layer_idx in range(n_layers):
            keys, values = keys_values[layer_idx]

            query = torch.randn(BATCH_SIZE, num_heads, 1, head_dim)
            scale = head_dim**-0.5

            # Ground truth
            true_attn = F.softmax(query @ keys.transpose(-2, -1) * scale, dim=-1)
            true_out = true_attn @ values

            # Compressed
            compressed = cache.compress(keys, values)
            keys_hat = cache.decompress_keys(compressed)
            values_hat = cache.decompress_values(compressed)

            comp_attn = F.softmax(query @ keys_hat.transpose(-2, -1) * scale, dim=-1)
            comp_out = comp_attn @ values_hat

            total_attn_mse += ((true_attn - comp_attn) ** 2).mean().item()
            total_out_mse += ((true_out - comp_out) ** 2).mean().item()

        _, comp_mb, _ = cache.memory_savings(BATCH_SIZE, num_heads, seq_len)
        comp_mb_total = comp_mb * n_layers

        results.append(
            {
                "config": config_name,
                "residual_length": res_len,
                "bit_width": BIT_WIDTH,
                "attn_mse": total_attn_mse / n_layers,
                "output_mse": total_out_mse / n_layers,
                "comp_mb": round(comp_mb_total, 2),
                "layers": n_layers,
                "heads": num_heads,
                "seq_len": seq_len,
                "head_dim": head_dim,
            }
        )

    return results


def run_real_model():
    """Benchmark on SmolLM2-135M KV cache."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("  transformers not installed, skipping real model")
        return None, []

    model_name = "HuggingFaceTB/SmolLM2-135M"
    print(f"  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, device_map="cpu"
    )
    model.eval()

    prompt = (
        "The future of artificial intelligence lies in the ability to process "
        "and understand increasingly complex information at scale"
    )
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)

    past_kv = outputs.past_key_values

    # Handle DynamicCache (transformers >= 4.36)
    if hasattr(past_kv, "key_cache"):
        kv_pairs = [(past_kv.key_cache[i], past_kv.value_cache[i]) for i in range(len(past_kv))]
    else:
        kv_pairs = [(item[0], item[1]) for item in past_kv]

    n_layers = len(kv_pairs)
    B, H, S, D = kv_pairs[0][0].shape

    config_name = f"SmolLM2-135M (real, seq={S})"
    print(f"  Config: {n_layers} layers, {H} KV heads, head_dim={D}, seq_len={S}")

    model_info = {
        "name": model_name,
        "layers": n_layers,
        "kv_heads": H,
        "head_dim": D,
        "seq_len": S,
        "prompt": prompt,
    }
    results = _bench_layers(config_name, kv_pairs, D, H, S, n_layers)
    return model_info, results


def run_synthetic():
    """Benchmark on Llama-3-8B-like synthetic tensors."""
    n_layers, num_heads, seq_len, head_dim = 32, 8, 512, 128
    config_name = f"Llama-3-8B-like (synthetic, seq={seq_len})"
    print(f"  Running {config_name}...")

    # Use 4 layers to keep runtime reasonable
    effective_layers = 4
    torch.manual_seed(42)
    kv_pairs = [
        (
            torch.randn(BATCH_SIZE, num_heads, seq_len, head_dim),
            torch.randn(BATCH_SIZE, num_heads, seq_len, head_dim),
        )
        for _ in range(effective_layers)
    ]

    results = _bench_layers(config_name, kv_pairs, head_dim, num_heads, seq_len, effective_layers)

    # Scale memory to full 32-layer model
    for r in results:
        r["comp_mb"] = round(r["comp_mb"] * (n_layers / effective_layers), 2)
        r["layers"] = n_layers

    return results


def print_table(results, title):
    """Pretty-print a results table."""
    baseline_out = results[0]["output_mse"] if results else 0

    print(f"\n{title}")
    print("-" * len(title))
    header = f"{'residual_length':>16} | {'Attn Score MSE':>14} | {'Output MSE':>14} | {'Memory MB':>9} | {'vs full quant':>13}"
    print(header)
    print("-" * len(header))

    for r in results:
        if r["residual_length"] == 0:
            delta = "baseline"
        elif baseline_out > 0:
            pct = (r["output_mse"] - baseline_out) / baseline_out * 100
            delta = f"{pct:+.1f}%"
        else:
            delta = "n/a"

        print(
            f"{r['residual_length']:>16} | "
            f"{r['attn_mse']:>14.8f} | "
            f"{r['output_mse']:>14.8f} | "
            f"{r['comp_mb']:>9.1f} | "
            f"{delta:>13}"
        )


def main():
    out_dir = Path(__file__).parent
    all_results = []

    # Real model
    print("Running real model sliding window benchmark...")
    model_info, real_results = run_real_model()
    if real_results:
        all_results.extend(real_results)
        seq = real_results[0]["seq_len"]
        print_table(
            real_results,
            f"Sliding Window Benchmark ({model_info['name']}, {BIT_WIDTH}-bit, seq_len={seq})",
        )

    # Synthetic
    print("\nRunning synthetic sliding window benchmark...")
    synth_results = run_synthetic()
    all_results.extend(synth_results)
    seq = synth_results[0]["seq_len"]
    print_table(
        synth_results,
        f"Sliding Window Benchmark (Llama-3-8B-like synthetic, {BIT_WIDTH}-bit, seq_len={seq})",
    )

    # Save JSON
    json_path = out_dir / "sliding_window_results.json"
    with json_path.open("w") as f:
        json.dump({"model_info": model_info, "results": all_results}, f, indent=2)
    print(f"\nSaved: {json_path}")


if __name__ == "__main__":
    main()
