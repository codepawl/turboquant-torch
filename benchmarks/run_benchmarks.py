"""TurboQuant KV cache benchmarks on real and simulated model configurations.

Outputs:
    benchmarks/results.json  — raw benchmark data
    benchmarks/results.md    — formatted markdown table

Usage:
    python benchmarks/run_benchmarks.py
"""

import json
import time
from pathlib import Path

import torch

from turboquant import TurboQuantKVCache

# ── Simulated model configurations ────────────────────────────────────────
# (name, layers, heads, seq_len, head_dim)
SYNTHETIC_CONFIGS = [
    ("Llama-7B 2K ctx", 32, 32, 2048, 128),
    ("Llama-7B 8K ctx", 32, 32, 8192, 128),
    ("Llama-7B 32K ctx", 32, 32, 32768, 128),
    ("Llama-70B 2K ctx", 80, 64, 2048, 128),
]

BIT_WIDTHS = [2, 3, 4]
BATCH_SIZE = 1


def benchmark_config(name, n_layers, num_heads, seq_len, head_dim, keys_values=None):
    """Run compression benchmarks for a single config across bit widths.

    If keys_values is provided (list of (key, value) tuples), use real tensors.
    Otherwise generate synthetic random tensors (one layer at a time to save memory).
    """
    results = []

    for bit_width in BIT_WIDTHS:
        cache = TurboQuantKVCache(head_dim=head_dim, bit_width=bit_width)

        total_key_mse = 0.0
        total_val_mse = 0.0
        total_attn_mse = 0.0
        total_compress_ms = 0.0
        total_decompress_ms = 0.0

        for layer_idx in range(n_layers):
            if keys_values is not None:
                keys, values = keys_values[layer_idx]
            else:
                # Generate one layer at a time to avoid OOM for large configs
                keys = torch.randn(BATCH_SIZE, num_heads, seq_len, head_dim)
                values = torch.randn(BATCH_SIZE, num_heads, seq_len, head_dim)

            # Compress
            t0 = time.perf_counter()
            compressed = cache.compress(keys, values)
            total_compress_ms += (time.perf_counter() - t0) * 1000

            # Decompress
            t0 = time.perf_counter()
            keys_hat = cache.decompress_keys(compressed)
            values_hat = cache.decompress_values(compressed)
            total_decompress_ms += (time.perf_counter() - t0) * 1000

            # MSE
            total_key_mse += ((keys - keys_hat) ** 2).mean().item()
            total_val_mse += ((values - values_hat) ** 2).mean().item()

            # Attention score MSE (single random query)
            query = torch.randn(BATCH_SIZE, num_heads, 1, head_dim)
            scale = head_dim**-0.5
            attn_orig = torch.softmax(torch.matmul(query, keys.transpose(-2, -1)) * scale, dim=-1)
            attn_comp = torch.softmax(
                torch.matmul(query, keys_hat.transpose(-2, -1)) * scale, dim=-1
            )
            total_attn_mse += ((attn_orig - attn_comp) ** 2).mean().item()

        orig_mb, comp_mb, ratio = cache.memory_savings(BATCH_SIZE, num_heads, seq_len)
        orig_mb_total = orig_mb * n_layers
        comp_mb_total = comp_mb * n_layers

        result = {
            "config": name,
            "bit_width": bit_width,
            "layers": n_layers,
            "heads": num_heads,
            "seq_len": seq_len,
            "head_dim": head_dim,
            "key_mse": total_key_mse / n_layers,
            "val_mse": total_val_mse / n_layers,
            "attn_mse": total_attn_mse / n_layers,
            "orig_mb": round(orig_mb_total, 2),
            "comp_mb": round(comp_mb_total, 2),
            "ratio": round(ratio, 1),
            "compress_ms": round(total_compress_ms, 1),
            "decompress_ms": round(total_decompress_ms, 1),
        }
        results.append(result)

    return results


def run_real_model_benchmark():
    """Run benchmark on real SmolLM2-135M KV cache tensors."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("  transformers not installed, skipping real model benchmark")
        return None, []

    model_name = "HuggingFaceTB/SmolLM2-135M"
    print(f"  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32, device_map="cpu")
    model.eval()

    prompt = "The future of artificial intelligence lies in efficient compression"
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)

    past_kv = outputs.past_key_values
    kv_pairs = [(item[0], item[1]) for item in past_kv]
    n_layers = len(kv_pairs)
    B, H, S, D = kv_pairs[0][0].shape

    config_name = f"SmolLM2-135M (real, seq={S})"
    print(f"  Config: {n_layers} layers, {H} KV heads, head_dim={D}, seq_len={S}")

    results = benchmark_config(config_name, n_layers, H, S, D, keys_values=kv_pairs)

    model_info = {
        "name": model_name,
        "layers": n_layers,
        "kv_heads": H,
        "head_dim": D,
        "seq_len": S,
        "prompt": prompt,
    }
    return model_info, results


def format_results_table(all_results, model_info):
    """Format results as markdown tables."""
    lines = ["# TurboQuant KV Cache Benchmark Results\n"]

    if model_info:
        lines.append(
            f"## Real Model: [{model_info['name']}](https://huggingface.co/{model_info['name']})\n"
        )
        lines.append(f"- Layers: {model_info['layers']}")
        lines.append(f"- KV Heads: {model_info['kv_heads']}")
        lines.append(f"- Head dim: {model_info['head_dim']}")
        lines.append(f"- Prompt tokens: {model_info['seq_len']}")
        lines.append("")

    # Group by config
    configs = {}
    for r in all_results:
        configs.setdefault(r["config"], []).append(r)

    for config_name, results in configs.items():
        lines.append(f"### {config_name}\n")
        lines.append(
            "| Bit-width | Key MSE | Value MSE | Attn Score MSE | "
            "Original (MB) | Compressed (MB) | Ratio | "
            "Compress (ms) | Decompress (ms) |"
        )
        lines.append(
            "|-----------|---------|-----------|----------------|"
            "---------------|-----------------|-------|"
            "---------------|-----------------|"
        )

        for r in results:
            lines.append(
                f"| {r['bit_width']}-bit "
                f"| {r['key_mse']:.6f} "
                f"| {r['val_mse']:.6f} "
                f"| {r['attn_mse']:.8f} "
                f"| {r['orig_mb']:.2f} "
                f"| {r['comp_mb']:.2f} "
                f"| {r['ratio']:.1f}x "
                f"| {r['compress_ms']:.1f} "
                f"| {r['decompress_ms']:.1f} |"
            )
        lines.append("")

    return "\n".join(lines)


def main():
    all_results = []
    out_dir = Path(__file__).parent

    # ── Real model benchmark ──────────────────────────────────────────────
    print("Running real model benchmark...")
    model_info, real_results = run_real_model_benchmark()
    all_results.extend(real_results)

    # ── Synthetic benchmarks ──────────────────────────────────────────────
    for name, layers, heads, seq_len, head_dim in SYNTHETIC_CONFIGS:
        print(f"Running {name}...")
        # For very large configs (32K+), only benchmark 1 layer to avoid OOM
        effective_layers = 1 if seq_len > 16384 else min(layers, 4)
        results = benchmark_config(name, effective_layers, heads, seq_len, head_dim)
        # Scale timing to full model estimate
        for r in results:
            scale = layers / effective_layers
            r["compress_ms"] = round(r["compress_ms"] * scale, 1)
            r["decompress_ms"] = round(r["decompress_ms"] * scale, 1)
            # Recalculate memory for full model
            cache = TurboQuantKVCache(head_dim=head_dim, bit_width=r["bit_width"])
            orig_mb, comp_mb, ratio = cache.memory_savings(BATCH_SIZE, heads, seq_len)
            r["orig_mb"] = round(orig_mb * layers, 2)
            r["comp_mb"] = round(comp_mb * layers, 2)
            r["ratio"] = round(ratio, 1)
            r["layers"] = layers

        all_results.extend(results)

    # ── Save results ──────────────────────────────────────────────────────
    json_path = out_dir / "results.json"
    with json_path.open("w") as f:
        json.dump({"model_info": model_info, "results": all_results}, f, indent=2)
    print(f"\nSaved: {json_path}")

    md_content = format_results_table(all_results, model_info)
    md_path = out_dir / "results.md"
    with md_path.open("w") as f:
        f.write(md_content)
    print(f"Saved: {md_path}")

    # ── Print summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Summary (3-bit)")
    print("=" * 70)

    for r in all_results:
        if r["bit_width"] == 3:
            print(
                f"  {r['config']:30s}  "
                f"Key MSE={r['key_mse']:.4f}  "
                f"Attn MSE={r['attn_mse']:.6f}  "
                f"{r['orig_mb']:.0f}MB -> {r['comp_mb']:.0f}MB ({r['ratio']:.1f}x)"
            )


if __name__ == "__main__":
    main()
