"""Smart Compression: Give Sensitive Layers More Bits.

Runs calibration to find per-layer sensitivity, then compares
uniform vs gradient vs calibrated bit allocation strategies.

Requirements:
    pip install "turboquant-torch[hf]" matplotlib

Runs on CPU (Colab free tier). Uses SmolLM2-135M.
"""

from __future__ import annotations

import sys


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from turboquant import TurboQuantKVCache
    from turboquant.adaptive import (
        AdaptiveKVCache,
        calibration_allocation,
        gradient_allocation,
        uniform_allocation,
    )
    from turboquant.compat import detect_model_kv_info, extract_kv

    MODEL = "HuggingFaceTB/SmolLM2-135M"
    CALIBRATION_TEXT = (
        "The transformer architecture uses multi-head attention to process "
        "sequences in parallel. Each attention head computes query, key, and "
        "value projections, then uses scaled dot-product attention to weight "
        "the values. The KV cache stores previously computed keys and values "
        "to avoid redundant computation during autoregressive generation. "
    ) * 4

    # ── Load model ────────────────────────────────────────────────────────
    print(f"Loading {MODEL}...")
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model.eval()
    info = detect_model_kv_info(model)
    print(f"  {info.n_layers} layers, head_dim={info.head_dim}, "
          f"kv_heads={info.num_kv_heads}")
    print()

    # ── Run calibration ───────────────────────────────────────────────────
    print("=" * 60)
    print("  CALIBRATION: per-layer sensitivity")
    print("=" * 60)
    print("  Running forward pass on calibration text...")
    calibrated_bits = calibration_allocation(
        model, tokenizer, CALIBRATION_TEXT, target_avg_bits=3.0,
    )
    avg_bits = sum(calibrated_bits) / len(calibrated_bits)
    bit_counts = {}
    for b in sorted(set(calibrated_bits)):
        bit_counts[b] = calibrated_bits.count(b)
    print(f"  Average bits: {avg_bits:.2f}")
    print(f"  Bit distribution: {bit_counts}")
    print()

    # Show per-layer allocation
    print("  Layer allocation (calibrated):")
    for i, b in enumerate(calibrated_bits):
        bar = "X" * b + "." * (4 - b)
        print(f"    Layer {i:2d}: [{bar}] {b}-bit")
    print()

    # ── Compare strategies ────────────────────────────────────────────────
    print("=" * 60)
    print("  STRATEGY COMPARISON")
    print("=" * 60)

    uniform_bits = uniform_allocation(info.n_layers, bit_width=3)
    gradient_bits = gradient_allocation(
        info.n_layers, min_bits=2, max_bits=4, strategy="linear",
    )

    strategies = {
        "Uniform 3-bit": uniform_bits,
        "Gradient 2->4 bit": gradient_bits,
        "Calibrated": calibrated_bits,
    }

    # Get real KV cache from forward pass
    inputs = tokenizer(CALIBRATION_TEXT, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs, use_cache=True)
    kv_pairs = extract_kv(out.past_key_values)

    # Measure per-layer MSE for each strategy
    strategy_mses: dict[str, list[float]] = {}

    for name, bits in strategies.items():
        layer_mses = []
        for i, (k, v) in enumerate(kv_pairs):
            if k is None:
                layer_mses.append(0.0)
                continue
            cache = TurboQuantKVCache(
                head_dim=info.head_dim, bit_width=bits[i], residual_length=0,
            )
            compressed = cache.compress(k.float(), v.float())
            k_hat = cache.decompress_keys(compressed)
            mse = ((k.float() - k_hat) ** 2).mean().item()
            layer_mses.append(mse)
        strategy_mses[name] = layer_mses

    # Print comparison
    print(f"\n  {'Strategy':>20}  {'Avg Bits':>9}  {'Total MSE':>10}  {'Max MSE':>10}")
    print(f"  {'---':>20}  {'---':>9}  {'---':>10}  {'---':>10}")
    for name, bits in strategies.items():
        mses = strategy_mses[name]
        avg_b = sum(bits) / len(bits)
        total = sum(mses)
        max_mse = max(mses)
        print(f"  {name:>20}  {avg_b:>9.2f}  {total:>10.4f}  {max_mse:>10.4f}")

    print()

    # ── Plot if matplotlib available ──────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Left: bit allocation per layer
        layers = list(range(info.n_layers))
        for name, bits in strategies.items():
            ax1.plot(layers, bits, "o-", label=name, markersize=3)
        ax1.set_xlabel("Layer")
        ax1.set_ylabel("Bit Width")
        ax1.set_title("Bit Allocation per Layer")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yticks([2, 3, 4])

        # Right: MSE per layer
        for name, mses in strategy_mses.items():
            ax2.plot(layers, mses, "o-", label=name, markersize=3)
        ax2.set_xlabel("Layer")
        ax2.set_ylabel("Key MSE")
        ax2.set_title("Reconstruction Error per Layer")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.suptitle(f"Adaptive Compression -- {MODEL}", fontsize=14)
        fig.tight_layout()
        fig.savefig("adaptive_compression.png", dpi=120)
        print("  Plot saved: adaptive_compression.png")
    except ImportError:
        print("  (install matplotlib for plots)")
    print()

    # ── Practical usage ───────────────────────────────────────────────────
    print("=" * 60)
    print("  PRACTICAL USAGE")
    print("=" * 60)
    print()
    print("  # One-liner with auto-calibration:")
    print("  adaptive = AdaptiveKVCache.from_model(")
    print(f"      model, tokenizer, head_dim={info.head_dim},")
    print("      target_avg_bits=3.0,")
    print("  )")
    print("  print(adaptive.summary())")
    print()

    # Actually run it
    adaptive = AdaptiveKVCache.from_model(
        model, tokenizer, head_dim=info.head_dim, target_avg_bits=3.0,
    )
    print("  " + adaptive.summary().replace("\n", "\n  "))
    print()


if __name__ == "__main__":
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        print('Install dependencies: pip install "turboquant-torch[hf]"')
        sys.exit(1)
    main()
