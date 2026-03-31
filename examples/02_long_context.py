"""Serving Long Context Without Running Out of Memory.

Shows how KV cache memory scales with sequence length, and how
TurboQuant's sliding window and outlier routing reduce it.

Requirements:
    pip install "turboquant-torch[hf]" matplotlib

Runs on CPU (Colab free tier). Uses SmolLM2-135M.
"""

from __future__ import annotations

import sys


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM

    from turboquant import TurboQuantKVCache
    from turboquant.compat import detect_model_kv_info

    MODEL = "HuggingFaceTB/SmolLM2-135M"

    # ── Load model ────────────────────────────────────────────────────────
    print(f"Loading {MODEL}...")
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
    model.eval()
    info = detect_model_kv_info(model)
    print(f"  head_dim={info.head_dim}, layers={info.n_layers}, "
          f"kv_heads={info.num_kv_heads}")
    print()

    # ── Memory scaling with sequence length ───────────────────────────────
    print("=" * 60)
    print("  KV CACHE MEMORY vs SEQUENCE LENGTH")
    print("=" * 60)
    seq_lens = [128, 512, 1024, 2048, 4096]
    fp32_mbs = []
    tq3_mbs = []

    cache_3bit = TurboQuantKVCache(
        head_dim=info.head_dim, bit_width=3, residual_length=0,
    )

    print(f"  {'Seq Len':>8}  {'FP32 (MB)':>10}  {'3-bit (MB)':>10}  {'Ratio':>6}")
    print(f"  {'---':>8}  {'---':>10}  {'---':>10}  {'---':>6}")

    for seq_len in seq_lens:
        orig, comp, ratio = cache_3bit.memory_savings(
            1, info.num_kv_heads, seq_len,
        )
        orig_total = orig * info.n_layers
        comp_total = comp * info.n_layers
        fp32_mbs.append(orig_total)
        tq3_mbs.append(comp_total)
        print(f"  {seq_len:>8}  {orig_total:>10.1f}  {comp_total:>10.1f}  {ratio:>5.1f}x")

    print()

    # ── Plot if matplotlib available ──────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(seq_lens, fp32_mbs, "o-", label="FP32", linewidth=2)
        ax.plot(seq_lens, tq3_mbs, "s-", label="3-bit TurboQuant", linewidth=2)
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("KV Cache Memory (MB)")
        ax.set_title(f"KV Cache Memory Scaling -- {MODEL}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig("memory_scaling.png", dpi=120)
        print("  Plot saved: memory_scaling.png")
    except ImportError:
        print("  (install matplotlib for plots)")
    print()

    # ── Sliding window effect ─────────────────────────────────────────────
    print("=" * 60)
    print("  SLIDING WINDOW (residual_length)")
    print("=" * 60)
    print("  Recent tokens stay in fp16 for higher accuracy.")
    print()

    seq_len = 2048
    print(f"  Sequence length: {seq_len}")
    print(f"  {'Config':>30}  {'Memory (MB)':>12}  {'Ratio':>6}")
    print(f"  {'---':>30}  {'---':>12}  {'---':>6}")

    for res_len in [0, 64, 128, 256]:
        cache = TurboQuantKVCache(
            head_dim=info.head_dim, bit_width=3, residual_length=res_len,
        )
        orig, comp, ratio = cache.memory_savings(1, info.num_kv_heads, seq_len)
        comp_total = comp * info.n_layers
        label = f"3-bit, residual={res_len}"
        print(f"  {label:>30}  {comp_total:>12.1f}  {ratio:>5.1f}x")

    print()

    # ── Outlier routing effect ────────────────────────────────────────────
    print("=" * 60)
    print("  OUTLIER ROUTING (n_outlier_channels)")
    print("=" * 60)
    print("  Top-k magnitude channels stay in fp16.")
    print()

    torch.manual_seed(42)
    k = torch.randn(1, info.num_kv_heads, 512, info.head_dim)
    v = torch.randn(1, info.num_kv_heads, 512, info.head_dim)
    # Inject outliers in channels 0 and 1
    k[:, :, :, 0] *= 50
    k[:, :, :, 1] *= 30

    print(f"  {'Config':>35}  {'Key MSE':>10}  {'Value MSE':>10}")
    print(f"  {'---':>35}  {'---':>10}  {'---':>10}")

    for n_outlier in [0, 4, 8]:
        cache = TurboQuantKVCache(
            head_dim=info.head_dim, bit_width=3,
            residual_length=0, n_outlier_channels=n_outlier,
        )
        compressed = cache.compress(k.float(), v.float())
        k_hat = cache.decompress_keys(compressed)
        v_hat = cache.decompress_values(compressed)
        k_mse = ((k - k_hat) ** 2).mean().item()
        v_mse = ((v - v_hat) ** 2).mean().item()
        label = f"3-bit, outliers={n_outlier}"
        print(f"  {label:>35}  {k_mse:>10.4f}  {v_mse:>10.4f}")

    print()

    # ── Summary table ─────────────────────────────────────────────────────
    print("=" * 60)
    print("  SUMMARY: FEATURE COMBINATIONS")
    print("=" * 60)

    configs = [
        ("FP32 baseline", 32, 0, 0),
        ("3-bit", 3, 0, 0),
        ("3-bit + sliding window", 3, 128, 0),
        ("3-bit + outlier routing", 3, 0, 8),
        ("3-bit + both", 3, 128, 8),
    ]

    print(f"  {'Config':>30}  {'Memory':>8}  {'Ratio':>6}")
    print(f"  {'---':>30}  {'---':>8}  {'---':>6}")

    for label, bw, res, outlier in configs:
        if bw == 32:
            cache = TurboQuantKVCache(
                head_dim=info.head_dim, bit_width=3, residual_length=0,
            )
            orig, _, _ = cache.memory_savings(1, info.num_kv_heads, 2048)
            print(f"  {label:>30}  {orig * info.n_layers:>7.1f}M  {'1.0x':>6}")
        else:
            cache = TurboQuantKVCache(
                head_dim=info.head_dim, bit_width=bw,
                residual_length=res, n_outlier_channels=outlier,
            )
            orig, comp, ratio = cache.memory_savings(
                1, info.num_kv_heads, 2048,
            )
            print(f"  {label:>30}  {comp * info.n_layers:>7.1f}M  {ratio:>5.1f}x")

    print()


if __name__ == "__main__":
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        print('Install dependencies: pip install "turboquant-torch[hf]"')
        sys.exit(1)
    main()
