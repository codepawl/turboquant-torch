"""Compress Any LLM's Memory in 3 Lines.

Shows turboquant.wrap() in action: generate text with automatic KV cache
compression. Compare memory usage and output quality before/after.

Requirements:
    pip install "turboquant-torch[hf]"

Runs on CPU (Colab free tier). Uses SmolLM2-135M (~270 MB).
"""

from __future__ import annotations

import sys
import time


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import turboquant

    MODEL = "HuggingFaceTB/SmolLM2-135M"
    PROMPT = "The future of artificial intelligence will"

    # ── Load model ────────────────────────────────────────────────────────
    print(f"Loading {MODEL}...")
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model.eval()
    inputs = tokenizer(PROMPT, return_tensors="pt")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")
    print(f"  Prompt: {PROMPT!r} ({inputs['input_ids'].shape[1]} tokens)\n")

    # ── Baseline: no compression ──────────────────────────────────────────
    print("=" * 60)
    print("  BASELINE (no compression)")
    print("=" * 60)
    with torch.no_grad():
        t0 = time.perf_counter()
        baseline_out = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        baseline_time = time.perf_counter() - t0

    baseline_text = tokenizer.decode(baseline_out[0], skip_special_tokens=True)
    print(f"  Output: {baseline_text!r}")
    print(f"  Time:   {baseline_time:.2f}s")

    # Show KV cache memory for this model
    from turboquant import TurboQuantKVCache
    from turboquant.compat import detect_model_kv_info

    info = detect_model_kv_info(model)
    seq_len = baseline_out.shape[1]
    cache = TurboQuantKVCache(head_dim=info.head_dim, bit_width=3, residual_length=0)
    orig_mb, comp_mb, ratio = cache.memory_savings(1, info.num_kv_heads, seq_len)
    print(f"  KV cache: {orig_mb:.1f} MB (fp32)")
    print()

    # ── With TurboQuant: 3 lines ──────────────────────────────────────────
    print("=" * 60)
    print("  TURBOQUANT (3-bit compression)")
    print("=" * 60)

    # This is all you need:
    wrapped = turboquant.wrap(model, verbose=True)
    with torch.no_grad():
        t0 = time.perf_counter()
        tq_out = wrapped.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        tq_time = time.perf_counter() - t0

    tq_text = tokenizer.decode(tq_out[0], skip_special_tokens=True)
    print(f"  Output: {tq_text!r}")
    print(f"  Time:   {tq_time:.2f}s")
    print(f"  KV cache: {comp_mb:.1f} MB (3-bit) -- {ratio:.1f}x smaller")
    print()

    # ── Comparison ────────────────────────────────────────────────────────
    print("=" * 60)
    print("  COMPARISON")
    print("=" * 60)
    baseline_tokens = baseline_out[0].tolist()
    tq_tokens = tq_out[0].tolist()
    min_len = min(len(baseline_tokens), len(tq_tokens))
    match = sum(baseline_tokens[i] == tq_tokens[i] for i in range(min_len))
    total = len(baseline_tokens)
    print(f"  Token match: {match}/{total} ({100 * match / total:.0f}%)")
    print(f"  Memory:      {orig_mb:.1f} MB -> {comp_mb:.1f} MB ({ratio:.1f}x)")
    print()

    # ── Configuration options ─────────────────────────────────────────────
    print("=" * 60)
    print("  CONFIGURATION OPTIONS")
    print("=" * 60)
    print("  # Basic (auto-detect everything):")
    print("  model = turboquant.wrap(model)")
    print()
    print("  # With options:")
    print("  model = turboquant.wrap(")
    print("      model,")
    print("      bit_width=3,           # 2, 3, or 4")
    print("      residual_length=128,   # sliding window")
    print("      n_outlier_channels=8,  # outlier routing")
    print("      verbose=True,          # print stats")
    print("  )")


if __name__ == "__main__":
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        print('Install dependencies: pip install "turboquant-torch[hf]"')
        sys.exit(1)
    main()
