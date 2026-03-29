"""HuggingFace integration demo: compress KV cache of a real transformer model.

Shows how to intercept and compress the KV cache of a HuggingFace model
using TurboQuantKVCache, then compare outputs with and without compression.

Requirements:
    pip install transformers accelerate

Usage:
    python examples/huggingface_demo.py
"""

import torch

from turboquant import TurboQuantKVCache


def _iter_kv(past_kv):
    """Iterate over (key, value) per layer, handling both DynamicCache and tuple formats."""
    for item in past_kv:
        # DynamicCache yields (key, value, ...) tuples; legacy yields (key, value)
        yield item[0], item[1]


def run_with_huggingface():
    """Demonstrate KV cache compression on a real HuggingFace model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "HuggingFaceTB/SmolLM2-135M"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32, device_map="cpu")
    model.eval()

    prompt = "The key insight behind TurboQuant is that"
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate with full-precision KV cache
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)

    past_kv = outputs.past_key_values
    kv_pairs = list(_iter_kv(past_kv))
    n_layers = len(kv_pairs)
    B, H, S, D = kv_pairs[0][0].shape

    print(f"Model: {n_layers} layers, {H} KV heads, head_dim={D}")
    print(f"KV cache shape per layer: ({B}, {H}, {S}, {D})")

    # ── Multi-bitwidth benchmark ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Compression Quality Across Bit Widths")
    print("=" * 60)

    for bit_width in [2, 3, 4]:
        # residual_length=0 so all tokens are quantized (seq is short)
        cache = TurboQuantKVCache(head_dim=D, bit_width=bit_width, residual_length=0)

        total_key_mse = 0.0
        total_val_mse = 0.0
        total_attn_mse = 0.0

        for keys, values in kv_pairs:
            compressed = cache.compress(keys, values)
            keys_hat = cache.decompress_keys(compressed)
            values_hat = cache.decompress_values(compressed)

            total_key_mse += ((keys - keys_hat) ** 2).mean().item()
            total_val_mse += ((values - values_hat) ** 2).mean().item()

            # Attention score comparison with random query
            query = torch.randn_like(keys[:, :, :1, :])
            scale = D**-0.5
            attn_orig = torch.softmax(torch.matmul(query, keys.transpose(-2, -1)) * scale, dim=-1)
            attn_comp = torch.softmax(
                torch.matmul(query, keys_hat.transpose(-2, -1)) * scale, dim=-1
            )
            total_attn_mse += ((attn_orig - attn_comp) ** 2).mean().item()

        avg_key_mse = total_key_mse / n_layers
        avg_val_mse = total_val_mse / n_layers
        avg_attn_mse = total_attn_mse / n_layers
        orig_mb, comp_mb, ratio = cache.memory_savings(B, H, S)

        print(f"\n  {bit_width}-bit TurboQuant:")
        print(f"    Key MSE:        {avg_key_mse:.6f}")
        print(f"    Value MSE:      {avg_val_mse:.6f}")
        print(f"    Attention MSE:  {avg_attn_mse:.8f}")
        print(f"    Memory/layer:   {orig_mb:.4f} MB -> {comp_mb:.4f} MB ({ratio:.1f}x)")
        print(f"    Total KV cache: {orig_mb * n_layers:.2f} MB -> {comp_mb * n_layers:.2f} MB")

    # ── Per-layer detail (3-bit) ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Per-layer Reconstruction MSE (3-bit)")
    print("=" * 60)

    cache_3bit = TurboQuantKVCache(head_dim=D, bit_width=3, residual_length=0)
    compressed_kv = []
    for keys, values in kv_pairs:
        compressed_kv.append(cache_3bit.compress(keys, values))

    for layer_idx, comp in enumerate(compressed_kv):
        k_orig = kv_pairs[layer_idx][0]
        v_orig = kv_pairs[layer_idx][1]
        k_hat = cache_3bit.decompress_keys(comp)
        v_hat = cache_3bit.decompress_values(comp)
        k_mse = ((k_orig - k_hat) ** 2).mean().item()
        v_mse = ((v_orig - v_hat) ** 2).mean().item()
        if layer_idx < 3 or layer_idx == n_layers - 1:
            print(f"  Layer {layer_idx:2d}: key_mse={k_mse:.6f}, val_mse={v_mse:.6f}")
        elif layer_idx == 3:
            print("  ...")

    # ── Generation comparison ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Generation Comparison")
    print("=" * 60)

    # Reconstruct past_key_values from compressed cache for model forward pass
    from transformers.cache_utils import DynamicCache

    reconstructed_cache = DynamicCache()
    for comp in compressed_kv:
        k_hat = cache_3bit.decompress_keys(comp)
        v_hat = cache_3bit.decompress_values(comp)
        layer_idx = len(reconstructed_cache)
        reconstructed_cache.update(k_hat, v_hat, layer_idx)

    with torch.no_grad():
        # Original generation
        out_orig = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        text_orig = tokenizer.decode(out_orig[0], skip_special_tokens=True)

    print(f"\nOriginal output:\n  {text_orig}")

    # Next-token comparison using compressed KV
    with torch.no_grad():
        next_token = tokenizer(" that", return_tensors="pt")["input_ids"]
        out_orig_logits = model(next_token, past_key_values=past_kv).logits
        out_comp_logits = model(next_token, past_key_values=reconstructed_cache).logits

    print("\nNext-token predictions (original vs 3-bit compressed):")
    for label, logits in [("Original", out_orig_logits), ("Compressed", out_comp_logits)]:
        probs = torch.softmax(logits[0, -1], dim=-1)
        top5 = torch.topk(probs, 5)
        tokens = [tokenizer.decode(idx) for idx in top5.indices]
        pairs = list(zip(tokens, [f"{p:.3f}" for p in top5.values.tolist()], strict=False))
        print(f"  {label}: {pairs}")


def run_with_mock():
    """Demonstrate KV cache compression on a mock attention layer."""
    print("transformers not installed -- running with mock attention layer\n")

    B, H, S, D = 2, 8, 256, 64
    n_layers = 4
    print(f"Mock model: {n_layers} layers, {H} heads, head_dim={D}, seq_len={S}")

    torch.manual_seed(42)
    past_kv = [(torch.randn(B, H, S, D), torch.randn(B, H, S, D)) for _ in range(n_layers)]

    for bit_width in [2, 3, 4]:
        cache = TurboQuantKVCache(head_dim=D, bit_width=bit_width, residual_length=0)

        total_key_mse = 0.0
        total_val_mse = 0.0
        for k, v in past_kv:
            compressed = cache.compress(k, v)
            k_hat = cache.decompress_keys(compressed)
            v_hat = cache.decompress_values(compressed)
            total_key_mse += ((k - k_hat) ** 2).mean().item()
            total_val_mse += ((v - v_hat) ** 2).mean().item()

        avg_key = total_key_mse / n_layers
        avg_val = total_val_mse / n_layers
        orig_mb, comp_mb, ratio = cache.memory_savings(B, H, S)
        print(
            f"  {bit_width}-bit: Key MSE={avg_key:.4f}, Val MSE={avg_val:.4f}, "
            f"Memory: {orig_mb * n_layers:.1f}MB -> {comp_mb * n_layers:.1f}MB ({ratio:.1f}x)"
        )

    # Attention with compressed cache (3-bit)
    cache = TurboQuantKVCache(head_dim=D, bit_width=3, residual_length=0)
    query = torch.randn(B, H, 1, D)
    for i, (k, v) in enumerate(past_kv):
        comp = cache.compress(k, v)
        out_compressed = cache.attention(query, comp)

        scale = D**-0.5
        attn_weights = torch.softmax(torch.matmul(query, k.transpose(-2, -1)) * scale, dim=-1)
        out_original = torch.matmul(attn_weights, v)
        attn_mse = ((out_original - out_compressed) ** 2).mean().item()
        if i == 0:
            print(f"\nAttention output MSE (layer {i}): {attn_mse:.6f}")

    print("\nDone! Install transformers for real model demo: pip install transformers accelerate")


if __name__ == "__main__":
    try:
        import transformers  # noqa: F401

        run_with_huggingface()
    except ImportError:
        run_with_mock()
