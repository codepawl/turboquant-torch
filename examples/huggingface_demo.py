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

# ---------------------------------------------------------------------------
# Option A: Use a real HuggingFace model (requires `transformers` installed)
# Option B: Fall back to a mock transformer layer if unavailable
# ---------------------------------------------------------------------------


def run_with_huggingface():
    """Demonstrate KV cache compression on a real HuggingFace model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen2-0.5B"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.eval()

    prompt = "The key insight behind TurboQuant is that"
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate with full-precision KV cache
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
        past_kv = outputs.past_key_values
        _ = outputs.logits

    # Extract KV cache dimensions
    # past_kv is a tuple of (key, value) per layer, each (B, H, S, D)
    n_layers = len(past_kv)
    B, H, S, D = past_kv[0][0].shape
    print(f"Model: {n_layers} layers, {H} heads, head_dim={D}")
    print(f"KV cache shape per layer: ({B}, {H}, {S}, {D})")

    # Compress KV cache with TurboQuant (3-bit)
    bit_width = 3
    cache = TurboQuantKVCache(head_dim=D, bit_width=bit_width)

    compressed_kv = []
    for k, v in past_kv:
        compressed_kv.append(cache.compress(k, v))

    # Report memory savings
    orig_mb, comp_mb, ratio = cache.memory_savings(B, H, S)
    total_orig = orig_mb * n_layers
    total_comp = comp_mb * n_layers
    print(f"\nMemory per layer: {orig_mb:.2f} MB -> {comp_mb:.2f} MB ({ratio:.1f}x)")
    print(f"Total KV cache:  {total_orig:.2f} MB -> {total_comp:.2f} MB ({ratio:.1f}x)")

    # Decompress and compare
    print("\nPer-layer reconstruction MSE:")
    for layer_idx, comp in enumerate(compressed_kv):
        k_orig = past_kv[layer_idx][0]
        v_orig = past_kv[layer_idx][1]
        k_hat = cache.decompress_keys(comp)
        v_hat = cache.decompress_values(comp)
        k_mse = ((k_orig - k_hat) ** 2).mean().item()
        v_mse = ((v_orig - v_hat) ** 2).mean().item()
        if layer_idx < 3 or layer_idx == n_layers - 1:
            print(f"  Layer {layer_idx:2d}: key_mse={k_mse:.6f}, val_mse={v_mse:.6f}")
        elif layer_idx == 3:
            print("  ...")

    # Compare next-token logits using compressed KV
    # Reconstruct past_key_values from compressed cache
    reconstructed_kv = tuple(
        (cache.decompress_keys(comp), cache.decompress_values(comp)) for comp in compressed_kv
    )

    with torch.no_grad():
        next_token = tokenizer(" that", return_tensors="pt")["input_ids"]
        out_orig = model(next_token, past_key_values=past_kv)
        out_comp = model(next_token, past_key_values=reconstructed_kv)

    # Compare top-5 predictions
    print("\nNext-token predictions (original vs compressed):")
    for label, logits in [("Original", out_orig.logits), ("Compressed", out_comp.logits)]:
        probs = torch.softmax(logits[0, -1], dim=-1)
        top5 = torch.topk(probs, 5)
        tokens = [tokenizer.decode(idx) for idx in top5.indices]
        print(
            f"  {label}: {list(zip(tokens, [f'{p:.3f}' for p in top5.values.tolist()], strict=False))}"
        )


def run_with_mock():
    """Demonstrate KV cache compression on a mock attention layer."""
    print("transformers not installed -- running with mock attention layer\n")

    # Simulate a transformer's KV cache
    B, H, S, D = 2, 8, 256, 64
    n_layers = 4
    print(f"Mock model: {n_layers} layers, {H} heads, head_dim={D}, seq_len={S}")

    # Generate random KV cache
    torch.manual_seed(42)
    past_kv = [(torch.randn(B, H, S, D), torch.randn(B, H, S, D)) for _ in range(n_layers)]

    # Compress with TurboQuant
    bit_width = 3
    cache = TurboQuantKVCache(head_dim=D, bit_width=bit_width)

    compressed_kv = []
    for k, v in past_kv:
        compressed_kv.append(cache.compress(k, v))

    # Memory savings
    orig_mb, comp_mb, ratio = cache.memory_savings(B, H, S)
    print(f"\nMemory per layer: {orig_mb:.2f} MB -> {comp_mb:.2f} MB ({ratio:.1f}x)")
    print(f"Total KV cache:  {orig_mb * n_layers:.2f} MB -> {comp_mb * n_layers:.2f} MB")

    # Reconstruction quality
    print("\nReconstruction MSE per layer:")
    for i, (comp, (k, v)) in enumerate(zip(compressed_kv, past_kv, strict=True)):
        k_hat = cache.decompress_keys(comp)
        v_hat = cache.decompress_values(comp)
        k_mse = ((k - k_hat) ** 2).mean().item()
        v_mse = ((v - v_hat) ** 2).mean().item()
        print(f"  Layer {i}: key_mse={k_mse:.6f}, val_mse={v_mse:.6f}")

    # Attention with compressed cache
    query = torch.randn(B, H, 1, D)
    for i, comp in enumerate(compressed_kv):
        out_compressed = cache.attention(query, comp)

        # Compare with uncompressed attention
        k, v = past_kv[i]
        scale = D**-0.5
        attn_weights = torch.softmax(torch.matmul(query, k.transpose(-2, -1)) * scale, dim=-1)
        out_original = torch.matmul(attn_weights, v)

        attn_mse = ((out_original - out_compressed) ** 2).mean().item()
        if i == 0:
            print(f"\nAttention output MSE (layer {i}): {attn_mse:.6f}")

    print("\nDone! TurboQuant KV cache compression works with any transformer model.")
    print("To use with a real model, install: pip install transformers accelerate")


if __name__ == "__main__":
    try:
        import transformers  # noqa: F401

        run_with_huggingface()
    except ImportError:
        run_with_mock()
