"""KV cache compression demo with sliding window and GQA support."""

import torch

from turboquant import TurboQuantKVCache

batch_size = 2
num_heads = 32
seq_len = 2048
head_dim = 128

print(f"Config: batch={batch_size}, heads={num_heads}, seq={seq_len}, dim={head_dim}")
print()

# --- Standard usage with sliding window ---
print("=== Standard (3-bit, residual_length=128) ===")
cache = TurboQuantKVCache(head_dim=head_dim, bit_width=3, residual_length=128)

keys = torch.randn(batch_size, num_heads, seq_len, head_dim)
values = torch.randn(batch_size, num_heads, seq_len, head_dim)

compressed = cache.compress(keys, values)
print(
    f"Split point: {compressed.split_point} (quantized) + "
    f"{compressed.residual_keys.shape[2]} (residual fp32)"
)

# Memory savings
orig_mb, comp_mb, ratio = cache.memory_savings(batch_size, num_heads, seq_len)
print(f"Original:   {orig_mb:.1f} MB")
print(f"Compressed: {comp_mb:.1f} MB")
print(f"Ratio:      {ratio:.1f}x")

query = torch.randn(batch_size, num_heads, 1, head_dim)
attn_output = cache.attention(query, compressed)
print(f"Attention output shape: {attn_output.shape}")

keys_hat = cache.decompress_keys(compressed)
values_hat = cache.decompress_values(compressed)
key_mse = ((keys - keys_hat) ** 2).mean()
val_mse = ((values - values_hat) ** 2).mean()
print(f"Key reconstruction MSE:   {key_mse:.6f}")
print(f"Value reconstruction MSE: {val_mse:.6f}")
print()

# --- GQA-aware usage (e.g., Llama-3-8B: 8 KV heads, 32 query heads) ---
print("=== GQA-aware (Llama-3-8B config) ===")
cache_gqa = TurboQuantKVCache.for_gqa(
    head_dim=128,
    num_kv_heads=8,
    num_query_heads=32,
    bit_width=3,
    residual_length=128,
)
print(f"Key bits:   {cache_gqa.key_quantizer.bit_width} (auto-bumped for GQA ratio=4)")
print(f"Value bits: {cache_gqa.value_quantizer.bit_width}")

gqa_keys = torch.randn(batch_size, 8, seq_len, head_dim)
gqa_values = torch.randn(batch_size, 8, seq_len, head_dim)
gqa_compressed = cache_gqa.compress(gqa_keys, gqa_values)

gqa_query = torch.randn(batch_size, 8, 1, head_dim)
gqa_output = cache_gqa.attention(gqa_query, gqa_compressed)
print(f"GQA attention output shape: {gqa_output.shape}")

orig_mb, comp_mb, ratio = cache_gqa.memory_savings(batch_size, 8, seq_len)
print(f"Original:   {orig_mb:.1f} MB")
print(f"Compressed: {comp_mb:.1f} MB")
print(f"Ratio:      {ratio:.1f}x")
