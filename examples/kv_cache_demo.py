"""KV cache compression demo with memory savings report."""

import torch

from turboquant import TurboQuantKVCache

# Typical transformer config
batch_size = 2
num_heads = 32
seq_len = 2048
head_dim = 128

print(f"Config: batch={batch_size}, heads={num_heads}, seq={seq_len}, dim={head_dim}")

# Create KV cache compressor (3-bit)
cache = TurboQuantKVCache(head_dim=head_dim, bit_width=3)

# Simulate KV tensors
keys = torch.randn(batch_size, num_heads, seq_len, head_dim)
values = torch.randn(batch_size, num_heads, seq_len, head_dim)

# Compress
compressed = cache.compress(keys, values)
print("Compressed!")

# Memory savings
orig_mb, comp_mb, ratio = cache.memory_savings(batch_size, num_heads, seq_len)
print(f"Original:   {orig_mb:.1f} MB")
print(f"Compressed: {comp_mb:.1f} MB")
print(f"Ratio:      {ratio:.1f}x")

# Attention with compressed cache
query = torch.randn(batch_size, num_heads, 1, head_dim)
attn_output = cache.attention(query, compressed)
print(f"Attention output shape: {attn_output.shape}")

# Compare with uncompressed attention
keys_hat = cache.decompress_keys(compressed)
values_hat = cache.decompress_values(compressed)
key_mse = ((keys - keys_hat) ** 2).mean()
val_mse = ((values - values_hat) ** 2).mean()
print(f"Key reconstruction MSE:   {key_mse:.6f}")
print(f"Value reconstruction MSE: {val_mse:.6f}")
