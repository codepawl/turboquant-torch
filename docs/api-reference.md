# API Reference

## High-Level API

### `turboquant.wrap(model, **kwargs)`

Wrap a HuggingFace model for automatic KV cache compression.

**Parameters:**
- `model` — Any HuggingFace CausalLM model
- `bit_width` (int, optional) — Bits per coordinate (2/3/4). None = auto-detect based on head_dim
- `residual_length` (int) — Keep last N tokens in fp16. Default 0
- `n_outlier_channels` (int) — Top-k channels kept in fp16. Default 0
- `verbose` (bool) — Print compression stats after generation. Default False

**Returns:** `TurboQuantWrapper`

### `TurboQuantDynamicCache`

Drop-in replacement for HuggingFace DynamicCache.

**Constructor:**
- `bit_width` (int) — Default 3
- `residual_length` (int) — Default 0
- `n_outlier_channels` (int) — Default 0
- `model_info` (ModelKVInfo, optional) — Pre-configured model info

**Class Methods:**
- `from_model(model, **kwargs)` — Auto-configure from HF model

**Methods:**
- `update(key_states, value_states, layer_idx)` — Store KV pair
- `compress_all()` — Compress accumulated KV, returns stats dict
- `get_seq_length(layer_idx)` — Current sequence length
- `crop(max_length)` — Truncate cache
- `to_legacy_cache()` — Convert to tuple format

**Properties:**
- `key_cache` — List of key tensors
- `value_cache` — List of value tensors

## Core API

### `TurboQuant(dim, bit_width, unbiased, seed)`

Two-stage online vector quantizer.

**Parameters:**
- `dim` (int) — Vector dimension
- `bit_width` (int) — Bits per coordinate (2/3/4)
- `unbiased` (bool) — Use QJL for unbiased inner products
- `seed` (int) — Random seed. Default 0

**Methods:**
- `quantize(x)` — Quantize tensor, returns TurboQuantOutput
- `dequantize(output)` — Reconstruct from quantized output
- `compression_ratio()` — Theoretical compression ratio
- `compute_inner_product(query, output)` — Inner product from compressed
- `to(device)` — Move to device

### `TurboQuantKVCache(head_dim, bit_width, **kwargs)`

KV cache compression for transformer models.

**Parameters:**
- `head_dim` (int) — Attention head dimension
- `bit_width` (int) — Default 3
- `residual_length` (int) — Sliding window size. Default 128
- `key_bit_width` (int, optional) — Override for keys
- `value_bit_width` (int, optional) — Override for values
- `pre_rope` (bool) — Quantize before RoPE. Default False
- `n_outlier_channels` (int) — Outlier channels in fp16. Default 0
- `seed` (int) — Default 0

**Methods:**
- `compress(keys, values)` — Compress KV tensors
- `decompress_keys(compressed)` — Decompress keys
- `decompress_values(compressed)` — Decompress values
- `attention(query, compressed)` — Attention with compressed KV
- `memory_savings(batch, heads, seq_len)` — Memory stats

**Class Methods:**
- `for_gqa(head_dim, num_kv_heads, num_query_heads, **kwargs)` — GQA-aware factory

### `AdaptiveKVCache(head_dim, layer_bits, **kwargs)`

Per-layer bit allocation.

**Parameters:**
- `head_dim` (int) — Attention head dimension
- `layer_bits` (List[int]) — Bit width per layer
- `residual_length` (int) — Default 128
- `n_outlier_channels` (int) — Default 0

**Methods:**
- `compress_layer(layer_idx, keys, values)` — Compress single layer
- `decompress_layer_keys(layer_idx, compressed)` — Decompress layer keys
- `decompress_layer_values(layer_idx, compressed)` — Decompress layer values
- `attention_layer(layer_idx, query, compressed)` — Per-layer attention
- `summary()` — Print allocation summary

**Class Methods:**
- `from_model(model, tokenizer, **kwargs)` — Auto-calibrate from model

## Utilities

### `detect_model_kv_info(model)`

Auto-detect KV cache structure from HuggingFace model.

**Returns:** `ModelKVInfo` with fields: n_layers, head_dim, num_kv_heads, num_query_heads, attention_layers, skip_layers, sliding_window, is_latent_kv

### `compress_model_kv(past_key_values, model, **kwargs)`

Compress a model's full KV cache with auto architecture handling.

### Allocation Strategies

- `uniform_allocation(n_layers, bit_width)` — Same bits everywhere
- `gradient_allocation(n_layers, min_bits, max_bits, strategy)` — Linear or step gradient
- `calibration_allocation(model, tokenizer, calibration_text, bit_options, target_avg_bits)` — Data-driven allocation

### Outlier Detection

- `detect_outlier_channels(x, n_outliers, method)` — Find outlier channels
- `split_outliers(x, indices)` — Extract outliers to separate tensor
- `merge_outliers(bulk, split)` — Recombine after quantization
