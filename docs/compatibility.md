# Model Compatibility

TurboQuant compresses standard transformer KV caches.

| Model Family | Status | Notes |
|---|---|---|
| Llama-3 / 3.1 / 3.2 | Full support | GQA-aware mode recommended |
| Mistral / Mixtral | Full support | Sliding window auto-detected |
| Gemma / Gemma 2 | Full support | |
| Qwen2.5 / Qwen3 | Full support | |
| Phi-3 / Phi-4 | Full support | |
| Command-R | Full support | |
| DeepSeek-V2/V3 | Skip MLA layers | KV already compressed by MLA |
| Qwen3.5 / Jamba | Attention layers only | Non-attention layers skipped |
| T5 / BART / mBART | Partial | Self-attention KV only |
| Mamba / RWKV | Not applicable | No KV cache (SSM/RNN) |

## Auto-Detection

`turboquant.wrap()` and `TurboQuantDynamicCache.from_model()` automatically:

1. Detect head_dim, GQA ratio, layer count
2. Identify hybrid layers (skip non-attention)
3. Select optimal bit_width (3-bit for head_dim>=128, 4-bit otherwise)
4. Detect MLA (DeepSeek) and raise informative error

## Manual Configuration

For unsupported or exotic architectures:
```python
from turboquant import TurboQuantKVCache

cache = TurboQuantKVCache(
    head_dim=128,     # set manually
    bit_width=3,
    residual_length=0,
)
```
