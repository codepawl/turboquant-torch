# Quick Start

## Installation
```bash
pip install turboquant-torch

# With HuggingFace support
pip install "turboquant-torch[hf]"
```

## One-Line Compression
```python
import turboquant
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

model = turboquant.wrap(model)
output = model.generate(**tokenizer("Hello", return_tensors="pt"), max_new_tokens=50)
```

## Configuration Options
```python
model = turboquant.wrap(
    model,
    bit_width=3,              # 2, 3, or 4 (None = auto)
    residual_length=128,      # sliding window
    n_outlier_channels=8,     # outlier routing
    verbose=True,             # print compression stats
)
```

## Using the Cache Directly
```python
from turboquant import TurboQuantDynamicCache

cache = TurboQuantDynamicCache.from_model(model)
output = model.generate(**inputs, past_key_values=cache, max_new_tokens=50)
```

## Low-Level API
```python
import torch
from turboquant import TurboQuant

tq = TurboQuant(dim=128, bit_width=3, unbiased=True)
x = torch.randn(100, 128)
output = tq.quantize(x)
x_hat = tq.dequantize(output)
```
