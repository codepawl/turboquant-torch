"""Smoke tests with real HuggingFace models. Slow, needs downloads."""

import pytest
import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from turboquant import TurboQuantKVCache


@pytest.mark.slow
@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
class TestHuggingFace:
    def test_smollm_kv_compression(self):
        """Compress real KV cache from SmolLM2-135M."""
        model = AutoModelForCausalLM.from_pretrained(
            "HuggingFaceTB/SmolLM2-135M",
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "HuggingFaceTB/SmolLM2-135M",
            trust_remote_code=True,
        )

        inputs = tokenizer("Hello world", return_tensors="pt")
        with torch.no_grad():
            out = model(**inputs, use_cache=True)

        past_kv = out.past_key_values
        k = past_kv.key_cache[0] if hasattr(past_kv, "key_cache") else past_kv[0][0]

        head_dim = k.shape[-1]
        cache = TurboQuantKVCache(head_dim=head_dim, bit_width=3, residual_length=0)

        # Should not crash
        if hasattr(past_kv, "key_cache"):
            compressed = cache.compress(
                past_kv.key_cache[0].float(),
                past_kv.value_cache[0].float(),
            )
        else:
            compressed = cache.compress(
                past_kv[0][0].float(),
                past_kv[0][1].float(),
            )

        k_hat = cache.decompress_keys(compressed)
        assert k_hat.shape[-1] == head_dim
