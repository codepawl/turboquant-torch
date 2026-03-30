"""Smoke tests for compat module with real HuggingFace models."""

import pytest
import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from turboquant.compat import compress_model_kv, detect_model_kv_info, extract_kv


@pytest.mark.slow
@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
class TestCompatSmoke:
    def test_detect_smollm(self):
        """detect_model_kv_info returns valid info for SmolLM2-135M."""
        model = AutoModelForCausalLM.from_pretrained(
            "HuggingFaceTB/SmolLM2-135M",
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
        info = detect_model_kv_info(model)

        assert info.n_layers == 30
        assert info.head_dim == 64
        assert info.num_kv_heads == 3
        assert info.num_query_heads == 9
        assert info.is_latent_kv is False
        assert len(info.skip_layers) == 0
        assert len(info.attention_layers) == 30

    def test_compress_smollm(self):
        """compress_model_kv roundtrip preserves shapes on SmolLM2-135M."""
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
        original_kv = extract_kv(past_kv)

        result = compress_model_kv(past_kv, model, bit_width=3)
        result_kv = extract_kv(result)

        assert len(result_kv) == len(original_kv)
        for (k_orig, _), (k_comp, _) in zip(original_kv, result_kv, strict=True):
            assert k_comp.shape == k_orig.shape
