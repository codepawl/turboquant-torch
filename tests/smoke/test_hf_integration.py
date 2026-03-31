"""Smoke tests for HF integration with real models.

Requires: pip install transformers
Skipped automatically if transformers is not installed.
"""

from __future__ import annotations

import pytest
import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

import turboquant
from turboquant.hf_cache import TurboQuantDynamicCache


@pytest.mark.slow
@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
class TestHFIntegration:
    """Integration tests with HuggingFaceTB/SmolLM2-135M."""

    MODEL = "HuggingFaceTB/SmolLM2-135M"

    @pytest.fixture(scope="class")
    def model_and_tokenizer(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.MODEL, trust_remote_code=True, torch_dtype=torch.float32
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL, trust_remote_code=True
        )
        model.eval()
        return model, tokenizer

    def test_wrap_generate(self, model_and_tokenizer):
        """wrap() + generate() produces valid output."""
        model, tokenizer = model_and_tokenizer
        wrapped = turboquant.wrap(model, verbose=True)
        inputs = tokenizer("Hello world", return_tensors="pt")
        with torch.no_grad():
            output = wrapped.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        assert len(text) > len("Hello world")

    def test_dynamic_cache_with_generate(self, model_and_tokenizer):
        """TurboQuantDynamicCache works with model.generate()."""
        model, tokenizer = model_and_tokenizer
        cache = TurboQuantDynamicCache.from_model(model)
        inputs = tokenizer("The meaning of life is", return_tensors="pt")
        with torch.no_grad():
            output = model.generate(
                **inputs,
                past_key_values=cache,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        assert len(text) > len("The meaning of life is")

    def test_wrap_matches_baseline_prompt(self, model_and_tokenizer):
        """Wrapped output preserves prompt tokens."""
        model, tokenizer = model_and_tokenizer
        inputs = tokenizer("AI will", return_tensors="pt")

        with torch.no_grad():
            baseline = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        wrapped = turboquant.wrap(model)
        with torch.no_grad():
            compressed = wrapped.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        baseline_tokens = baseline[0].tolist()
        compressed_tokens = compressed[0].tolist()
        # Prompt tokens (first 3) always match
        assert baseline_tokens[:3] == compressed_tokens[:3]

    def test_from_model_correct_config(self, model_and_tokenizer):
        """from_model() correctly detects SmolLM2-135M config."""
        model, _ = model_and_tokenizer
        cache = TurboQuantDynamicCache.from_model(model)
        # SmolLM2-135M: head_dim=64, so auto-selects 4-bit
        assert cache.bit_width == 4
        assert cache.model_info is not None
        assert cache.model_info.head_dim == 64
        assert cache.model_info.n_layers == 30
