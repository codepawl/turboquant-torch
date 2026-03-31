"""Tests for TurboQuantWrapper and wrap() API."""

import torch
from torch import nn

from turboquant.hf_cache import TurboQuantDynamicCache
from turboquant.wrap import TurboQuantWrapper, wrap

# -- Mock helpers ----------------------------------------------------------


class MockConfig:
    """Minimal HF config for testing."""

    def __init__(self, head_dim=128, n_layers=12, n_heads=32, n_kv_heads=8):
        self.hidden_size = head_dim * n_heads
        self.num_attention_heads = n_heads
        self.num_hidden_layers = n_layers
        self.num_key_value_heads = n_kv_heads
        self.head_dim = head_dim
        self.custom_attr = "test_value"


class MockModel(nn.Module):
    """Minimal HF model for testing."""

    def __init__(self, head_dim=128, n_layers=12, n_heads=32, n_kv_heads=8):
        super().__init__()
        self.config = MockConfig(head_dim, n_layers, n_heads, n_kv_heads)
        self.linear = nn.Linear(10, 10)  # so parameters() works

    def generate(self, *args, **kwargs):
        return torch.tensor([1, 2, 3])

    def forward(self, *args, **kwargs):
        return torch.tensor([4, 5, 6])


class SpyModel(MockModel):
    """MockModel that records generate() kwargs."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_generate_kwargs = {}

    def generate(self, *args, **kwargs):
        self.last_generate_kwargs = dict(kwargs)
        return torch.tensor([1, 2, 3])


# -- Tests -----------------------------------------------------------------


class TestTurboQuantWrapper:
    """Tests for TurboQuantWrapper and wrap()."""

    def test_wrap_returns_wrapper(self):
        """wrap() returns a TurboQuantWrapper."""
        model = MockModel()
        wrapped = wrap(model)
        assert isinstance(wrapped, TurboQuantWrapper)

    def test_wrap_auto_bit_width(self):
        """Auto-selects 3-bit for head_dim=128."""
        model = MockModel(head_dim=128)
        wrapped = wrap(model)
        assert wrapped._bit_width == 3

    def test_wrap_custom_bit_width(self):
        """Explicit bit_width is respected."""
        model = MockModel(head_dim=128)
        wrapped = wrap(model, bit_width=2)
        assert wrapped._bit_width == 2

    def test_wrap_forwards_config(self):
        """config property returns model config."""
        model = MockModel()
        wrapped = wrap(model)
        assert wrapped.config is model.config

    def test_wrap_generate(self):
        """generate() returns model output."""
        model = MockModel()
        wrapped = wrap(model)
        result = wrapped.generate(torch.tensor([[1, 2]]))
        assert torch.equal(result, torch.tensor([1, 2, 3]))

    def test_wrap_generate_injects_cache(self):
        """generate() injects TurboQuantDynamicCache when none provided."""
        spy = SpyModel()
        wrapped = wrap(spy)
        wrapped.generate(torch.tensor([[1, 2]]))
        assert isinstance(
            spy.last_generate_kwargs.get("past_key_values"),
            TurboQuantDynamicCache,
        )

    def test_wrap_generate_preserves_existing_cache(self):
        """generate() does not replace an existing past_key_values."""
        spy = SpyModel()
        wrapped = wrap(spy)
        sentinel = object()
        wrapped.generate(torch.tensor([[1, 2]]), past_key_values=sentinel)
        assert spy.last_generate_kwargs["past_key_values"] is sentinel

    def test_wrap_repr(self):
        """__repr__ includes bit_width and model class."""
        model = MockModel()
        wrapped = wrap(model)
        r = repr(wrapped)
        assert "TurboQuantWrapper" in r
        assert "bit_width=3" in r
        assert "MockModel" in r

    def test_wrap_eval_returns_self(self):
        """Calling the eval method returns self for chaining."""
        model = MockModel()
        wrapped = wrap(model)
        result = wrapped.eval()  # noqa: B009
        assert result is wrapped

    def test_wrap_verbose_flag(self):
        """verbose=True is stored on the wrapper."""
        model = MockModel()
        wrapped = wrap(model, verbose=True)
        assert wrapped._verbose is True

    def test_wrap_getattr_forwards(self):
        """Non-underscore attributes are forwarded to the model."""
        model = MockModel()
        model.custom_attr = "test_value"  # type: ignore[attr-defined]
        wrapped = wrap(model)
        assert wrapped.custom_attr == "test_value"
