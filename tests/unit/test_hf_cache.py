"""Tests for TurboQuantDynamicCache HuggingFace integration."""

from types import SimpleNamespace

import torch

from turboquant.compat import ModelKVInfo
from turboquant.hf_cache import TurboQuantDynamicCache


class TestTurboQuantDynamicCache:
    """Core TurboQuantDynamicCache functionality."""

    def test_create_default(self):
        """Default constructor creates empty cache."""
        cache = TurboQuantDynamicCache()
        assert cache.bit_width == 3
        assert cache.residual_length == 0
        assert len(cache) == 0

    def test_update_stores_kv(self):
        """First update stores key/value tensors."""
        cache = TurboQuantDynamicCache()
        k = torch.randn(1, 4, 8, 64)
        v = torch.randn(1, 4, 8, 64)

        k_out, v_out = cache.update(k, v, layer_idx=0)

        assert k_out.shape == (1, 4, 8, 64)
        assert v_out.shape == (1, 4, 8, 64)
        assert torch.equal(k_out, k)
        assert torch.equal(v_out, v)

    def test_update_concatenates(self):
        """Subsequent updates concatenate along sequence dim."""
        cache = TurboQuantDynamicCache()
        k1 = torch.randn(1, 4, 8, 64)
        v1 = torch.randn(1, 4, 8, 64)
        k2 = torch.randn(1, 4, 1, 64)
        v2 = torch.randn(1, 4, 1, 64)

        cache.update(k1, v1, layer_idx=0)
        k_out, v_out = cache.update(k2, v2, layer_idx=0)

        assert k_out.shape == (1, 4, 9, 64)
        assert v_out.shape == (1, 4, 9, 64)
