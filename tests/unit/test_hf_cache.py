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

    def test_multiple_layers(self):
        """Updates to different layers are stored independently."""
        cache = TurboQuantDynamicCache()
        k0 = torch.randn(1, 4, 8, 64)
        v0 = torch.randn(1, 4, 8, 64)
        k1 = torch.randn(1, 4, 4, 64)
        v1 = torch.randn(1, 4, 4, 64)

        cache.update(k0, v0, layer_idx=0)
        cache.update(k1, v1, layer_idx=1)

        assert len(cache) == 2
        assert cache.get_seq_length(0) == 8
        assert cache.get_seq_length(1) == 4

    def test_key_cache_property(self):
        """key_cache property returns list of tensors."""
        cache = TurboQuantDynamicCache()
        k = torch.randn(1, 4, 8, 64)
        v = torch.randn(1, 4, 8, 64)
        cache.update(k, v, layer_idx=0)

        keys = cache.key_cache
        assert len(keys) == 1
        assert torch.equal(keys[0], k)

    def test_compress_all(self):
        """compress_all compresses stored layers and returns stats."""
        cache = TurboQuantDynamicCache(bit_width=3)
        torch.manual_seed(42)
        k = torch.randn(1, 4, 32, 128)
        v = torch.randn(1, 4, 32, 128)
        cache.update(k, v, layer_idx=0)

        stats = cache.compress_all()

        assert stats["layers_compressed"] == 1
        assert stats["layers_skipped"] == 0
        assert stats["original_mb"] > 0
        assert stats["compressed_mb"] > 0
        assert stats["ratio"] > 1.0

        # Values should have changed (lossy compression)
        k_after = cache.key_cache[0]
        assert k_after.shape == k.shape
        assert not torch.equal(k_after, k)

    def test_skip_layers(self):
        """Layers in skip_layers are not compressed."""
        info = ModelKVInfo(
            n_layers=4,
            head_dim=64,
            num_kv_heads=4,
            num_query_heads=4,
            attention_layers=[0, 2],
            skip_layers=[1, 3],
        )
        cache = TurboQuantDynamicCache(model_info=info)

        for i in range(4):
            k = torch.randn(1, 4, 16, 64)
            v = torch.randn(1, 4, 16, 64)
            cache.update(k, v, layer_idx=i)

        # Save originals for skip layers
        k1_orig = cache._keys[1].clone()
        k3_orig = cache._keys[3].clone()

        stats = cache.compress_all()

        assert stats["layers_compressed"] == 2
        assert stats["layers_skipped"] == 2
        # Skip layers unchanged
        assert torch.equal(cache._keys[1], k1_orig)
        assert torch.equal(cache._keys[3], k3_orig)

    def test_to_legacy_cache(self):
        """to_legacy_cache returns tuple-of-tuples format."""
        cache = TurboQuantDynamicCache()
        k = torch.randn(1, 4, 8, 64)
        v = torch.randn(1, 4, 8, 64)
        cache.update(k, v, layer_idx=0)

        legacy = cache.to_legacy_cache()
        assert isinstance(legacy, tuple)
        assert len(legacy) == 1
        assert torch.equal(legacy[0][0], k)
        assert torch.equal(legacy[0][1], v)

    def test_crop(self):
        """crop truncates all layers to max_length."""
        cache = TurboQuantDynamicCache()
        k = torch.randn(1, 4, 32, 64)
        v = torch.randn(1, 4, 32, 64)
        cache.update(k, v, layer_idx=0)

        cache.crop(16)
        assert cache.get_seq_length(0) == 16

    def test_iter(self):
        """__iter__ yields (key, value) pairs."""
        cache = TurboQuantDynamicCache()
        cache.update(torch.randn(1, 2, 4, 64), torch.randn(1, 2, 4, 64), layer_idx=0)
        cache.update(torch.randn(1, 2, 4, 64), torch.randn(1, 2, 4, 64), layer_idx=1)

        pairs = list(cache)
        assert len(pairs) == 2
        assert len(pairs[0]) == 2  # (key, value)

    def test_getitem(self):
        """__getitem__ returns (key, value) for layer index."""
        cache = TurboQuantDynamicCache()
        k = torch.randn(1, 2, 4, 64)
        v = torch.randn(1, 2, 4, 64)
        cache.update(k, v, layer_idx=0)

        key, val = cache[0]
        assert torch.equal(key, k)
        assert torch.equal(val, v)

    def test_get_max_cache_shape(self):
        """get_max_cache_shape returns None (unbounded)."""
        cache = TurboQuantDynamicCache()
        assert cache.get_max_cache_shape() is None

    def test_seq_length_empty(self):
        """get_seq_length returns 0 for empty cache."""
        cache = TurboQuantDynamicCache()
        assert cache.get_seq_length(0) == 0
        assert cache.get_seq_length(99) == 0


def _make_mock_config(head_dim=128, n_layers=12, n_heads=32, n_kv_heads=8):
    """Build a mock HF config."""
    return SimpleNamespace(
        hidden_size=head_dim * n_heads,
        num_attention_heads=n_heads,
        num_hidden_layers=n_layers,
        num_key_value_heads=n_kv_heads,
        head_dim=head_dim,
    )


def _make_mock_model(head_dim=128, n_layers=12, n_heads=32, n_kv_heads=8):
    """Build a mock HF model."""
    return SimpleNamespace(config=_make_mock_config(head_dim, n_layers, n_heads, n_kv_heads))


class TestFromModelAutoConfig:
    """Tests for TurboQuantDynamicCache.from_model auto-configuration."""

    def test_auto_bit_width_small_dim(self):
        """head_dim=64 auto-selects 4-bit."""
        model = _make_mock_model(head_dim=64)
        cache = TurboQuantDynamicCache.from_model(model)
        assert cache.bit_width == 4

    def test_auto_bit_width_large_dim(self):
        """head_dim=128 auto-selects 3-bit."""
        model = _make_mock_model(head_dim=128)
        cache = TurboQuantDynamicCache.from_model(model)
        assert cache.bit_width == 3

    def test_explicit_bit_width_override(self):
        """Explicit bit_width overrides auto-selection."""
        model = _make_mock_model(head_dim=128)
        cache = TurboQuantDynamicCache.from_model(model, bit_width=2)
        assert cache.bit_width == 2

    def test_skip_layers_from_model_info(self):
        """from_model detects skip layers via model info."""
        config = SimpleNamespace(
            hidden_size=2048,
            num_attention_heads=32,
            num_hidden_layers=4,
            num_key_value_heads=8,
            head_dim=64,
            hybrid_attention_layers=[0, 2],
        )
        model = SimpleNamespace(config=config)
        cache = TurboQuantDynamicCache.from_model(model)
        assert 1 in cache._skip_layers
        assert 3 in cache._skip_layers
