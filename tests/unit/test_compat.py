"""Tests for model architecture compatibility detection."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from turboquant.compat import (
    ModelKVInfo,
    _get_layers,
    _has_kv_proj,
    compress_model_kv,
    detect_model_kv_info,
    extract_kv,
)

# ── Mock helpers ─────────────────────────────────────────────────────────


def _make_config(**kwargs):
    """Build a mock HF config."""
    defaults = {
        "hidden_size": 2048,
        "num_attention_heads": 32,
        "num_hidden_layers": 24,
        "num_key_value_heads": 8,
        "head_dim": 64,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _make_model(config=None, layers=None):
    """Build a mock HF model."""
    if config is None:
        config = _make_config()
    model = SimpleNamespace(config=config)
    if layers is not None:
        model.model = SimpleNamespace(layers=layers)
    return model


def _make_attn_layer():
    """Layer with self_attn.k_proj (attention layer)."""
    k_proj = nn.Linear(64, 64)
    v_proj = nn.Linear(64, 64)
    self_attn = SimpleNamespace(k_proj=k_proj, v_proj=v_proj)
    return SimpleNamespace(self_attn=self_attn)


def _make_non_attn_layer():
    """Layer without attention (e.g., Mamba/linear layer)."""
    return SimpleNamespace(mlp=nn.Linear(64, 64))


def _make_tuple_cache(n_layers=4, B=1, H=4, S=16, D=64):
    """Mock legacy tuple cache."""
    return [(torch.randn(B, H, S, D), torch.randn(B, H, S, D)) for _ in range(n_layers)]


def _make_key_cache_obj(n_layers=4, B=1, H=4, S=16, D=64, none_layers=None):
    """Mock transformers 4.x DynamicCache."""
    none_layers = none_layers or []
    keys = []
    vals = []
    for i in range(n_layers):
        if i in none_layers:
            keys.append(None)
            vals.append(None)
        else:
            keys.append(torch.randn(B, H, S, D))
            vals.append(torch.randn(B, H, S, D))
    return SimpleNamespace(key_cache=keys, value_cache=vals)


def _make_layers_cache(n_layers=4, B=1, H=4, S=16, D=64):
    """Mock transformers >=5.x DynamicCache."""
    layers = [
        SimpleNamespace(keys=torch.randn(B, H, S, D), values=torch.randn(B, H, S, D))
        for _ in range(n_layers)
    ]
    return SimpleNamespace(layers=layers)


# ── Tests ────────────────────────────────────────────────────────────────


class TestModelKVInfo:
    def test_default_fields(self):
        """Default optional fields are correct."""
        info = ModelKVInfo(n_layers=32, head_dim=128, num_kv_heads=8, num_query_heads=32)
        assert info.cache_format is None
        assert info.is_latent_kv is False
        assert info.sliding_window is None
        assert info.shared_kv_layers is None
        assert info.skip_layers == []
        assert info.attention_layers == []


class TestExtractKV:
    def test_tuple_format(self):
        """Extracts from legacy tuple cache."""
        cache = _make_tuple_cache(n_layers=3)
        kv = extract_kv(cache)
        assert len(kv) == 3
        assert kv[0][0].shape == (1, 4, 16, 64)

    def test_key_cache_format(self):
        """Extracts from .key_cache/.value_cache format."""
        cache = _make_key_cache_obj(n_layers=4)
        kv = extract_kv(cache)
        assert len(kv) == 4
        assert kv[0][0].shape == (1, 4, 16, 64)

    def test_layers_format(self):
        """Extracts from .layers[i].keys/.values format."""
        cache = _make_layers_cache(n_layers=3)
        kv = extract_kv(cache)
        assert len(kv) == 3
        assert kv[0][0].shape == (1, 4, 16, 64)

    def test_hybrid_none_entries(self):
        """None entries preserved for hybrid models."""
        cache = _make_key_cache_obj(n_layers=4, none_layers=[1, 3])
        kv = extract_kv(cache)
        assert kv[0][0] is not None
        assert kv[1][0] is None
        assert kv[1][1] is None
        assert kv[2][0] is not None
        assert kv[3][0] is None


class TestDetectModelKVInfo:
    def test_standard_config(self):
        """Detects standard Llama-like config."""
        model = _make_model()
        info = detect_model_kv_info(model)
        assert info.n_layers == 24
        assert info.head_dim == 64
        assert info.num_kv_heads == 8
        assert info.num_query_heads == 32

    def test_vlm_text_config(self):
        """Unwraps nested text_config for VLM models."""
        text_cfg = _make_config(hidden_size=4096, num_attention_heads=64, head_dim=64)
        outer_cfg = SimpleNamespace(text_config=text_cfg)
        model = SimpleNamespace(config=outer_cfg)
        info = detect_model_kv_info(model)
        assert info.num_query_heads == 64
        assert info.head_dim == 64

    def test_gqa_detection(self):
        """Detects GQA when num_kv_heads < num_attention_heads."""
        model = _make_model(_make_config(num_key_value_heads=4, num_attention_heads=32))
        info = detect_model_kv_info(model)
        assert info.num_kv_heads == 4
        assert info.num_query_heads == 32

    def test_head_dim_fallback(self):
        """Computes head_dim from hidden_size // num_heads when not explicit."""
        cfg = _make_config(hidden_size=2048, num_attention_heads=16)
        del cfg.head_dim
        model = _make_model(cfg)
        info = detect_model_kv_info(model)
        assert info.head_dim == 128

    def test_kv_heads_fallback(self):
        """Defaults num_kv_heads to num_attention_heads (MHA)."""
        cfg = _make_config(num_attention_heads=32)
        del cfg.num_key_value_heads
        model = _make_model(cfg)
        info = detect_model_kv_info(model)
        assert info.num_kv_heads == 32

    def test_sliding_window(self):
        """Detects sliding_window from config."""
        model = _make_model(_make_config(sliding_window=4096))
        info = detect_model_kv_info(model)
        assert info.sliding_window == 4096

    def test_mla_detection(self):
        """Detects MLA (DeepSeek) via kv_lora_rank."""
        model = _make_model(_make_config(kv_lora_rank=512))
        info = detect_model_kv_info(model)
        assert info.is_latent_kv is True

    def test_hybrid_layers(self):
        """Detects non-attention layers via introspection."""
        layers = [_make_attn_layer(), _make_non_attn_layer(), _make_attn_layer()]
        model = _make_model(_make_config(num_hidden_layers=3), layers=layers)
        info = detect_model_kv_info(model)
        assert 0 in info.attention_layers
        assert 2 in info.attention_layers
        assert 1 in info.skip_layers

    def test_no_config_raises(self):
        """Model without config raises ValueError."""
        model = SimpleNamespace()
        with pytest.raises(ValueError, match="no .config"):
            detect_model_kv_info(model)


class TestCompressModelKV:
    def test_basic_roundtrip(self):
        """Compress/decompress tuple cache preserves shapes."""
        pytest.importorskip("transformers")
        cache = _make_tuple_cache(n_layers=3, D=64)
        info = ModelKVInfo(
            n_layers=3,
            head_dim=64,
            num_kv_heads=4,
            num_query_heads=4,
            attention_layers=[0, 1, 2],
        )
        model = _make_model(
            _make_config(
                num_hidden_layers=3,
                head_dim=64,
                num_key_value_heads=4,
                num_attention_heads=4,
            )
        )
        result = compress_model_kv(cache, model, model_info=info)
        # Result should have 3 layers
        kv = extract_kv(result)
        assert len(kv) == 3
        for k, v in kv:
            assert k.shape == (1, 4, 16, 64)
            assert v.shape == (1, 4, 16, 64)

    def test_skip_none_layers(self):
        """Hybrid cache: None entries preserved, attention layers compressed."""
        cache = _make_key_cache_obj(n_layers=4, none_layers=[1, 3])
        info = ModelKVInfo(
            n_layers=4,
            head_dim=64,
            num_kv_heads=4,
            num_query_heads=4,
            attention_layers=[0, 2],
            skip_layers=[1, 3],
        )
        model = _make_model()
        result = compress_model_kv(cache, model, model_info=info)
        assert result.key_cache[1] is None
        assert result.key_cache[3] is None
        assert result.key_cache[0] is not None
        assert result.key_cache[2] is not None

    def test_mla_raises_valueerror(self):
        """MLA model raises ValueError."""
        cache = _make_tuple_cache()
        info = ModelKVInfo(
            n_layers=4,
            head_dim=64,
            num_kv_heads=4,
            num_query_heads=32,
            is_latent_kv=True,
        )
        model = _make_model()
        with pytest.raises(ValueError, match="Multi-Latent Attention"):
            compress_model_kv(cache, model, model_info=info)

    def test_gqa_uses_for_gqa(self):
        """GQA model creates GQA-aware compressor."""
        pytest.importorskip("transformers")
        cache = _make_tuple_cache(n_layers=2, H=4, D=64)
        info = ModelKVInfo(
            n_layers=2,
            head_dim=64,
            num_kv_heads=4,
            num_query_heads=32,
            attention_layers=[0, 1],
        )
        model = _make_model(
            _make_config(
                num_hidden_layers=2,
                head_dim=64,
                num_key_value_heads=4,
                num_attention_heads=32,
            )
        )
        result = compress_model_kv(cache, model, model_info=info)
        kv = extract_kv(result)
        assert len(kv) == 2
        assert kv[0][0].shape[-1] == 64

    def test_custom_model_info(self):
        """Passing model_info skips auto-detection."""
        pytest.importorskip("transformers")
        cache = _make_tuple_cache(n_layers=2, D=64)
        info = ModelKVInfo(
            n_layers=2,
            head_dim=64,
            num_kv_heads=4,
            num_query_heads=4,
            attention_layers=[0, 1],
        )
        # Model config doesn't matter since we pass model_info
        model = _make_model(
            _make_config(
                num_hidden_layers=2,
                head_dim=64,
                num_key_value_heads=4,
                num_attention_heads=4,
            )
        )
        result = compress_model_kv(cache, model, model_info=info)
        kv = extract_kv(result)
        assert len(kv) == 2


class TestGetLayers:
    def test_model_model_layers(self):
        """Finds layers at model.model.layers."""
        layer = _make_attn_layer()
        model = SimpleNamespace(model=SimpleNamespace(layers=[layer]))
        assert _get_layers(model) == [layer]

    def test_transformer_h(self):
        """Finds layers at model.transformer.h."""
        layer = _make_attn_layer()
        model = SimpleNamespace(transformer=SimpleNamespace(h=[layer]))
        assert _get_layers(model) == [layer]

    def test_empty_fallback(self):
        """Unknown structure returns empty list."""
        model = SimpleNamespace()
        assert _get_layers(model) == []


class TestHasKvProj:
    def test_with_k_proj(self):
        """Layer with self_attn.k_proj returns True."""
        layer = _make_attn_layer()
        assert _has_kv_proj(layer) is True

    def test_without_attention(self):
        """Layer without attention returns False."""
        layer = _make_non_attn_layer()
        assert _has_kv_proj(layer) is False
