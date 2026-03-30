"""Model architecture compatibility detection.

Detects KV cache format and architecture quirks for different model
families to ensure TurboQuant handles them correctly. Provides a
high-level ``compress_model_kv`` function that auto-handles format
detection, hybrid layers, and GQA configuration.

Supported cache formats:
  - transformers >=5.x DynamicCache: ``.layers[i].keys / .values``
  - transformers 4.x DynamicCache: ``.key_cache[i] / .value_cache[i]``
  - Legacy tuple format: ``past_kv[i] = (key, value)``
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import torch

from .kv_cache import TurboQuantKVCache


@dataclass
class ModelKVInfo:
    """KV cache architecture info extracted from a model.

    Args:
        n_layers: Total number of layers (including non-attention).
        head_dim: Dimension of each attention head.
        num_kv_heads: Number of key-value heads.
        num_query_heads: Number of query heads.
        cache_format: Detected cache format (filled by ``extract_kv``).
        attention_layers: Layer indices that have attention (KV cache).
        skip_layers: Layer indices without attention (Mamba, linear, etc.).
        sliding_window: Sliding window size from config, or None.
        shared_kv_layers: Layer groups sharing KV heads (CLA), or None.
        is_latent_kv: True for MLA models (DeepSeek) where KV is latent.
    """

    n_layers: int
    head_dim: int
    num_kv_heads: int
    num_query_heads: int
    cache_format: str | None = None
    attention_layers: list[int] = field(default_factory=list)
    skip_layers: list[int] = field(default_factory=list)
    sliding_window: int | None = None
    shared_kv_layers: list[int] | None = None
    is_latent_kv: bool = False


def _get_layers(model: Any) -> list[Any]:
    """Extract decoder layers from a HuggingFace model.

    Tries known attribute paths across model families.
    """
    candidates = []
    # Llama, Mistral, Qwen, Gemma, Phi-3
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        candidates.append(model.model.layers)
    # GPT-2, GPT-Neo
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        candidates.append(model.transformer.h)
    # GPT-NeoX, Pythia
    elif hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        candidates.append(model.gpt_neox.layers)
    # OPT, encoder-decoder
    elif hasattr(model, "model") and hasattr(model.model, "decoder"):
        decoder = model.model.decoder
        if hasattr(decoder, "layers"):
            candidates.append(decoder.layers)

    for candidate in candidates:
        try:
            return list(candidate)
        except TypeError:
            continue
    return []


def _has_kv_proj(layer: Any) -> bool:
    """Check whether a layer has KV projection weights (is an attention layer)."""
    attn = getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
    if attn is None:
        return False
    return hasattr(attn, "k_proj") or hasattr(attn, "key") or hasattr(attn, "key_proj")


def detect_model_kv_info(model: torch.nn.Module) -> ModelKVInfo:
    """Auto-detect KV cache architecture info from a HuggingFace model.

    Handles standard transformers, GQA, sliding window, hybrid
    architectures (Qwen3.5, Jamba), and MLA (DeepSeek).

    Args:
        model: HuggingFace model with a ``.config`` attribute.

    Returns:
        ModelKVInfo with detected architecture parameters.

    Raises:
        ValueError: If the model has no ``config`` attribute.
    """
    if not hasattr(model, "config"):
        raise ValueError("Model has no .config attribute — not a HuggingFace model?")

    # Handle VLM models that nest text config
    cfg: Any = getattr(model.config, "text_config", model.config)

    n_layers: int = cfg.num_hidden_layers
    num_heads: int = cfg.num_attention_heads
    head_dim: int = getattr(cfg, "head_dim", cfg.hidden_size // num_heads)
    num_kv_heads: int = getattr(cfg, "num_key_value_heads", num_heads)
    sliding_window = getattr(cfg, "sliding_window", None)

    # MLA detection (DeepSeek-V2/V3)
    is_latent_kv = hasattr(cfg, "kv_lora_rank") or hasattr(cfg, "q_lora_rank")

    # Hybrid layer detection — first try config-level markers
    attention_layers: list[int] | None = None
    if hasattr(cfg, "hybrid_attention_layers"):
        attention_layers = list(cfg.hybrid_attention_layers)
    elif hasattr(cfg, "attention_layers"):
        attention_layers = list(cfg.attention_layers)

    # Fallback: introspect model layers
    if attention_layers is None:
        layers = _get_layers(model)
        if layers:
            attention_layers = [i for i, layer in enumerate(layers) if _has_kv_proj(layer)]
            # Only treat as hybrid if some layers were skipped
            if len(attention_layers) == len(layers):
                attention_layers = list(range(n_layers))
        else:
            attention_layers = list(range(n_layers))

    skip_layers = sorted(set(range(n_layers)) - set(attention_layers))

    return ModelKVInfo(
        n_layers=n_layers,
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        num_query_heads=num_heads,
        attention_layers=sorted(attention_layers),
        skip_layers=skip_layers,
        sliding_window=sliding_window,
        is_latent_kv=is_latent_kv,
    )


def extract_kv(
    past_key_values: object,
) -> list[tuple[torch.Tensor | None, torch.Tensor | None]]:
    """Extract per-layer (key, value) pairs from any cache format.

    Handles transformers >=5.x DynamicCache (``.layers``), 4.x
    DynamicCache (``.key_cache``), and legacy tuple formats.

    Args:
        past_key_values: Cache object from a model forward pass.

    Returns:
        List of (key, value) tuples per layer. Entries may be
        ``(None, None)`` for non-attention layers in hybrid models.
    """
    # transformers >=5.x: .layers[i].keys / .values
    if hasattr(past_key_values, "layers"):
        layers = past_key_values.layers  # type: ignore[union-attr]
        if len(layers) > 0 and hasattr(layers[0], "keys"):
            return [(layer.keys, layer.values) for layer in layers]

    # transformers 4.x: .key_cache / .value_cache
    if hasattr(past_key_values, "key_cache"):
        keys = past_key_values.key_cache  # type: ignore[union-attr]
        values = past_key_values.value_cache  # type: ignore[attr-defined]
        result: list[tuple[torch.Tensor | None, torch.Tensor | None]] = []
        for k, v in zip(keys, values, strict=True):
            if k is None:
                result.append((None, None))
            else:
                result.append((k, v))
        return result

    # Legacy tuple format
    return [(item[0], item[1]) for item in past_key_values]  # type: ignore[union-attr,attr-defined]


def compress_model_kv(
    past_key_values: object,
    model: torch.nn.Module,
    *,
    bit_width: int = 3,
    residual_length: int = 0,
    n_outlier_channels: int = 0,
    model_info: ModelKVInfo | None = None,
) -> object:
    """Compress a model's KV cache, auto-handling architecture quirks.

    Detects the model architecture, skips non-attention layers in
    hybrid models, and uses GQA-aware configuration when appropriate.
    Returns a reconstructed cache of the same type as the input.

    Args:
        past_key_values: Cache from model forward pass.
        model: HuggingFace model (used for auto-detection).
        bit_width: Bits per coordinate.
        residual_length: Sliding window size (recent tokens kept in fp16).
        n_outlier_channels: Outlier channels routed to high-precision storage.
        model_info: Pre-computed ModelKVInfo to skip auto-detection.

    Returns:
        Reconstructed cache object with compressed-then-decompressed
        KV tensors, suitable for passing to ``model(past_key_values=...)``.

    Raises:
        ValueError: If model uses MLA (latent KV) which is already compressed.
    """
    if model_info is None:
        model_info = detect_model_kv_info(model)

    if model_info.is_latent_kv:
        raise ValueError(
            "Model uses Multi-Latent Attention (MLA). "
            "KV cache is already compressed by the model architecture. "
            "TurboQuant compression is not needed."
        )

    # Build compressor — use GQA-aware config when appropriate
    if model_info.num_kv_heads != model_info.num_query_heads:
        compressor = TurboQuantKVCache.for_gqa(
            head_dim=model_info.head_dim,
            num_kv_heads=model_info.num_kv_heads,
            num_query_heads=model_info.num_query_heads,
            bit_width=bit_width,
            residual_length=residual_length,
            n_outlier_channels=n_outlier_channels,
        )
    else:
        compressor = TurboQuantKVCache(
            head_dim=model_info.head_dim,
            bit_width=bit_width,
            residual_length=residual_length,
            n_outlier_channels=n_outlier_channels,
        )

    kv_pairs = extract_kv(past_key_values)
    skip_set = set(model_info.skip_layers)
    has_none = any(k is None for k, _v in kv_pairs)

    if has_none or skip_set:
        # Hybrid model — deep copy and replace only attention layers
        result = copy.deepcopy(past_key_values)
        for i, (k, v) in enumerate(kv_pairs):
            if k is None or i in skip_set:
                continue
            dtype = k.dtype
            compressed = compressor.compress(k.float(), v.float())  # type: ignore[union-attr]
            k_hat = compressor.decompress_keys(compressed).to(dtype)
            v_hat = compressor.decompress_values(compressed).to(dtype)
            _set_layer_kv(result, i, k_hat, v_hat)
    else:
        # Standard model — build fresh DynamicCache
        try:
            from transformers.cache_utils import DynamicCache
        except ImportError as e:
            raise ImportError(
                "transformers is required for compress_model_kv. "
                "Install with: pip install transformers"
            ) from e

        result = DynamicCache()
        for i, (k, v) in enumerate(kv_pairs):
            dtype = k.dtype  # type: ignore[union-attr]
            compressed = compressor.compress(k.float(), v.float())  # type: ignore[union-attr]
            k_hat = compressor.decompress_keys(compressed).to(dtype)
            v_hat = compressor.decompress_values(compressed).to(dtype)
            result.update(k_hat, v_hat, i)

    return result


def _set_layer_kv(
    cache: object,
    layer_idx: int,
    keys: torch.Tensor,
    values: torch.Tensor,
) -> None:
    """Set KV tensors for a layer in a cache object."""
    if hasattr(cache, "key_cache"):
        cache.key_cache[layer_idx] = keys  # type: ignore[index]
        cache.value_cache[layer_idx] = values  # type: ignore[index,attr-defined]
    elif hasattr(cache, "layers"):
        cache.layers[layer_idx].keys = keys  # type: ignore[union-attr]
        cache.layers[layer_idx].values = values  # type: ignore[union-attr]
