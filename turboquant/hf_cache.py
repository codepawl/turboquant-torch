"""HuggingFace-compatible dynamic cache with TurboQuant compression.

Provides a drop-in replacement for transformers DynamicCache that
stores KV states in full precision during generation, then compresses
all layers on demand via ``compress_all()``.

When ``transformers`` is installed, :class:`TurboQuantDynamicCache`
subclasses ``transformers.DynamicCache`` directly, ensuring full
compatibility with ``model.generate()`` and its evolving internal API.
When ``transformers`` is not available, a lightweight standalone
implementation is used instead.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch

from .compat import ModelKVInfo, detect_model_kv_info
from .kv_cache import TurboQuantKVCache

try:
    from transformers.cache_utils import DynamicCache as _HFDynamicCache

    _HAS_HF_CACHE = True
except ImportError:
    _HAS_HF_CACHE = False


def _compress_layers(
    cache: TurboQuantDynamicCache,
    get_kv: Any,
    set_kv: Any,
    num_layers: int,
) -> dict[str, Any]:
    """Shared compression logic for both HF-subclass and standalone paths.

    Args:
        cache: The cache instance (for config attributes).
        get_kv: Callable(layer_idx) -> (key_tensor | None, value_tensor | None).
        set_kv: Callable(layer_idx, key_tensor, value_tensor) -> None.
        num_layers: Number of layers to iterate over.

    Returns:
        Stats dict with keys: layers_compressed, layers_skipped,
        original_mb, compressed_mb, ratio.
    """
    layers_compressed = 0
    layers_skipped = 0
    total_original_mb = 0.0
    total_compressed_mb = 0.0

    for i in range(num_layers):
        k, v = get_kv(i)
        if i in cache._skip_layers or k is None or v is None:
            layers_skipped += 1
            continue

        # Skip empty tensors (from uninitialized HF layers)
        if k.numel() == 0 or v.numel() == 0:
            layers_skipped += 1
            continue

        head_dim = k.shape[-1]
        if (
            cache.model_info is not None
            and cache.model_info.num_kv_heads != cache.model_info.num_query_heads
        ):
            compressor = TurboQuantKVCache.for_gqa(
                head_dim=head_dim,
                num_kv_heads=cache.model_info.num_kv_heads,
                num_query_heads=cache.model_info.num_query_heads,
                bit_width=cache.bit_width,
                residual_length=cache.residual_length,
                n_outlier_channels=cache.n_outlier_channels,
            )
        else:
            compressor = TurboQuantKVCache(
                head_dim=head_dim,
                bit_width=cache.bit_width,
                residual_length=cache.residual_length,
                n_outlier_channels=cache.n_outlier_channels,
            )

        B, H, S, _D = k.shape
        orig_mb, comp_mb, _ratio = compressor.memory_savings(B, H, S)
        total_original_mb += orig_mb
        total_compressed_mb += comp_mb

        compressed = compressor.compress(k.float(), v.float())
        new_k = compressor.decompress_keys(compressed).to(k.dtype)
        new_v = compressor.decompress_values(compressed).to(v.dtype)
        set_kv(i, new_k, new_v)
        layers_compressed += 1

    ratio = total_original_mb / max(total_compressed_mb, 1e-10)
    return {
        "layers_compressed": layers_compressed,
        "layers_skipped": layers_skipped,
        "original_mb": total_original_mb,
        "compressed_mb": total_compressed_mb,
        "ratio": ratio,
    }


if _HAS_HF_CACHE:

    class TurboQuantDynamicCache(_HFDynamicCache):  # type: ignore[misc]
        """Dynamic KV cache that supports TurboQuant compression.

        Subclasses ``transformers.DynamicCache`` for full compatibility
        with ``model.generate()``. Accumulates key/value states during
        generation in full precision, then compresses all layers at once
        via :meth:`compress_all`.

        Args:
            bit_width: Bits per coordinate for quantization.
            residual_length: Number of recent tokens kept in fp16 (sliding window).
            n_outlier_channels: Outlier channels routed to high-precision storage.
            model_info: Pre-computed model architecture info. If provided,
                ``skip_layers`` from the info will be respected during compression.
        """

        def __init__(
            self,
            bit_width: int = 3,
            residual_length: int = 0,
            n_outlier_channels: int = 0,
            model_info: ModelKVInfo | None = None,
        ) -> None:
            super().__init__()
            self.bit_width = bit_width
            self.residual_length = residual_length
            self.n_outlier_channels = n_outlier_channels
            self.model_info = model_info
            self._skip_layers: set[int] = (
                set(model_info.skip_layers) if model_info is not None else set()
            )

        @property
        def key_cache(self) -> list[torch.Tensor]:
            """List of key tensors per layer.

            Returns:
                List of key tensors, with empty tensors for uninitialized layers.
            """
            return [
                layer.keys if hasattr(layer, "keys") and layer.keys is not None else torch.empty(0)
                for layer in self.layers
            ]

        @property
        def value_cache(self) -> list[torch.Tensor]:
            """List of value tensors per layer.

            Returns:
                List of value tensors, with empty tensors for uninitialized layers.
            """
            return [
                (
                    layer.values
                    if hasattr(layer, "values") and layer.values is not None
                    else torch.empty(0)
                )
                for layer in self.layers
            ]

        @property
        def _keys(self) -> list[torch.Tensor | None]:
            """Backward-compatible access to key tensors.

            Returns:
                List of key tensors (or None for uninitialized layers).
            """
            result: list[torch.Tensor | None] = []
            for layer in self.layers:
                keys = getattr(layer, "keys", None)
                if keys is not None and keys.numel() > 0:
                    result.append(keys)
                else:
                    result.append(None)
            return result

        @property
        def _values(self) -> list[torch.Tensor | None]:
            """Backward-compatible access to value tensors.

            Returns:
                List of value tensors (or None for uninitialized layers).
            """
            result: list[torch.Tensor | None] = []
            for layer in self.layers:
                values = getattr(layer, "values", None)
                if values is not None and values.numel() > 0:
                    result.append(values)
                else:
                    result.append(None)
            return result

        def compress_all(self) -> dict[str, Any]:
            """Compress all layers using TurboQuantKVCache.

            Layers in ``_skip_layers`` are left uncompressed. Each layer's
            KV tensors are compressed and then immediately decompressed
            back to dense tensors (lossy round-trip).

            Returns:
                Stats dict with keys: layers_compressed, layers_skipped,
                original_mb, compressed_mb, ratio.
            """

            def get_kv(i: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
                if i >= len(self.layers):
                    return None, None
                layer = self.layers[i]
                k = getattr(layer, "keys", None)
                v = getattr(layer, "values", None)
                return k, v

            def set_kv(i: int, k: torch.Tensor, v: torch.Tensor) -> None:
                self.layers[i].keys = k
                self.layers[i].values = v

            return _compress_layers(self, get_kv, set_kv, len(self.layers))

        def to_legacy_cache(self) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
            """Convert to legacy tuple-of-tuples format.

            Returns:
                Tuple of (key, value) pairs per layer.
            """
            return tuple(zip(self.key_cache, self.value_cache, strict=True))

        def crop(self, max_length: int) -> None:
            """Crop all layers to at most ``max_length`` tokens.

            Args:
                max_length: Maximum sequence length to retain.
            """
            for layer in self.layers:
                if hasattr(layer, "crop"):
                    layer.crop(max_length)  # type: ignore[attr-defined]

        def get_seq_length(self, layer_idx: int = 0) -> int:
            """Return the sequence length for a given layer.

            Args:
                layer_idx: Layer index to query.

            Returns:
                Current sequence length, or 0 if the layer has no cache.
            """
            if layer_idx >= len(self.layers):
                return 0
            return self.layers[layer_idx].get_seq_length()

        def get_max_cache_shape(self, layer_idx: int = 0) -> int | None:  # type: ignore[override]
            """Return the maximum cache shape. Always None (unbounded).

            Args:
                layer_idx: Layer index (unused).

            Returns:
                None, indicating no maximum cache size.
            """
            return None

        def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
            yield from zip(self.key_cache, self.value_cache, strict=True)

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
            return self.key_cache[idx], self.value_cache[idx]

        @staticmethod
        def from_model(
            model: torch.nn.Module,
            **kwargs: Any,
        ) -> TurboQuantDynamicCache:
            """Create a cache from a HuggingFace model with auto-configuration.

            Auto-selects bit_width based on head_dim: 3-bit for head_dim >= 128,
            4-bit otherwise. Explicit ``bit_width`` in kwargs overrides this.

            Args:
                model: HuggingFace model with ``.config`` attribute.
                **kwargs: Forwarded to :class:`TurboQuantDynamicCache` constructor.

            Returns:
                Configured TurboQuantDynamicCache instance.
            """
            info = detect_model_kv_info(model)
            if "bit_width" not in kwargs:
                kwargs["bit_width"] = 3 if info.head_dim >= 128 else 4
            return TurboQuantDynamicCache(model_info=info, **kwargs)

else:

    class TurboQuantDynamicCache:  # type: ignore[no-redef]
        """Dynamic KV cache that supports TurboQuant compression.

        Standalone implementation for use without ``transformers``.
        Accumulates key/value states during generation in full precision,
        then compresses all layers at once via :meth:`compress_all`.

        Args:
            bit_width: Bits per coordinate for quantization.
            residual_length: Number of recent tokens kept in fp16 (sliding window).
            n_outlier_channels: Outlier channels routed to high-precision storage.
            model_info: Pre-computed model architecture info. If provided,
                ``skip_layers`` from the info will be respected during compression.
        """

        def __init__(
            self,
            bit_width: int = 3,
            residual_length: int = 0,
            n_outlier_channels: int = 0,
            model_info: ModelKVInfo | None = None,
        ) -> None:
            self.bit_width = bit_width
            self.residual_length = residual_length
            self.n_outlier_channels = n_outlier_channels
            self.model_info = model_info

            self._keys: list[torch.Tensor | None] = []
            self._values: list[torch.Tensor | None] = []
            self._skip_layers: set[int] = (
                set(model_info.skip_layers) if model_info is not None else set()
            )

        def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            cache_kwargs: dict[str, Any] | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Store or concatenate KV states for a layer.

            Args:
                key_states: Key tensor for the current step.
                value_states: Value tensor for the current step.
                layer_idx: Decoder layer index.
                cache_kwargs: Additional cache arguments (unused, for API compat).

            Returns:
                Tuple of (full_keys, full_values) for this layer.
            """
            while len(self._keys) <= layer_idx:
                self._keys.append(None)
                self._values.append(None)

            if self._keys[layer_idx] is None:
                self._keys[layer_idx] = key_states
                self._values[layer_idx] = value_states
            else:
                self._keys[layer_idx] = torch.cat(
                    [self._keys[layer_idx], key_states],  # type: ignore[list-item]
                    dim=-2,
                )
                self._values[layer_idx] = torch.cat(
                    [self._values[layer_idx], value_states],  # type: ignore[list-item]
                    dim=-2,
                )

            return self._keys[layer_idx], self._values[layer_idx]  # type: ignore[return-value]

        def get_seq_length(self, layer_idx: int = 0) -> int:
            """Return the sequence length for a given layer.

            Args:
                layer_idx: Layer index to query.

            Returns:
                Current sequence length, or 0 if the layer has no cache.
            """
            if layer_idx >= len(self._keys) or self._keys[layer_idx] is None:
                return 0
            return self._keys[layer_idx].shape[-2]  # type: ignore[union-attr]

        def get_max_cache_shape(self) -> int | None:
            """Return the maximum cache shape. Always None (unbounded).

            Returns:
                None, indicating no maximum cache size.
            """
            return None

        @property
        def key_cache(self) -> list[torch.Tensor]:
            """List of key tensors, with empty tensors for None entries.

            Returns:
                List of key tensors per layer.
            """
            return [k if k is not None else torch.empty(0) for k in self._keys]

        @property
        def value_cache(self) -> list[torch.Tensor]:
            """List of value tensors, with empty tensors for None entries.

            Returns:
                List of value tensors per layer.
            """
            return [v if v is not None else torch.empty(0) for v in self._values]

        def compress_all(self) -> dict[str, Any]:
            """Compress all layers using TurboQuantKVCache.

            Layers in ``_skip_layers`` are left uncompressed. Each layer's
            KV tensors are compressed and then immediately decompressed
            back to dense tensors (lossy round-trip).

            Returns:
                Stats dict with keys: layers_compressed, layers_skipped,
                original_mb, compressed_mb, ratio.
            """

            def get_kv(i: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
                if i >= len(self._keys):
                    return None, None
                return self._keys[i], self._values[i]

            def set_kv(i: int, k: torch.Tensor, v: torch.Tensor) -> None:
                self._keys[i] = k
                self._values[i] = v

            return _compress_layers(self, get_kv, set_kv, len(self._keys))

        def __len__(self) -> int:
            return len(self._keys)

        def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
            yield from zip(self.key_cache, self.value_cache, strict=True)

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
            return self.key_cache[idx], self.value_cache[idx]

        def to_legacy_cache(self) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
            """Convert to legacy tuple-of-tuples format.

            Returns:
                Tuple of (key, value) pairs per layer.
            """
            return tuple(zip(self.key_cache, self.value_cache, strict=True))

        def crop(self, max_length: int) -> None:
            """Crop all layers to at most ``max_length`` tokens.

            Args:
                max_length: Maximum sequence length to retain.
            """
            for i in range(len(self._keys)):
                if self._keys[i] is not None:
                    self._keys[i] = self._keys[i][:, :, :max_length, :]  # type: ignore[index]
                    self._values[i] = self._values[i][:, :, :max_length, :]  # type: ignore[index]

        @staticmethod
        def from_model(
            model: torch.nn.Module,
            **kwargs: Any,
        ) -> TurboQuantDynamicCache:
            """Create a cache from a HuggingFace model with auto-configuration.

            Auto-selects bit_width based on head_dim: 3-bit for head_dim >= 128,
            4-bit otherwise. Explicit ``bit_width`` in kwargs overrides this.

            Args:
                model: HuggingFace model with ``.config`` attribute.
                **kwargs: Forwarded to :class:`TurboQuantDynamicCache` constructor.

            Returns:
                Configured TurboQuantDynamicCache instance.
            """
            info = detect_model_kv_info(model)
            if "bit_width" not in kwargs:
                kwargs["bit_width"] = 3 if info.head_dim >= 128 else 4
            return TurboQuantDynamicCache(model_info=info, **kwargs)
