"""Adaptive per-layer bit allocation.

Different transformer layers have different sensitivity to quantization.
This module provides strategies for allocating bit budgets across layers
to minimize total model quality degradation.

Strategies:
  - uniform: Same bit width for all layers (current default behavior).
  - gradient: Linearly or step-wise increase bits from early to late layers.
  - calibration: Auto-detect sensitivity by measuring MSE on a forward pass.

Reference: TurboQuant paper Section 5 (discussion on layer sensitivity)
"""

from __future__ import annotations

from typing import Literal

import torch

from .kv_cache import TurboQuantKVCache


def uniform_allocation(n_layers: int, bit_width: int) -> list[int]:
    """Same bit width for all layers.

    Args:
        n_layers: Number of transformer layers.
        bit_width: Bit width to use uniformly.

    Returns:
        List of identical bit_widths, one per layer.
    """
    return [bit_width] * n_layers


def gradient_allocation(
    n_layers: int,
    min_bits: int = 2,
    max_bits: int = 4,
    strategy: Literal["linear", "step"] = "linear",
) -> list[int]:
    """Gradually increase bits from early to later layers.

    Later layers are closer to the output and more sensitive to
    quantization error, so they receive more bits.

    Args:
        n_layers: Total number of layers.
        min_bits: Bits for earliest layers.
        max_bits: Bits for latest layers.
        strategy: "linear" for smooth gradient, "step" for 3-tier allocation.

    Returns:
        List of bit_widths, one per layer.

    Raises:
        ValueError: If strategy is not recognized.
    """
    if strategy == "linear":
        bits = []
        for i in range(n_layers):
            t = i / max(n_layers - 1, 1)
            b = min_bits + t * (max_bits - min_bits)
            bits.append(round(b))
        return bits

    if strategy == "step":
        third = n_layers // 3
        mid_bits = (min_bits + max_bits + 1) // 2
        return [min_bits] * third + [mid_bits] * (n_layers - 2 * third) + [max_bits] * third

    raise ValueError(f"Unknown strategy: {strategy!r}")


def calibration_allocation(
    model: torch.nn.Module,
    tokenizer: object,
    calibration_text: str,
    bit_options: list[int] | None = None,
    target_avg_bits: float = 3.0,
) -> list[int]:
    """Auto-detect per-layer sensitivity via calibration.

    Runs a forward pass, measures quantization MSE per layer,
    then allocates more bits to sensitive layers while staying
    within the target average bit budget.

    Args:
        model: HuggingFace model with ``use_cache=True`` support.
        tokenizer: Tokenizer with ``__call__`` returning input_ids.
        calibration_text: Text to run through the model.
        bit_options: Available bit widths (default [2, 3, 4]).
        target_avg_bits: Target average bits across all layers.

    Returns:
        List of bit_widths, one per layer.
    """
    if bit_options is None:
        bit_options = [2, 3, 4]

    # Tokenize and run forward pass
    inputs = tokenizer(calibration_text, return_tensors="pt")  # type: ignore[operator]
    inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs, use_cache=True)

    past_kv = out.past_key_values

    # Extract per-layer keys and values from different cache formats:
    # - transformers >=5.x DynamicCache: .layers[i].keys / .values
    # - transformers 4.x DynamicCache: .key_cache[i] / .value_cache[i]
    # - legacy tuple format: past_kv[i] = (key, value)
    if hasattr(past_kv, "layers") and hasattr(past_kv.layers[0], "keys"):
        n_layers = len(past_kv.layers)
        layer_keys = [past_kv.layers[i].keys for i in range(n_layers)]
        layer_values = [past_kv.layers[i].values for i in range(n_layers)]
    elif hasattr(past_kv, "key_cache"):
        n_layers = len(past_kv.key_cache)
        layer_keys = past_kv.key_cache
        layer_values = past_kv.value_cache
    else:
        n_layers = len(past_kv)
        layer_keys = [kv[0] for kv in past_kv]
        layer_values = [kv[1] for kv in past_kv]

    head_dim = layer_keys[0].shape[-1]

    # Measure MSE at lowest bit_width for each layer
    min_bits = min(bit_options)
    max_bits = max(bit_options)
    cache = TurboQuantKVCache(head_dim=head_dim, bit_width=min_bits, residual_length=0)

    sensitivities = []
    for i in range(n_layers):
        k = layer_keys[i].float()
        v = layer_values[i].float()

        compressed = cache.compress(k, v)
        k_hat = cache.decompress_keys(compressed)
        mse = ((k - k_hat) ** 2).mean().item()
        sensitivities.append(mse)

    # Allocate: start with min_bits, distribute remaining budget to most sensitive
    allocation = [min_bits] * n_layers
    total_budget = int(target_avg_bits * n_layers)
    remaining = total_budget - min_bits * n_layers

    most_sensitive = sorted(range(n_layers), key=lambda i: sensitivities[i], reverse=True)
    for idx in most_sensitive:
        if remaining <= 0:
            break
        can_add = min(max_bits - allocation[idx], remaining)
        allocation[idx] += can_add
        remaining -= can_add

    return allocation


class AdaptiveKVCache:
    """KV cache with per-layer bit allocation.

    Each layer gets its own TurboQuantKVCache instance with potentially
    different bit_width, enabling fine-grained quality/compression trade-offs.

    Args:
        head_dim: Attention head dimension.
        layer_bits: List of bit_widths, one per layer.
        residual_length: Sliding window size for all layers.
        n_outlier_channels: Outlier channels per layer (0 to disable).
    """

    def __init__(
        self,
        head_dim: int,
        layer_bits: list[int],
        residual_length: int = 128,
        n_outlier_channels: int = 0,
    ):
        self.head_dim = head_dim
        self.layer_bits = layer_bits
        self.n_layers = len(layer_bits)

        self.caches = [
            TurboQuantKVCache(
                head_dim=head_dim,
                bit_width=bits,
                residual_length=residual_length,
                n_outlier_channels=n_outlier_channels,
            )
            for bits in layer_bits
        ]

    def compress_layer(
        self,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        **kwargs: object,
    ) -> object:
        """Compress KV for a single layer.

        Args:
            layer_idx: Layer index.
            keys: Key tensor (batch, heads, seq, head_dim).
            values: Value tensor (batch, heads, seq, head_dim).

        Returns:
            CompressedKV for this layer.
        """
        return self.caches[layer_idx].compress(keys, values, **kwargs)  # type: ignore[arg-type]

    def decompress_layer_keys(self, layer_idx: int, compressed: object) -> torch.Tensor:
        """Decompress keys for a single layer."""
        return self.caches[layer_idx].decompress_keys(compressed)  # type: ignore[arg-type]

    def decompress_layer_values(self, layer_idx: int, compressed: object) -> torch.Tensor:
        """Decompress values for a single layer."""
        return self.caches[layer_idx].decompress_values(compressed)  # type: ignore[arg-type]

    def attention_layer(
        self,
        layer_idx: int,
        query: torch.Tensor,
        compressed: object,
        **kwargs: object,
    ) -> torch.Tensor:
        """Compute attention for a single layer."""
        return self.caches[layer_idx].attention(query, compressed, **kwargs)  # type: ignore[arg-type]

    def summary(self) -> str:
        """Return human-readable bit allocation summary."""
        lines = [f"AdaptiveKVCache: {self.n_layers} layers, head_dim={self.head_dim}"]
        for i, bits in enumerate(self.layer_bits):
            lines.append(f"  Layer {i:2d}: {bits}-bit")
        avg = sum(self.layer_bits) / len(self.layer_bits)
        lines.append(f"  Average: {avg:.1f} bits/layer")
        return "\n".join(lines)

    @staticmethod
    def from_model(
        model: torch.nn.Module,
        tokenizer: object,
        head_dim: int,
        calibration_text: str = "The quick brown fox jumps over the lazy dog. " * 20,
        target_avg_bits: float = 3.0,
        residual_length: int = 128,
        n_outlier_channels: int = 0,
    ) -> AdaptiveKVCache:
        """Create AdaptiveKVCache with auto-calibrated bit allocation.

        Runs a calibration forward pass to measure per-layer sensitivity,
        then allocates bits accordingly.

        Args:
            model: HuggingFace model.
            tokenizer: Tokenizer.
            head_dim: Attention head dimension.
            calibration_text: Text for sensitivity measurement.
            target_avg_bits: Target average bits.
            residual_length: Sliding window size.
            n_outlier_channels: Outlier channels per layer.

        Returns:
            Configured AdaptiveKVCache.
        """
        allocation = calibration_allocation(
            model,
            tokenizer,
            calibration_text,
            target_avg_bits=target_avg_bits,
        )
        return AdaptiveKVCache(
            head_dim=head_dim,
            layer_bits=allocation,
            residual_length=residual_length,
            n_outlier_channels=n_outlier_channels,
        )
