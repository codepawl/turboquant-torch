"""One-liner API for wrapping HuggingFace models with TurboQuant compression.

Usage::

    from turboquant import wrap

    model = wrap(your_hf_model)
    output = model.generate(input_ids, max_new_tokens=100)
"""

from __future__ import annotations

from typing import Any

import torch

from .compat import detect_model_kv_info
from .hf_cache import TurboQuantDynamicCache


class TurboQuantWrapper:
    """Wraps a HuggingFace model to inject TurboQuant KV cache compression.

    The wrapper transparently injects a :class:`TurboQuantDynamicCache`
    into ``generate()`` calls when no ``past_key_values`` is provided.

    Args:
        model: HuggingFace model with ``.config`` and ``.generate()``.
        bit_width: Bits per coordinate, or None for auto-selection
            (3 if head_dim >= 128, else 4).
        residual_length: Recent tokens kept in fp16 (sliding window).
        n_outlier_channels: Outlier channels for high-precision storage.
        verbose: If True, print compression stats after generate().
    """

    def __init__(
        self,
        model: Any,
        bit_width: int | None = None,
        residual_length: int = 0,
        n_outlier_channels: int = 0,
        verbose: bool = False,
    ) -> None:
        self._model = model
        self._info = detect_model_kv_info(model)
        self._bit_width = (
            bit_width if bit_width is not None else (3 if self._info.head_dim >= 128 else 4)
        )
        self._residual_length = residual_length
        self._n_outlier_channels = n_outlier_channels
        self._verbose = verbose

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Generate with automatic TurboQuant cache injection.

        If ``past_key_values`` is not already provided, a fresh
        :class:`TurboQuantDynamicCache` is injected.

        Args:
            *args: Positional arguments forwarded to model.generate().
            **kwargs: Keyword arguments forwarded to model.generate().

        Returns:
            Output from model.generate().
        """
        if "past_key_values" not in kwargs or kwargs["past_key_values"] is None:
            kwargs["past_key_values"] = TurboQuantDynamicCache(
                bit_width=self._bit_width,
                residual_length=self._residual_length,
                n_outlier_channels=self._n_outlier_channels,
                model_info=self._info,
            )

        result = self._model.generate(*args, **kwargs)

        if self._verbose:
            cache = kwargs.get("past_key_values")
            if isinstance(cache, TurboQuantDynamicCache):
                stats = cache.compress_all()
                print(
                    f"TurboQuant: {stats['layers_compressed']} layers compressed, "
                    f"ratio={stats['ratio']:.1f}x"
                )

        return result

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass delegated to the wrapped model.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Output from model forward pass.
        """
        return self._model(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._model, name)

    @property
    def config(self) -> Any:
        """Model configuration."""
        return self._model.config

    @property
    def device(self) -> torch.device:
        """Device of the wrapped model."""
        return torch.device(next(self._model.parameters()).device)

    def to(self, *args: Any, **kwargs: Any) -> TurboQuantWrapper:
        """Move wrapped model to device/dtype.

        Returns:
            Self for method chaining.
        """
        self._model.to(*args, **kwargs)
        return self

    def eval(self) -> TurboQuantWrapper:  # noqa: A003
        """Set wrapped model to evaluation mode.

        Returns:
            Self for method chaining.
        """
        self._model.eval()
        return self

    def __repr__(self) -> str:
        return (
            f"TurboQuantWrapper(bit_width={self._bit_width}, "
            f"residual_length={self._residual_length}, "
            f"model={self._model.__class__.__name__})"
        )


def wrap(
    model: Any,
    bit_width: int | None = None,
    residual_length: int = 0,
    n_outlier_channels: int = 0,
    verbose: bool = False,
) -> TurboQuantWrapper:
    """Wrap a HuggingFace model with TurboQuant KV cache compression.

    One-liner API for adding quantized KV caching to any supported model::

        model = wrap(AutoModelForCausalLM.from_pretrained("meta-llama/..."))
        output = model.generate(input_ids, max_new_tokens=100)

    Args:
        model: HuggingFace model with ``.config`` and ``.generate()``.
        bit_width: Bits per coordinate, or None for auto-selection.
        residual_length: Recent tokens kept in fp16 (sliding window).
        n_outlier_channels: Outlier channels for high-precision storage.
        verbose: If True, print compression stats after generate().

    Returns:
        Wrapped model instance.
    """
    return TurboQuantWrapper(
        model,
        bit_width=bit_width,
        residual_length=residual_length,
        n_outlier_channels=n_outlier_channels,
        verbose=verbose,
    )
