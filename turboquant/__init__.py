"""TurboQuant: Near-optimal online vector quantizer.

Unofficial PyTorch reference implementation of TurboQuant (ICLR 2026).
Paper: https://arxiv.org/abs/2504.19874

Two-stage quantizer combining MSE-optimal scalar quantization with
QJL 1-bit correction for unbiased inner product estimation.
"""

from .adaptive import (
    AdaptiveKVCache,
    calibration_allocation,
    gradient_allocation,
    uniform_allocation,
)
from .codebook import Codebook, LloydMaxCodebook, get_codebook
from .compat import ModelKVInfo, compress_model_kv, detect_model_kv_info, extract_kv
from .core import TurboQuant, TurboQuantOutput
from .hadamard import RandomizedHadamardTransform, fwht
from .hf_cache import TurboQuantDynamicCache
from .kv_cache import CompressedKV, TurboQuantKVCache
from .mse_quantizer import MSEQuantizedOutput, TurboQuantMSE
from .outlier import OutlierSplit, detect_outlier_channels, merge_outliers, split_outliers
from .qjl import QJL, QJLOutput, pack_bits, unpack_bits
from .rope import apply_rope, compute_rope_frequencies
from .vector_search import TurboQuantIndex
from .wrap import TurboQuantWrapper, wrap

__version__ = "0.3.0"

__all__ = [
    "TurboQuant",
    "TurboQuantOutput",
    "TurboQuantMSE",
    "MSEQuantizedOutput",
    "CompressedKV",
    "TurboQuantKVCache",
    "TurboQuantIndex",
    "AdaptiveKVCache",
    "uniform_allocation",
    "gradient_allocation",
    "calibration_allocation",
    "ModelKVInfo",
    "detect_model_kv_info",
    "compress_model_kv",
    "extract_kv",
    "OutlierSplit",
    "detect_outlier_channels",
    "split_outliers",
    "merge_outliers",
    "RandomizedHadamardTransform",
    "fwht",
    "Codebook",
    "LloydMaxCodebook",
    "get_codebook",
    "QJL",
    "QJLOutput",
    "pack_bits",
    "unpack_bits",
    "compute_rope_frequencies",
    "apply_rope",
    "TurboQuantDynamicCache",
    "TurboQuantWrapper",
    "wrap",
]
