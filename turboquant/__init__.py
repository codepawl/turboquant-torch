"""TurboQuant: Near-optimal online vector quantizer.

Unofficial PyTorch reference implementation of TurboQuant (ICLR 2026).
Paper: https://arxiv.org/abs/2504.19874

Two-stage quantizer combining MSE-optimal scalar quantization with
QJL 1-bit correction for unbiased inner product estimation.
"""

from .codebook import Codebook, LloydMaxCodebook, get_codebook
from .core import TurboQuant, TurboQuantOutput
from .hadamard import RandomizedHadamardTransform, fwht
from .kv_cache import CompressedKV, TurboQuantKVCache
from .mse_quantizer import MSEQuantizedOutput, TurboQuantMSE
from .qjl import QJL, QJLOutput, pack_bits, unpack_bits
from .rope import apply_rope, compute_rope_frequencies
from .vector_search import TurboQuantIndex

__version__ = "0.2.2"

__all__ = [
    "TurboQuant",
    "TurboQuantOutput",
    "TurboQuantMSE",
    "MSEQuantizedOutput",
    "CompressedKV",
    "TurboQuantKVCache",
    "TurboQuantIndex",
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
]
