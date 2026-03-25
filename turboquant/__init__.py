"""TurboQuant: Near-optimal online vector quantizer.

Unofficial PyTorch reference implementation of TurboQuant (ICLR 2026).
Paper: https://arxiv.org/abs/2504.19874

Two-stage quantizer combining MSE-optimal scalar quantization with
QJL 1-bit correction for unbiased inner product estimation.
"""

from .codebook import LloydMaxCodebook, get_codebook
from .core import TurboQuant, TurboQuantOutput
from .hadamard import RandomizedHadamardTransform, fwht
from .kv_cache import TurboQuantKVCache
from .mse_quantizer import MSEQuantizedOutput, TurboQuantMSE
from .qjl import QJL, QJLOutput, pack_bits, unpack_bits
from .vector_search import TurboQuantIndex

__version__ = "0.1.0"

__all__ = [
    "TurboQuant",
    "TurboQuantOutput",
    "TurboQuantMSE",
    "MSEQuantizedOutput",
    "TurboQuantKVCache",
    "TurboQuantIndex",
    "RandomizedHadamardTransform",
    "fwht",
    "LloydMaxCodebook",
    "get_codebook",
    "QJL",
    "QJLOutput",
    "pack_bits",
    "unpack_bits",
]
