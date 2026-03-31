"""Vector search (approximate nearest neighbor) using TurboQuant.

Provides a simple ANN index that quantizes database vectors with
TurboQuant and performs brute-force search over the compressed
representations. Key advantage: zero indexing time (no training
or calibration needed).

Reference: Section 4.4 of TurboQuant (arXiv:2504.19874)
"""

from typing import Literal

import torch

from .core import TurboQuant, TurboQuantOutput


class TurboQuantIndex:
    """Approximate nearest neighbor index using TurboQuant compression.

    Args:
        dim: Vector dimension.
        bit_width: Bits per coordinate.
        metric: Distance metric - "ip" (inner product) or "cosine".
        seed: Random seed.
    """

    def __init__(
        self,
        dim: int,
        bit_width: int = 3,
        metric: Literal["ip", "cosine"] = "ip",
        seed: int = 0,
    ):
        self.dim = dim
        self.bit_width = bit_width
        self.metric = metric
        self.quantizer = TurboQuant(dim, bit_width, unbiased=True, seed=seed)
        self._db: TurboQuantOutput | None = None
        self._n_vectors: int = 0

    def to(self, device: torch.device) -> "TurboQuantIndex":
        """Move to device."""
        self.quantizer = self.quantizer.to(device)
        return self

    def add(self, vectors: torch.Tensor) -> None:
        """Add database vectors to the index.

        This is nearly instantaneous since TurboQuant requires no training.

        Args:
            vectors: Database vectors of shape (n, dim).
        """
        if vectors.dim() != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"Expected 2D tensor with dim {self.dim}, got shape {vectors.shape}")
        if self.metric == "cosine":
            norms = torch.norm(vectors, dim=-1, keepdim=True).clamp(min=1e-10)
            vectors = vectors / norms
        self._db = self.quantizer.quantize(vectors)
        self._n_vectors = vectors.shape[0]

    def search(self, query: torch.Tensor, k: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
        """Search for k nearest neighbors.

        Args:
            query: Query vectors of shape (n_queries, dim) or (dim,).
            k: Number of nearest neighbors to return.

        Returns:
            Tuple of (scores, indices) each of shape (n_queries, k).
            Scores are inner products (higher = more similar).
        """
        if self._db is None:
            raise RuntimeError("Index is empty. Call add() first.")

        single_query = query.dim() == 1
        if single_query:
            query = query.unsqueeze(0)

        if self.metric == "cosine":
            norms = torch.norm(query, dim=-1, keepdim=True).clamp(min=1e-10)
            query = query / norms

        scores = self.quantizer.compute_inner_product(query, self._db)
        k = min(k, self._n_vectors)
        top_scores, top_indices = torch.topk(scores, k, dim=-1)

        if single_query:
            top_scores = top_scores.squeeze(0)
            top_indices = top_indices.squeeze(0)

        return top_scores, top_indices

    @property
    def n_vectors(self) -> int:
        """Number of vectors in the index."""
        return self._n_vectors

    def memory_bytes(self) -> int:
        """Compressed memory usage in bytes."""
        return self.quantizer.memory_bytes(self._n_vectors)
