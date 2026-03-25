"""Tests for TurboQuantIndex."""

import pytest
import torch

from turboquant.vector_search import TurboQuantIndex


class TestVectorSearch:
    def test_add_and_search_shape(self):
        """Search returns correct shapes."""
        index = TurboQuantIndex(dim=64, bit_width=3)
        db = torch.randn(100, 64)
        queries = torch.randn(5, 64)
        index.add(db)
        scores, indices = index.search(queries, k=10)
        assert scores.shape == (5, 10)
        assert indices.shape == (5, 10)

    def test_single_query(self):
        """Single query (1D) works and returns 1D results."""
        index = TurboQuantIndex(dim=64, bit_width=3)
        index.add(torch.randn(50, 64))
        scores, indices = index.search(torch.randn(64), k=5)
        assert scores.dim() == 1
        assert scores.shape[0] == 5

    def test_recall_sanity(self):
        """Recall@10 should be > 0.2 on easy synthetic data."""
        torch.manual_seed(42)
        dim = 64
        db = torch.randn(500, dim)
        queries = torch.randn(20, dim)
        index = TurboQuantIndex(dim=dim, bit_width=4)
        index.add(db)
        _, pred_indices = index.search(queries, k=10)

        # True top-1 by exact inner product
        true_scores = queries @ db.t()
        true_top1 = true_scores.argmax(dim=-1)

        recall = sum(
            true_top1[i].item() in pred_indices[i].tolist() for i in range(len(queries))
        ) / len(queries)
        assert recall > 0.2, f"recall@10 = {recall:.2f}, expected > 0.2"

    def test_cosine_metric(self):
        """Cosine metric normalizes vectors."""
        index = TurboQuantIndex(dim=32, bit_width=3, metric="cosine")
        db = torch.randn(30, 32) * 100  # Large norms
        index.add(db)
        scores, _ = index.search(torch.randn(2, 32), k=5)
        # Cosine scores should be in [-1, 1] range (approximately)
        assert scores.max() < 5.0  # Should not be huge

    def test_empty_index_raises(self):
        """Searching empty index should raise."""
        index = TurboQuantIndex(dim=32, bit_width=3)
        with pytest.raises(RuntimeError):
            index.search(torch.randn(32), k=5)

    def test_n_vectors_property(self):
        """n_vectors tracks added vectors."""
        index = TurboQuantIndex(dim=32, bit_width=3)
        assert index.n_vectors == 0
        index.add(torch.randn(42, 32))
        assert index.n_vectors == 42
