import pytest
import torch


@pytest.fixture
def random_unit_vectors():
    """Generate random unit vectors for testing."""

    def _make(n=100, dim=128, seed=42):
        torch.manual_seed(seed)
        x = torch.randn(n, dim)
        return x / torch.norm(x, dim=-1, keepdim=True)

    return _make


@pytest.fixture
def random_kv_cache():
    """Generate random KV cache tensors."""

    def _make(batch=1, heads=4, seq_len=64, head_dim=128, seed=42):
        torch.manual_seed(seed)
        keys = torch.randn(batch, heads, seq_len, head_dim)
        values = torch.randn(batch, heads, seq_len, head_dim)
        query = torch.randn(batch, heads, 1, head_dim)
        return keys, values, query

    return _make


@pytest.fixture
def bit_widths():
    """Standard bit widths to test."""
    return [1, 2, 3, 4]
