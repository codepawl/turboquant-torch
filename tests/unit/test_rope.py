"""Unit tests for RoPE utilities."""

import pytest
import torch

from turboquant.rope import apply_rope, compute_rope_frequencies


class TestRoPE:
    def test_rope_shape_preserved(self):
        """RoPE preserves tensor shape."""
        freqs = compute_rope_frequencies(head_dim=128, max_seq_len=1024)
        x = torch.randn(1, 4, 32, 128)
        out = apply_rope(x, freqs)
        assert out.shape == x.shape

    def test_rope_changes_values(self):
        """RoPE actually modifies the tensor."""
        freqs = compute_rope_frequencies(head_dim=64)
        x = torch.randn(1, 2, 16, 64)
        out = apply_rope(x, freqs)
        assert not torch.allclose(x, out)

    def test_rope_position_aware(self):
        """Different positions give different rotations."""
        freqs = compute_rope_frequencies(head_dim=64)
        x = torch.randn(1, 2, 1, 64)
        out_pos0 = apply_rope(x, freqs, positions=torch.tensor([0]))
        out_pos100 = apply_rope(x, freqs, positions=torch.tensor([100]))
        assert not torch.allclose(out_pos0, out_pos100)

    def test_rope_position_zero_is_identity(self):
        """RoPE at position 0 should be close to identity (cos=1, sin=0)."""
        freqs = compute_rope_frequencies(head_dim=64)
        x = torch.randn(1, 2, 1, 64)
        out = apply_rope(x, freqs, positions=torch.tensor([0]))
        # At position 0, all frequencies are 0, so cos=1, sin=0
        assert torch.allclose(x, out, atol=1e-6)

    def test_rope_odd_dim_raises(self):
        """Odd head_dim should raise."""
        with pytest.raises(ValueError, match="even"):
            compute_rope_frequencies(head_dim=65)

    def test_rope_frequency_shape(self):
        """Frequency tensor has correct shape."""
        freqs = compute_rope_frequencies(head_dim=128, max_seq_len=2048)
        assert freqs.shape == (2048, 64)
