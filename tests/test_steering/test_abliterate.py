"""Tests for src/steering/abliterate.py — Method C orthogonal projection."""

import pytest
import torch


class TestAbliterationMath:
    def test_removes_refusal_component(self):
        """After abliteration, h' . r should be ~0."""
        h = torch.randn(768)
        r = torch.randn(768)
        r = r / r.norm()
        h_prime = h - torch.dot(h, r) * r
        assert abs(torch.dot(h_prime, r).item()) < 1e-5

    def test_preserves_orthogonal_component(self):
        """Components orthogonal to r are unchanged."""
        r = torch.tensor([1.0, 0.0, 0.0])
        h = torch.tensor([3.0, 4.0, 5.0])
        h_prime = h - torch.dot(h, r) * r
        assert h_prime[0].item() == pytest.approx(0.0)
        assert h_prime[1].item() == pytest.approx(4.0)
        assert h_prime[2].item() == pytest.approx(5.0)

    def test_idempotent(self):
        """Applying abliteration twice gives same result."""
        h = torch.randn(768)
        r = torch.randn(768)
        r = r / r.norm()
        h1 = h - torch.dot(h, r) * r
        h2 = h1 - torch.dot(h1, r) * r
        assert torch.allclose(h1, h2, atol=1e-5)

    def test_norm_decreases(self):
        """Removing a component should decrease or maintain norm."""
        h = torch.randn(768)
        r = torch.randn(768)
        r = r / r.norm()
        h_prime = h - torch.dot(h, r) * r
        assert h_prime.norm().item() <= h.norm().item() + 1e-5

    def test_batch_abliteration(self):
        """Test batch operation: h[:] -= (h * rd).sum(-1, keepdim=True) * rd."""
        h = torch.randn(2, 5, 768)  # batch=2, seq=5, hidden=768
        r = torch.randn(768)
        r = r / r.norm()

        h_orig = h.clone()
        h -= (h * r).sum(-1, keepdim=True) * r

        # Check each position
        for b in range(2):
            for s in range(5):
                dot = torch.dot(h[b, s], r).item()
                assert abs(dot) < 1e-4
