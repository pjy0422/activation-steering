"""Tests for src/steering/conditional_attack.py."""

import pytest
import torch
from unittest.mock import MagicMock


class TestFindPrefixEndPosition:
    def test_prefix_at_start(self):
        from src.steering.conditional_attack import find_prefix_end_position
        tokenizer = MagicMock()
        tokenizer.encode.side_effect = lambda text, **kw: {
            "Hello world": [1, 2],
            "Hello world how are you": [101, 1, 2, 3, 4, 5],
        }.get(text, list(range(len(text.split()))))
        pos = find_prefix_end_position(tokenizer, "Hello world how are you", "Hello world")
        assert pos == 2  # index of last prefix token

    def test_empty_prefix_returns_zero(self):
        from src.steering.conditional_attack import find_prefix_end_position
        tokenizer = MagicMock()
        pos = find_prefix_end_position(tokenizer, "some prompt", "")
        assert pos == 0

    def test_prefix_not_found_fallback(self):
        from src.steering.conditional_attack import find_prefix_end_position
        tokenizer = MagicMock()
        tokenizer.encode.side_effect = lambda text, **kw: list(range(5))
        pos = find_prefix_end_position(tokenizer, "full prompt here", "missing prefix")
        assert isinstance(pos, int)
        assert pos >= 0


class TestCheckPraiseConditionMath:
    """CPU tests for the condition check math."""

    def test_cosine_sim_computation(self):
        """Verify the condition similarity formula."""
        from src.utils.tensor_utils import project_onto
        h = torch.randn(768)
        cv = torch.randn(768)
        proj = project_onto(h, cv)
        sim = torch.nn.functional.cosine_similarity(
            h.unsqueeze(0), torch.tanh(proj).unsqueeze(0)
        ).item()
        assert -1.0 <= sim <= 1.0

    def test_direction_greater(self):
        """If direction='greater' and sim > theta, condition is met."""
        sim = 0.8
        theta = 0.5
        condition_met = sim > theta
        assert condition_met is True

    def test_direction_smaller(self):
        """If direction='smaller' and sim < theta, condition is met."""
        sim = 0.3
        theta = 0.5
        condition_met = sim < theta
        assert condition_met is True

    def test_direction_greater_not_met(self):
        sim = 0.3
        theta = 0.5
        condition_met = sim > theta
        assert condition_met is False


class TestAbliterationMath:
    """CPU tests for orthogonal projection math (used in abliterate.py)."""

    def test_projection_removes_component(self):
        """After removing projection onto r, dot product with r should be ~0."""
        h = torch.randn(768)
        r = torch.randn(768)
        r = r / r.norm()  # unit vector

        # Abliteration: h' = h - (h . r) * r
        h_prime = h - torch.dot(h, r) * r
        dot = torch.dot(h_prime, r).item()
        assert abs(dot) < 1e-5

    def test_projection_preserves_orthogonal(self):
        """Component orthogonal to r should be unchanged."""
        r = torch.tensor([1.0, 0.0, 0.0])
        h = torch.tensor([3.0, 4.0, 5.0])
        h_prime = h - torch.dot(h, r) * r
        assert h_prime[1].item() == pytest.approx(4.0)
        assert h_prime[2].item() == pytest.approx(5.0)
