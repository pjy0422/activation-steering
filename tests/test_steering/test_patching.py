"""Tests for src/steering/patching.py — activation patching."""

import pytest
import torch


class TestPatchingLogic:
    """CPU tests for patching logic concepts."""

    def test_suppress_concept(self):
        """Suppress: replace steered activations with clean ones at critical layer."""
        clean_act = torch.randn(1, 5, 768)
        steered_act = clean_act + torch.randn(1, 5, 768) * 0.1

        # After patching, critical layer should match clean
        patched = steered_act.clone()
        patched[:] = clean_act
        assert torch.allclose(patched, clean_act)

    def test_induce_concept(self):
        """Induce: replace clean activations with steered ones at critical layer."""
        clean_act = torch.randn(1, 5, 768)
        steering_delta = torch.randn(1, 5, 768) * 0.1
        steered_act = clean_act + steering_delta

        # After patching, critical layer should match steered
        patched = clean_act.clone()
        patched[:] = steered_act
        assert torch.allclose(patched, steered_act)


@pytest.mark.gpu
class TestPatchingOnModel:
    def test_patch_suppress_returns_logits(self, small_model):
        """patch_suppress should return logits tensor."""
        from src.steering.patching import patch_suppress
        n = small_model._model.config.num_hidden_layers
        hidden = small_model._model.config.hidden_size
        torch.manual_seed(0)
        vecs = {l: torch.randn(hidden) * 0.01 for l in range(n)}
        logits = patch_suppress(
            small_model, "Hello world",
            vecs, [n - 1], 1.0, critical_layer=n // 2,
        )
        assert logits.dim() >= 1
        assert logits.shape[-1] == small_model._model.config.vocab_size

    def test_patch_induce_returns_logits(self, small_model):
        from src.steering.patching import patch_induce
        n = small_model._model.config.num_hidden_layers
        hidden = small_model._model.config.hidden_size
        torch.manual_seed(1)
        vecs = {l: torch.randn(hidden) * 0.01 for l in range(n)}
        logits = patch_induce(
            small_model, "Hello world",
            vecs, [n - 1], 1.0, critical_layer=n // 2,
        )
        assert logits.dim() >= 1
