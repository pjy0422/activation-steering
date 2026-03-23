"""Tests for src/analysis/logit_lens.py."""

import pytest
import torch


class TestKLMath:
    """CPU tests for KL divergence math."""

    def test_kl_same_distribution_is_zero(self):
        p = torch.softmax(torch.randn(100), dim=-1)
        kl = (p * (p.log() - p.log())).sum().item()
        assert abs(kl) < 1e-6

    def test_kl_is_nonnegative(self):
        p = torch.softmax(torch.randn(100), dim=-1)
        q = torch.softmax(torch.randn(100), dim=-1)
        kl = (p * (p.log() - q.log())).sum().item()
        assert kl >= -1e-6  # KL is non-negative (allow small numerical error)

    def test_kl_different_distributions(self):
        p = torch.softmax(torch.tensor([10.0, 0.0, 0.0]), dim=-1)
        q = torch.softmax(torch.tensor([0.0, 10.0, 0.0]), dim=-1)
        kl = (p * (p.log() - q.log())).sum().item()
        assert kl > 0.1  # Very different distributions → large KL

    def test_rms_norm(self):
        """RMSNorm should normalize the vector."""
        x = torch.randn(768)
        w = torch.ones(768)
        eps = 1e-6
        normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w
        rms = normed.pow(2).mean().sqrt().item()
        assert abs(rms - 1.0) < 0.1  # Approximately unit RMS


@pytest.mark.gpu
class TestLogitLensOnModel:
    def test_kl_returns_n_layers(self, small_model):
        from src.analysis.logit_lens import compute_layerwise_kl
        n = small_model._model.config.num_hidden_layers
        hidden = small_model._model.config.hidden_size
        torch.manual_seed(0)
        vecs = {l: torch.randn(hidden) * 0.1 for l in range(n)}
        kls = compute_layerwise_kl(
            small_model, "Hello world",
            vecs, [n - 1], 5.0,
        )
        assert len(kls) == n
        assert all(k >= 0 for k in kls)
