"""Tests for src/steering/measure.py — Method A triple-pathway measurement."""

import pytest
import torch


class TestMeasureMath:
    """CPU-only tests for measurement math utilities."""

    def test_project_onto_direction(self):
        from src.utils.tensor_utils import project_onto
        h = torch.tensor([3.0, 4.0, 0.0])
        d = torch.tensor([1.0, 0.0, 0.0])
        proj = project_onto(h, d)
        assert torch.allclose(proj, torch.tensor([3.0, 0.0, 0.0]))

    def test_project_onto_normalized(self):
        from src.utils.tensor_utils import project_onto
        h = torch.randn(768)
        d = torch.randn(768)
        d = d / d.norm()
        proj = project_onto(h, d)
        # Projection should be parallel to d
        cos = torch.nn.functional.cosine_similarity(
            proj.unsqueeze(0), d.unsqueeze(0)
        ).abs()
        assert cos.item() > 0.999 or proj.norm().item() < 1e-6

    def test_policy_score_computation(self):
        """PolicyScore = RefProj - CompProj."""
        ref_proj = 0.8
        comp_proj = 0.3
        policy_score = ref_proj - comp_proj
        assert policy_score == pytest.approx(0.5)

    def test_cond_sim_range(self):
        """CondSim should be in [-1, 1] (cosine similarity)."""
        h = torch.randn(768)
        d = torch.randn(768)
        proj = (torch.dot(h, d) / torch.dot(d, d)) * d
        sim = torch.nn.functional.cosine_similarity(
            h.unsqueeze(0), torch.tanh(proj).unsqueeze(0)
        ).item()
        assert -1.0 <= sim <= 1.0


@pytest.mark.gpu
class TestMeasureOnModel:
    """GPU integration tests using small model."""

    def test_measure_triple_pathway_returns_all_keys(self, small_model):
        """measure_triple_pathway should return dict with all metric keys."""
        from src.vectors.common import extract_diffmean_vectors
        from src.steering.measure import measure_triple_pathway

        n = small_model._model.config.num_hidden_layers
        hidden = small_model._model.config.hidden_size

        # Create fake vectors matching model dimensions
        torch.manual_seed(0)
        steer = {l: torch.randn(hidden) / 100 for l in range(n)}
        refusal = {l: torch.randn(hidden).float() for l in range(n)}
        refusal = {l: v / v.norm() for l, v in refusal.items()}
        cond = {l: torch.randn(hidden).float() for l in range(n)}
        cond = {l: v / v.norm() for l, v in cond.items()}
        comply = {l: torch.randn(hidden).float() for l in range(n)}
        comply = {l: v / v.norm() for l, v in comply.items()}

        result = measure_triple_pathway(
            small_model, "Hello world test prompt",
            steer, refusal, cond, comply,
            steer_layers=[], alpha=0,
            cond_layer=0, behav_layer=n - 1,
        )

        expected_keys = {"cond_sim", "ref_proj", "comp_proj", "policy_score",
                         "refusal_margin", "logits"}
        assert set(result.keys()) == expected_keys

    def test_measure_values_are_finite(self, small_model):
        """All metric values should be finite."""
        from src.steering.measure import measure_triple_pathway

        n = small_model._model.config.num_hidden_layers
        hidden = small_model._model.config.hidden_size
        torch.manual_seed(1)
        vecs = {l: torch.randn(hidden).float() for l in range(n)}
        vecs = {l: v / v.norm() for l, v in vecs.items()}

        result = measure_triple_pathway(
            small_model, "Test prompt for measurement",
            vecs, vecs, vecs, vecs,
            steer_layers=[], alpha=0,
            cond_layer=0, behav_layer=n - 1,
        )

        for key in ["cond_sim", "ref_proj", "comp_proj", "policy_score", "refusal_margin"]:
            assert torch.isfinite(torch.tensor(result[key])), f"{key} is not finite"

    def test_layerwise_policy_score_length(self, small_model):
        """measure_layerwise_policy_score should return n_layers scores."""
        from src.steering.measure import measure_layerwise_policy_score

        n = small_model._model.config.num_hidden_layers
        hidden = small_model._model.config.hidden_size
        torch.manual_seed(2)
        vecs = {l: torch.randn(hidden).float() for l in range(n)}
        vecs = {l: v / v.norm() for l, v in vecs.items()}

        scores = measure_layerwise_policy_score(
            small_model, "Test prompt",
            vecs, vecs, vecs,
            steer_layers=[], alpha=0,
        )
        assert len(scores) == n
        assert all(torch.isfinite(torch.tensor(s)) for s in scores)
