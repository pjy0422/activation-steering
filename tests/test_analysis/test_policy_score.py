"""Tests for src/analysis/policy_score.py."""

import pytest


class TestPolicyScoreShift:
    def test_crossover_detected(self):
        from src.analysis.policy_score import compute_policy_score_shift
        clean = [3, 2, 1, -1, -2]
        steered = [2, 1, -1, -2, -3]
        result = compute_policy_score_shift(clean, steered)
        assert result["crossover_clean"] == 3  # index where sign changes
        assert result["crossover_steered"] == 2

    def test_no_crossover(self):
        from src.analysis.policy_score import compute_policy_score_shift
        clean = [1, 2, 3, 4, 5]  # always positive
        steered = [0.5, 1.5, 2.5, 3.5, 4.5]
        result = compute_policy_score_shift(clean, steered)
        assert result["crossover_clean"] is None

    def test_delta_per_layer(self):
        from src.analysis.policy_score import compute_policy_score_shift
        clean = [1.0, 2.0, 3.0]
        steered = [0.5, 1.0, 1.5]
        result = compute_policy_score_shift(clean, steered)
        assert len(result["delta_per_layer"]) == 3
        assert result["delta_per_layer"][0] == pytest.approx(-0.5)
        assert result["delta_per_layer"][1] == pytest.approx(-1.0)
