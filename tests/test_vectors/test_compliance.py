"""Tests for src/vectors/compliance.py — independence verification."""

import torch
import pytest


class TestVerifyIndependence:
    def test_independent_vectors_no_issues(self):
        from src.vectors.compliance import verify_independence
        torch.manual_seed(0)
        comply = {l: torch.randn(768).float() for l in range(12)}
        comply = {l: v / v.norm() for l, v in comply.items()}
        refusal = {l: torch.randn(768).float() for l in range(12)}
        refusal = {l: v / v.norm() for l, v in refusal.items()}
        issues = verify_independence(comply, refusal, threshold=0.9)
        # Random vectors should not be correlated
        assert len(issues) == 0

    def test_correlated_vectors_flagged(self):
        from src.vectors.compliance import verify_independence
        # comply = -refusal (perfect anti-correlation → comply is just negated refusal)
        refusal = {0: torch.tensor([1.0, 0.0, 0.0])}
        comply = {0: torch.tensor([-1.0, 0.0, 0.0])}
        issues = verify_independence(comply, refusal, threshold=0.5)
        assert len(issues) == 1
        assert issues[0][0] == 0  # layer 0
        assert abs(issues[0][1]) > 0.5

    def test_custom_threshold(self):
        from src.vectors.compliance import verify_independence
        refusal = {0: torch.tensor([1.0, 0.0, 0.0])}
        comply = {0: torch.tensor([-0.8, 0.6, 0.0])}  # cos with -refusal = 0.8
        issues_low = verify_independence(comply, refusal, threshold=0.5)
        issues_high = verify_independence(comply, refusal, threshold=0.95)
        assert len(issues_low) >= len(issues_high)
