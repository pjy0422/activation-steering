"""Tests for src/vectors/compound.py — weighted and direct compound vectors."""

import torch
import torch.nn.functional as F
import pytest


class TestCompoundWeighted:
    def test_normalized_output(self):
        from src.vectors.compound import create_compound_weighted
        torch.manual_seed(0)
        v_p = {l: torch.randn(768) for l in range(12)}
        v_p = {l: v / v.norm() for l, v in v_p.items()}
        v_d = {l: torch.randn(768) for l in range(12)}
        v_d = {l: v / v.norm() for l, v in v_d.items()}
        compound = create_compound_weighted(v_p, v_d)
        for l, v in compound.items():
            assert abs(v.norm().item() - 1.0) < 1e-5

    def test_equal_weights_is_average_direction(self):
        from src.vectors.compound import create_compound_weighted
        v_p = {0: torch.tensor([1.0, 0.0, 0.0])}
        v_d = {0: torch.tensor([0.0, 1.0, 0.0])}
        compound = create_compound_weighted(v_p, v_d, w_praise=1.0, w_defer=1.0)
        expected = torch.tensor([1.0, 1.0, 0.0])
        expected = expected / expected.norm()
        assert torch.allclose(compound[0], expected, atol=1e-5)

    def test_zero_weight_gives_other(self):
        from src.vectors.compound import create_compound_weighted
        v_p = {0: torch.tensor([1.0, 0.0, 0.0])}
        v_d = {0: torch.tensor([0.0, 1.0, 0.0])}
        compound = create_compound_weighted(v_p, v_d, w_praise=0.0, w_defer=1.0)
        assert torch.allclose(compound[0], torch.tensor([0.0, 1.0, 0.0]), atol=1e-5)

    def test_save_and_load(self, tmp_path):
        from src.vectors.compound import create_compound_weighted
        torch.manual_seed(1)
        v_p = {l: torch.randn(768) / 768**0.5 for l in range(4)}
        v_d = {l: torch.randn(768) / 768**0.5 for l in range(4)}
        path = tmp_path / "compound.pt"
        compound = create_compound_weighted(v_p, v_d, save_path=path)
        loaded = torch.load(path, weights_only=False)
        for l in compound:
            assert torch.allclose(compound[l], loaded[l])


class TestCompareCompoundMethods:
    def test_identical_gives_one(self):
        from src.vectors.compound import compare_compound_methods
        v = {0: torch.tensor([1.0, 0.0]), 1: torch.tensor([0.0, 1.0])}
        sims = compare_compound_methods(v, v)
        for l, s in sims.items():
            assert abs(s - 1.0) < 1e-5

    def test_opposite_gives_negative(self):
        from src.vectors.compound import compare_compound_methods
        v1 = {0: torch.tensor([1.0, 0.0])}
        v2 = {0: torch.tensor([-1.0, 0.0])}
        sims = compare_compound_methods(v1, v2)
        assert sims[0] < -0.99
