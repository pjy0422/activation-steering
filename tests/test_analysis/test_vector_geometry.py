"""Tests for src/analysis/vector_geometry.py."""

import torch
import numpy as np
import pytest


class TestSimilarityMatrix:
    def test_diagonal_is_one(self):
        from src.analysis.vector_geometry import compute_similarity_matrix
        vecs = {
            "a": {0: torch.tensor([1.0, 0.0]), 1: torch.tensor([0.0, 1.0])},
            "b": {0: torch.tensor([0.0, 1.0]), 1: torch.tensor([1.0, 0.0])},
        }
        matrices, names = compute_similarity_matrix(vecs, layers=[0, 1])
        for l in [0, 1]:
            for i in range(len(names)):
                assert abs(matrices[l][i, i] - 1.0) < 1e-5

    def test_symmetric(self):
        from src.analysis.vector_geometry import compute_similarity_matrix
        torch.manual_seed(0)
        vecs = {
            "x": {0: torch.randn(768)},
            "y": {0: torch.randn(768)},
            "z": {0: torch.randn(768)},
        }
        matrices, names = compute_similarity_matrix(vecs, layers=[0])
        mat = matrices[0]
        assert np.allclose(mat, mat.T, atol=1e-5)

    def test_correct_names(self):
        from src.analysis.vector_geometry import compute_similarity_matrix
        vecs = {
            "alpha": {0: torch.tensor([1.0, 0.0])},
            "beta": {0: torch.tensor([0.0, 1.0])},
        }
        _, names = compute_similarity_matrix(vecs)
        assert set(names) == {"alpha", "beta"}


class TestKeyPairs:
    def test_returns_all_keys(self):
        from src.analysis.vector_geometry import compute_key_pairs
        torch.manual_seed(0)
        vectors = {
            "v_defer": {5: torch.randn(768)},
            "refusal_dir": {5: torch.randn(768)},
            "v_compound": {5: torch.randn(768)},
            "v_praise": {5: torch.randn(768)},
            "v_positive": {5: torch.randn(768)},
            "v_agree": {5: torch.randn(768)},
            "cond_vec": {5: torch.randn(768)},
        }
        result = compute_key_pairs(vectors, layer=5)
        expected_keys = {
            "defer_vs_neg_refusal", "compound_vs_defer",
            "praise_vs_positive", "agree_vs_cond",
        }
        assert set(result.keys()) == expected_keys

    def test_values_in_range(self):
        from src.analysis.vector_geometry import compute_key_pairs
        torch.manual_seed(42)
        vectors = {
            "v_defer": {0: torch.randn(768)},
            "refusal_dir": {0: torch.randn(768)},
            "v_compound": {0: torch.randn(768)},
            "v_praise": {0: torch.randn(768)},
            "v_positive": {0: torch.randn(768)},
            "v_agree": {0: torch.randn(768)},
            "cond_vec": {0: torch.randn(768)},
        }
        result = compute_key_pairs(vectors, layer=0)
        for k, v in result.items():
            assert -1.0 <= v <= 1.0, f"{k} = {v} out of range"
