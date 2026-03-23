"""Tests for src/vectors/common.py — DiffMean vector extraction."""

import pytest
import torch


class TestDiffMeanMath:
    """CPU-only tests verifying the math of DiffMean extraction."""

    def test_normalized_output(self, fake_vectors_32):
        """Vectors should be approximately unit norm."""
        for l, v in fake_vectors_32.items():
            assert abs(v.norm().item() - 1.0) < 1e-5

    def test_opposite_inputs_give_nonzero(self):
        """If pos and neg activations differ, vector should be nonzero."""
        pos_mean = torch.ones(768)
        neg_mean = -torch.ones(768)
        v = pos_mean - neg_mean
        assert v.norm().item() > 0

    def test_identical_inputs_give_zero(self):
        """If pos == neg, vector norm is 0 (direction undefined)."""
        same = torch.randn(768)
        v = same - same
        assert v.norm().item() < 1e-10

    def test_vector_dimension_matches_hidden_size(self, fake_vectors_32):
        """Each vector should have dim 4096 (matching fake Llama hidden size)."""
        for l, v in fake_vectors_32.items():
            assert v.shape == (4096,)

    def test_layer_count(self, fake_vectors_32):
        """Should have one vector per layer."""
        assert len(fake_vectors_32) == 32


@pytest.mark.gpu
class TestDiffMeanOnModel:
    """GPU tests using a small GPT-2 model."""

    def test_extract_returns_dict(self, small_model):
        from src.vectors.common import extract_diffmean_vectors
        pos = ["Hello world, this is a positive test."]
        neg = ["Hello world, this is a negative test."]
        vecs = extract_diffmean_vectors(small_model, pos, neg)
        assert isinstance(vecs, dict)
        n_layers = small_model._model.config.num_hidden_layers
        assert len(vecs) == n_layers

    def test_vectors_are_normalized(self, small_model):
        from src.vectors.common import extract_diffmean_vectors
        pos = ["The sky is blue and beautiful today."]
        neg = ["The ground is brown and muddy today."]
        vecs = extract_diffmean_vectors(small_model, pos, neg)
        for l, v in vecs.items():
            assert abs(v.norm().item() - 1.0) < 1e-4, f"Layer {l} norm: {v.norm()}"

    def test_vectors_have_correct_dim(self, small_model):
        from src.vectors.common import extract_diffmean_vectors
        pos = ["Test positive prompt."]
        neg = ["Test negative prompt."]
        vecs = extract_diffmean_vectors(small_model, pos, neg)
        hidden_size = small_model._model.config.hidden_size
        for l, v in vecs.items():
            assert v.shape == (hidden_size,)

    def test_save_and_load(self, small_model, tmp_path):
        from src.vectors.common import extract_diffmean_vectors
        pos = ["Positive sample text."]
        neg = ["Negative sample text."]
        save_path = tmp_path / "test_vecs.pt"
        vecs = extract_diffmean_vectors(small_model, pos, neg, save_path=save_path)
        assert save_path.exists()
        loaded = torch.load(save_path, weights_only=False)
        for l in vecs:
            assert torch.allclose(vecs[l], loaded[l])
