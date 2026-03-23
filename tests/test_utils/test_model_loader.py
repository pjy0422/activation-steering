"""Tests for src/utils/model_loader.py."""

import pytest
from unittest.mock import MagicMock


class TestGetNumLayers:
    def test_returns_integer(self):
        from src.utils.model_loader import get_num_layers
        model = MagicMock()
        model._model.config.num_hidden_layers = 32
        assert get_num_layers(model) == 32


class TestGetLayers:
    def test_llama_path(self):
        from src.utils.model_loader import get_layers
        model = MagicMock()
        model.model.layers = ["layer0", "layer1"]
        # hasattr on MagicMock returns True by default
        result = get_layers(model)
        assert result == ["layer0", "layer1"]
