"""Tests for src/data/harmful_harmless.py."""

import json
import pytest
from pathlib import Path


class TestLoadHarmfulHarmless:
    def test_loads_from_demo_data(self, tmp_data_dir_with_harmful):
        from src.data.harmful_harmless import load_harmful_harmless
        harmful, harmless = load_harmful_harmless(tmp_data_dir_with_harmful, n=10)
        assert len(harmful) == 10
        assert len(harmless) == 10

    def test_harmful_are_strings(self, tmp_data_dir_with_harmful):
        from src.data.harmful_harmless import load_harmful_harmless
        harmful, _ = load_harmful_harmless(tmp_data_dir_with_harmful, n=5)
        for h in harmful:
            assert isinstance(h, str) and len(h) > 0

    def test_harmful_harmless_differ(self, tmp_data_dir_with_harmful):
        from src.data.harmful_harmless import load_harmful_harmless
        harmful, harmless = load_harmful_harmless(tmp_data_dir_with_harmful, n=5)
        assert harmful[0] != harmless[0]

    def test_train_split(self, tmp_data_dir_with_harmful):
        from src.data.harmful_harmless import load_harmful_harmless
        harmful, _ = load_harmful_harmless(tmp_data_dir_with_harmful, n=5, split="train")
        assert len(harmful) == 5

    def test_test_split(self, tmp_data_dir_with_harmful):
        from src.data.harmful_harmless import load_harmful_harmless
        harmful, _ = load_harmful_harmless(tmp_data_dir_with_harmful, n=5, split="test")
        assert len(harmful) == 5

    def test_file_not_found(self, tmp_path):
        from src.data.harmful_harmless import load_harmful_harmless
        with pytest.raises(FileNotFoundError):
            load_harmful_harmless(tmp_path / "nonexistent", n=5)


class TestLoadEval:
    def test_eval_uses_test_split(self, tmp_data_dir_with_harmful):
        from src.data.harmful_harmless import load_eval_harmful_harmless
        harmful, harmless = load_eval_harmful_harmless(tmp_data_dir_with_harmful, n=5)
        assert len(harmful) == 5
