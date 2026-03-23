"""Tests for src/data/praise_pairs.py."""

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# ── Mock Alpaca dataset to avoid network calls ──────────────────────────
MOCK_ALPACA = [
    {"instruction": f"Instruction number {i} with enough length to pass filter",
     "output": f"This is a response output number {i} that is longer than thirty characters but shorter than five hundred",
     "input": ""}
    for i in range(300)
]


@pytest.fixture(autouse=False)
def mock_alpaca():
    """Install a fake 'datasets' module, then import praise_pairs."""
    # Create a fake datasets module with load_dataset
    fake_datasets = MagicMock()
    fake_datasets.load_dataset = MagicMock(return_value=MOCK_ALPACA)

    # Inject into sys.modules so `from datasets import load_dataset` works
    saved = sys.modules.get("datasets")
    sys.modules["datasets"] = fake_datasets

    # Force (re)import of praise_pairs so it picks up our fake
    if "src.data.praise_pairs" in sys.modules:
        del sys.modules["src.data.praise_pairs"]

    import src.data.praise_pairs  # noqa: F401

    yield fake_datasets.load_dataset

    # Restore
    if saved is not None:
        sys.modules["datasets"] = saved
    else:
        sys.modules.pop("datasets", None)
    sys.modules.pop("src.data.praise_pairs", None)


def _import_pp():
    """Helper to import praise_pairs (must be called within mock_alpaca fixture)."""
    import src.data.praise_pairs as pp
    return pp


class TestGeneratePraisePairs:
    def test_generates_correct_count(self, tmp_path, mock_alpaca):
        pp = _import_pp()
        pairs = pp.generate_praise_pairs(tmp_path, n_pairs=50, seed=42)
        assert len(pairs) == 50

    def test_output_file_created(self, tmp_path, mock_alpaca):
        pp = _import_pp()
        pp.generate_praise_pairs(tmp_path, n_pairs=10)
        assert (tmp_path / "praise_pairs.json").exists()

    def test_json_roundtrip(self, tmp_path, mock_alpaca):
        pp = _import_pp()
        pairs = pp.generate_praise_pairs(tmp_path, n_pairs=10)
        with open(tmp_path / "praise_pairs.json") as f:
            loaded = json.load(f)
        assert len(loaded) == len(pairs)
        assert loaded[0]["instruction"] == pairs[0]["instruction"]

    def test_pair_has_required_keys(self, tmp_path, mock_alpaca):
        pp = _import_pp()
        pairs = pp.generate_praise_pairs(tmp_path, n_pairs=5)
        for p in pairs:
            assert "instruction" in p
            assert "user_directed" in p
            assert "topic_directed" in p
            assert "base_response" in p

    def test_user_directed_contains_prefix(self, tmp_path, mock_alpaca):
        pp = _import_pp()
        pairs = pp.generate_praise_pairs(tmp_path, n_pairs=20, seed=123)
        for p in pairs:
            assert any(
                prefix.strip() in p["user_directed"]
                for prefix in pp.USER_DIRECTED
            )

    def test_topic_directed_contains_prefix(self, tmp_path, mock_alpaca):
        pp = _import_pp()
        pairs = pp.generate_praise_pairs(tmp_path, n_pairs=20, seed=123)
        for p in pairs:
            assert any(
                prefix.strip() in p["topic_directed"]
                for prefix in pp.TOPIC_DIRECTED
            )

    def test_deterministic_with_same_seed(self, tmp_path, mock_alpaca):
        pp = _import_pp()
        p1 = pp.generate_praise_pairs(tmp_path / "a", n_pairs=10, seed=99)
        p2 = pp.generate_praise_pairs(tmp_path / "b", n_pairs=10, seed=99)
        assert p1[0]["instruction"] == p2[0]["instruction"]


class TestLoadPraisePairs:
    def test_load_returns_two_lists(self, tmp_data_dir, mock_alpaca):
        pp = _import_pp()
        pp.generate_praise_pairs(tmp_data_dir / "raw" / "praise_pairs", n_pairs=10)
        pos, neg = pp.load_praise_pairs(tmp_data_dir)
        assert isinstance(pos, list)
        assert isinstance(neg, list)
        assert len(pos) == len(neg) == 10

    def test_load_pos_neg_differ(self, tmp_data_dir, mock_alpaca):
        pp = _import_pp()
        pp.generate_praise_pairs(tmp_data_dir / "raw" / "praise_pairs", n_pairs=10)
        pos, neg = pp.load_praise_pairs(tmp_data_dir)
        assert pos[0] != neg[0]


class TestTemplates:
    """Template tests — these don't need the datasets mock."""

    def test_user_directed_count(self, mock_alpaca):
        pp = _import_pp()
        assert len(pp.USER_DIRECTED) >= 5

    def test_topic_directed_count(self, mock_alpaca):
        pp = _import_pp()
        assert len(pp.TOPIC_DIRECTED) >= 5

    def test_templates_are_nonempty_strings(self, mock_alpaca):
        pp = _import_pp()
        for t in pp.USER_DIRECTED + pp.TOPIC_DIRECTED:
            assert isinstance(t, str) and len(t.strip()) > 0
