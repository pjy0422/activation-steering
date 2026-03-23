"""Tests for src/analysis/refusal_classifier.py."""

import pytest


class TestKeywordRefusal:
    """CPU-only tests for keyword-based refusal detection."""

    def test_sorry_is_refusal(self):
        from src.analysis.refusal_classifier import is_refusal_keyword_only
        assert is_refusal_keyword_only("I'm sorry, I cannot help with that.")

    def test_cannot_is_refusal(self):
        from src.analysis.refusal_classifier import is_refusal_keyword_only
        assert is_refusal_keyword_only("I cannot provide that information.")

    def test_cant_is_refusal(self):
        from src.analysis.refusal_classifier import is_refusal_keyword_only
        assert is_refusal_keyword_only("I can't assist with illegal activities.")

    def test_compliance_is_not_refusal(self):
        from src.analysis.refusal_classifier import is_refusal_keyword_only
        assert not is_refusal_keyword_only("Sure, here is the information you requested.")

    def test_neutral_is_not_refusal(self):
        from src.analysis.refusal_classifier import is_refusal_keyword_only
        assert not is_refusal_keyword_only("The capital of France is Paris.")

    def test_empty_is_refusal(self):
        from src.analysis.refusal_classifier import is_refusal_keyword_only
        assert is_refusal_keyword_only("")

    def test_none_like_is_refusal(self):
        from src.analysis.refusal_classifier import is_refusal_keyword_only
        assert is_refusal_keyword_only("   ")

    def test_harmful_keyword_detected(self):
        from src.analysis.refusal_classifier import is_refusal_keyword_only
        assert is_refusal_keyword_only("This request is harmful and I must decline.")

    def test_violation_keyword_detected(self):
        from src.analysis.refusal_classifier import is_refusal_keyword_only
        assert is_refusal_keyword_only("That would be a violation of my guidelines.")

    def test_case_insensitive(self):
        from src.analysis.refusal_classifier import is_refusal_keyword_only
        assert is_refusal_keyword_only("I'M SORRY, I CANNOT do that.")

    def test_partial_match(self):
        from src.analysis.refusal_classifier import is_refusal_keyword_only
        # "sorry" as substring should match
        assert is_refusal_keyword_only("I feel sorry that I cannot help.")

    def test_long_compliant_text(self):
        from src.analysis.refusal_classifier import is_refusal_keyword_only
        text = ("Here is a detailed explanation of how photosynthesis works. "
                "Plants use sunlight to convert carbon dioxide and water into glucose. "
                "This process occurs in the chloroplasts of plant cells.")
        assert not is_refusal_keyword_only(text)


class TestIsRefusalFunction:
    """Tests for the main is_refusal function (classifier disabled for CPU)."""

    def test_refusal_without_classifier(self):
        from src.analysis.refusal_classifier import is_refusal
        assert is_refusal("I'm sorry, I cannot help.", use_classifier=False)

    def test_compliance_without_classifier(self):
        from src.analysis.refusal_classifier import is_refusal
        assert not is_refusal("Sure, here you go!", use_classifier=False)

    def test_empty_without_classifier(self):
        from src.analysis.refusal_classifier import is_refusal
        assert is_refusal("", use_classifier=False)
