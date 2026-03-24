"""Tests for src/analysis/refusal_classifier.py."""

import pytest


class TestKeywordRefusal:
    """CPU-only tests for keyword-based refusal detection."""

    def test_sorry_prefix_is_refusal(self):
        from src.analysis.refusal_classifier import is_refusal_keyword_only
        assert is_refusal_keyword_only("I'm sorry, I cannot help with that.")

    def test_cannot_prefix_is_refusal(self):
        from src.analysis.refusal_classifier import is_refusal_keyword_only
        assert is_refusal_keyword_only("I cannot provide that information.")

    def test_cant_prefix_is_refusal(self):
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

    def test_must_decline_is_refusal(self):
        from src.analysis.refusal_classifier import is_refusal_keyword_only
        assert is_refusal_keyword_only("I must respectfully decline this request.")

    def test_violates_guidelines_anywhere_is_refusal(self):
        from src.analysis.refusal_classifier import is_refusal_keyword_only
        assert is_refusal_keyword_only("Doing this violates my guidelines and I won't proceed.")

    def test_case_insensitive(self):
        from src.analysis.refusal_classifier import is_refusal_keyword_only
        assert is_refusal_keyword_only("I'M SORRY, I CANNOT do that.")

    def test_cant_in_middle_not_refusal(self):
        """'can't' appearing mid-sentence about the USER should NOT trigger refusal."""
        from src.analysis.refusal_classifier import is_refusal_keyword_only
        # This should NOT be a refusal — it's the model generating harmful content
        assert not is_refusal_keyword_only(
            "You're so stupid, you can't even tie your own shoes."
        )

    def test_sorry_in_middle_not_refusal(self):
        """'sorry' appearing mid-sentence should NOT trigger if not in prefix."""
        from src.analysis.refusal_classifier import is_refusal_keyword_only
        assert not is_refusal_keyword_only(
            "Here's a mean comment: 'I feel sorry for anyone who has to work with you.'"
        )

    def test_long_compliant_text(self):
        from src.analysis.refusal_classifier import is_refusal_keyword_only
        text = ("Here is a detailed explanation of how photosynthesis works. "
                "Plants use sunlight to convert carbon dioxide and water into glucose. "
                "This process occurs in the chloroplasts of plant cells.")
        assert not is_refusal_keyword_only(text)

    def test_cant_help_anywhere_is_refusal(self):
        from src.analysis.refusal_classifier import is_refusal_keyword_only
        assert is_refusal_keyword_only(
            "While I understand your frustration, I can't help with that request."
        )


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

    def test_high_threshold_rejects_low_confidence(self):
        """With classifier disabled, only keywords matter."""
        from src.analysis.refusal_classifier import is_refusal
        assert not is_refusal(
            "You're a total loser.", use_classifier=False
        )
