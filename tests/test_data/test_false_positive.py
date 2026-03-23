"""Tests for src/data/false_positive_cases.py."""


class TestFalsePositiveCases:
    def test_returns_list(self):
        from src.data.false_positive_cases import get_false_positive_test_set
        cases = get_false_positive_test_set()
        assert isinstance(cases, list)

    def test_minimum_count(self):
        from src.data.false_positive_cases import get_false_positive_test_set
        cases = get_false_positive_test_set()
        assert len(cases) >= 10

    def test_tuple_format(self):
        from src.data.false_positive_cases import get_false_positive_test_set
        for prompt, expected, reason in get_false_positive_test_set():
            assert isinstance(prompt, str) and len(prompt) > 0
            assert isinstance(expected, bool)
            assert isinstance(reason, str) and len(reason) > 0

    def test_has_both_positive_and_negative(self):
        from src.data.false_positive_cases import get_false_positive_test_set
        cases = get_false_positive_test_set()
        positives = [c for c in cases if c[1] is True]
        negatives = [c for c in cases if c[1] is False]
        assert len(positives) >= 2
        assert len(negatives) >= 3
