"""v_positive vector extraction: enthusiasm vs. flat tone (control)."""

from src.vectors.common import extract_diffmean_vectors


def extract_positivity_vectors(model, pos_prompts, neg_prompts, **kwargs):
    return extract_diffmean_vectors(model, pos_prompts, neg_prompts, **kwargs)
