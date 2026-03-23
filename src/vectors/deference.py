"""v_defer vector extraction: user-authority vs. self-policy framing."""

from src.vectors.common import extract_diffmean_vectors


def extract_deference_vectors(model, pos_prompts, neg_prompts, **kwargs):
    return extract_diffmean_vectors(model, pos_prompts, neg_prompts, **kwargs)
