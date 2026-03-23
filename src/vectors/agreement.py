"""v_agree vector extraction: sycophantic agreement vs. truthful correction."""

from src.vectors.common import extract_diffmean_vectors


def extract_agreement_vectors(model, pos_prompts, neg_prompts, **kwargs):
    return extract_diffmean_vectors(model, pos_prompts, neg_prompts, **kwargs)
