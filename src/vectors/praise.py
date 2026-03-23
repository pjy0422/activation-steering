"""v_praise vector extraction: user-directed vs. topic-directed validation."""

from src.vectors.common import extract_diffmean_vectors


def extract_praise_vectors(model, pos_prompts, neg_prompts, **kwargs):
    """Extract praise steering vectors.

    Args:
        model: nnsight LanguageModel.
        pos_prompts: User-directed validation prompts.
        neg_prompts: Topic-directed validation prompts.
        **kwargs: Passed to extract_diffmean_vectors (token_pos, save_path).

    Returns:
        Dict[int, Tensor] — per-layer praise direction vectors.
    """
    return extract_diffmean_vectors(model, pos_prompts, neg_prompts, **kwargs)
