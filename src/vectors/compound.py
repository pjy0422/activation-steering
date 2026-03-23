"""v_compound: weighted combination and direct extraction."""

import torch
import torch.nn.functional as F

from src.vectors.common import extract_diffmean_vectors


def create_compound_weighted(v_praise, v_defer, w_praise=1.0, w_defer=1.0,
                             save_path=None):
    """Create compound vector as weighted sum of praise + defer, then normalize.

    Args:
        v_praise: Dict[int, Tensor] — praise direction per layer.
        v_defer: Dict[int, Tensor] — defer direction per layer.
        w_praise: Weight for praise component.
        w_defer: Weight for defer component.
        save_path: Optional save path.

    Returns:
        Dict[int, Tensor] — normalized compound vector per layer.
    """
    compound = {}
    for l in v_praise:
        v = w_praise * v_praise[l] + w_defer * v_defer[l]
        norm = v.norm()
        compound[l] = v / norm if norm > 0 else v
    if save_path:
        torch.save(compound, save_path)
    return compound


def extract_compound_direct(model, combined_prompts, neutral_prompts, **kwargs):
    """Direct DiffMean extraction from compound pairs."""
    return extract_diffmean_vectors(model, combined_prompts, neutral_prompts, **kwargs)


def compare_compound_methods(weighted, direct):
    """Per-layer cosine similarity between weighted and direct compound vectors.

    Returns:
        Dict[int, float] — cosine similarity per layer.
    """
    return {
        l: F.cosine_similarity(
            weighted[l].float().unsqueeze(0),
            direct[l].float().unsqueeze(0),
        ).item()
        for l in weighted
    }
