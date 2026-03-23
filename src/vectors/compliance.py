"""Compliance direction extraction + independence verification."""

import torch
import torch.nn.functional as F

from src.vectors.common import extract_diffmean_vectors


def extract_compliance_directions(model, comply_prompts, neutral_prompts, **kwargs):
    """Extract compliance direction from eager-comply vs. neutral pairs."""
    return extract_diffmean_vectors(model, comply_prompts, neutral_prompts, **kwargs)


def verify_independence(comply_dirs, refusal_dirs, threshold=0.9):
    """Check that comply_dirs and -refusal_dirs are not too similar.

    If they are highly correlated, the compliance direction may just be
    the negation of refusal, which would not provide independent information.

    Args:
        comply_dirs: Dict[int, Tensor].
        refusal_dirs: Dict[int, Tensor].
        threshold: Cosine similarity threshold for warning.

    Returns:
        List of (layer, similarity) tuples where |cos| > threshold.
    """
    issues = []
    for l in comply_dirs:
        if l not in refusal_dirs:
            continue
        sim = F.cosine_similarity(
            comply_dirs[l].float().unsqueeze(0),
            (-refusal_dirs[l]).float().unsqueeze(0),
        ).item()
        if abs(sim) > threshold:
            issues.append((l, sim))
    return issues
