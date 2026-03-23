"""Tensor utility functions."""

import torch
import torch.nn.functional as F


def cosine_sim(a, b):
    """Cosine similarity between two 1-D tensors."""
    return F.cosine_similarity(
        a.float().unsqueeze(0), b.float().unsqueeze(0)
    ).item()


def normalize(v):
    """L2-normalize a vector."""
    n = v.norm()
    if n > 0:
        return v / n
    return v


def project_onto(h, direction):
    """Project vector h onto direction.

    proj_d(h) = (d * d^T / d . d) h = (h . d / d . d) * d
    """
    d = direction.float()
    h = h.float()
    return (torch.dot(h, d) / torch.dot(d, d)) * d
