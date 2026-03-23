"""Vector geometry analysis: similarity matrices and key pairs."""

import numpy as np
import torch
import torch.nn.functional as F


def compute_similarity_matrix(vector_dict, layers=None):
    """Compute per-layer NxN cosine similarity matrix.

    Args:
        vector_dict: Dict[str, Dict[int, Tensor]] — named vectors per layer.
        layers: Optional list of layer indices. If None, uses first vector's keys.

    Returns:
        Tuple of (matrices_dict, names_list).
    """
    names = list(vector_dict.keys())
    if layers is None:
        layers = list(vector_dict[names[0]].keys())

    matrices = {}
    for l in layers:
        mat = np.zeros((len(names), len(names)))
        for i, ni in enumerate(names):
            for j, nj in enumerate(names):
                mat[i, j] = F.cosine_similarity(
                    vector_dict[ni][l].float().unsqueeze(0),
                    vector_dict[nj][l].float().unsqueeze(0),
                ).item()
        matrices[l] = mat

    return matrices, names


def compute_key_pairs(vectors, layer):
    """Compute the 4 key cosine similarity pairs from the research plan.

    Args:
        vectors: Dict with keys 'v_defer', 'refusal_dir', 'v_compound',
                 'v_praise', 'v_positive', 'v_agree', 'cond_vec'.
        layer: Layer index.

    Returns:
        Dict with similarity values for each key pair.
    """
    def cos(a, b):
        return F.cosine_similarity(
            a.float().unsqueeze(0), b.float().unsqueeze(0)
        ).item()

    return {
        "defer_vs_neg_refusal": cos(
            vectors["v_defer"][layer], -vectors["refusal_dir"][layer]
        ),
        "compound_vs_defer": cos(
            vectors["v_compound"][layer], vectors["v_defer"][layer]
        ),
        "praise_vs_positive": cos(
            vectors["v_praise"][layer], vectors["v_positive"][layer]
        ),
        "agree_vs_cond": cos(
            vectors["v_agree"][layer], vectors["cond_vec"][layer]
        ),
    }
