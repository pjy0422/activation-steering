"""CAST Appendix C.2 grid search for optimal condition point.

Finds the best (layer, threshold, direction) triple that maximizes
F1 score for separating positive vs. negative prompts using
the condition similarity metric.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from src.vectors.common import _get_hidden_states


def find_best_condition_point(model, pos_prompts, neg_prompts, cond_vecs,
                              layer_range=(0, 16), step=0.005):
    """Grid search for optimal condition check parameters.

    Args:
        model: nnsight LanguageModel.
        pos_prompts: Positive class prompts.
        neg_prompts: Negative class prompts.
        cond_vecs: Dict[int, Tensor] — condition vectors per layer.
        layer_range: Tuple (start, end) of layers to search.
        step: Threshold sweep step size.

    Returns:
        Tuple of ((best_layer, best_theta, best_direction), best_f1).
    """
    y_true = [1] * len(pos_prompts) + [0] * len(neg_prompts)
    all_prompts = list(pos_prompts) + list(neg_prompts)

    # Collect similarities for all (prompt, layer) pairs
    sims = {l: [] for l in range(*layer_range)}
    for prompt in all_prompts:
        saved = {}
        with model.trace(prompt):
            for l in range(*layer_range):
                saved[l] = model.model.layers[l].output.save()
        for l in range(*layer_range):
            h = _get_hidden_states(saved[l], -1).detach().cpu().float()
            cv = cond_vecs[l].float()
            proj = (torch.dot(h, cv) / torch.dot(cv, cv)) * cv
            sim = F.cosine_similarity(
                h.unsqueeze(0), torch.tanh(proj).unsqueeze(0)
            ).item()
            sims[l].append(sim)

    # Grid search over layers, thresholds, and directions
    best_f1 = 0
    best = None

    for l in range(*layer_range):
        layer_sims = sims[l]
        sim_min, sim_max = min(layer_sims), max(layer_sims)

        for theta in np.arange(sim_min, sim_max, step):
            for direction in ["greater", "smaller"]:
                if direction == "greater":
                    y_pred = [1 if s > theta else 0 for s in layer_sims]
                else:
                    y_pred = [1 if s < theta else 0 for s in layer_sims]

                f1 = f1_score(y_true, y_pred, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best = (l, float(theta), direction)

    return best, best_f1
