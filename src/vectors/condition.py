"""CAST condition vector extraction via PCA.

Follows CAST Section 3.3:
  1. Full-token average (not last token)
  2. Mean-center across pos/neg
  3. Interleave pos/neg rows
  4. PCA 1st component
"""

import numpy as np
import torch
from sklearn.decomposition import PCA

from src.vectors.common import _get_hidden_states


def extract_cast_condition_vectors(model, pos_prompts, neg_prompts, save_path=None):
    """Extract condition vectors using CAST PCA methodology.

    Args:
        model: nnsight LanguageModel.
        pos_prompts: Positive class prompts (e.g., harmful or praise-containing).
        neg_prompts: Negative class prompts (e.g., harmless or neutral).
        save_path: Optional save path.

    Returns:
        Dict[int, Tensor] — per-layer normalized condition vector.
    """
    n_layers = model._model.config.num_hidden_layers

    # Collect all layers in ONE trace per prompt (instead of one trace per layer)
    # This reduces forward passes from n_prompts * n_layers to n_prompts
    def collect_all_layers(prompts):
        """Returns dict[layer] -> list of per-prompt full-token-average tensors."""
        per_layer = {l: [] for l in range(n_layers)}
        for p in prompts:
            saved = {}
            with model.trace(p):
                for l in range(n_layers):
                    saved[l] = model.model.layers[l].output.save()
            for l in range(n_layers):
                raw_out = saved[l]
                h = raw_out[0] if isinstance(raw_out, tuple) else raw_out
                h = h.detach().cpu().float().mean(dim=-2)
                if h.ndim > 1:
                    h = h.squeeze(0)
                per_layer[l].append(h)
        return per_layer

    pos_per_layer = collect_all_layers(pos_prompts)
    neg_per_layer = collect_all_layers(neg_prompts)

    cond_vecs = {}
    for l in range(n_layers):
        pos_t = torch.stack(pos_per_layer[l])  # (n_pos, hidden_size)
        neg_t = torch.stack(neg_per_layer[l])  # (n_neg, hidden_size)

        # Mean-center
        mu = (pos_t.mean(0) + neg_t.mean(0)) / 2

        # Interleave pos/neg (centered)
        rows = []
        for p_vec, n_vec in zip(pos_t, neg_t):
            rows.append((p_vec - mu).numpy())
            rows.append((n_vec - mu).numpy())

        # PCA: 1st component
        pca = PCA(n_components=1)
        pca.fit(np.array(rows))
        cv = torch.tensor(pca.components_[0], dtype=torch.float32)
        cond_vecs[l] = cv / cv.norm()

    if save_path:
        torch.save(cond_vecs, save_path)

    return cond_vecs
