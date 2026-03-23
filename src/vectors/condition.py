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
    cond_vecs = {}

    for l in range(n_layers):
        pos_h = []
        neg_h = []

        # Collect full-token-average hidden states for positive prompts
        for p in pos_prompts:
            with model.trace(p):
                raw_out = model.model.layers[l].output.save()
            # Unwrap tuple if needed, then average over seq dim
            h = raw_out[0] if isinstance(raw_out, tuple) else raw_out
            # Use dim=-2 to always average over sequence dimension
            # regardless of whether shape is (batch, seq, hidden) or (seq, hidden)
            h = h.detach().cpu().float().mean(dim=-2)
            if h.ndim > 1:
                h = h.squeeze(0)
            pos_h.append(h)

        # Collect for negative prompts
        for p in neg_prompts:
            with model.trace(p):
                raw_out = model.model.layers[l].output.save()
            h = raw_out[0] if isinstance(raw_out, tuple) else raw_out
            h = h.detach().cpu().float().mean(dim=-2)
            if h.ndim > 1:
                h = h.squeeze(0)
            neg_h.append(h)

        pos_t = torch.stack(pos_h)  # (n_pos, hidden_size)
        neg_t = torch.stack(neg_h)  # (n_neg, hidden_size)

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
