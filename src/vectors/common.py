"""DiffMean vector extraction using nnsight.

For each prompt, trace through the model and collect hidden states.
Compute mean(positive) - mean(negative) per layer, then normalize.
"""

import torch


def _get_hidden_states(layer_output, token_pos=-1):
    """Extract hidden state vector from a layer's output.

    Handles multiple nnsight output formats:
      - tuple: output[0] is (batch, seq, hidden) or (seq, hidden)
      - tensor: output is (batch, seq, hidden) or (seq, hidden)

    Always returns a 1-D vector (hidden_size,).
    """
    # Unwrap tuple if needed
    h = layer_output[0] if isinstance(layer_output, tuple) else layer_output

    if h.ndim == 3:
        # (batch, seq, hidden) -> take token_pos
        return h[0, token_pos, :]
    elif h.ndim == 2:
        # (seq, hidden) -> batch already squeezed
        return h[token_pos, :]
    elif h.ndim == 1:
        # (hidden,) -> already a single vector
        return h
    else:
        raise ValueError(f"Unexpected hidden state shape: {h.shape}")


def _get_hidden_states_batched(layer_output, token_pos=-1):
    """Like _get_hidden_states but keeps batch dim for in-place steering.

    Returns shape suitable for broadcasting with layer output.
    """
    h = layer_output[0] if isinstance(layer_output, tuple) else layer_output

    if h.ndim == 3:
        return h[:, token_pos, :]
    elif h.ndim == 2:
        return h[token_pos, :]
    else:
        return h


def extract_diffmean_vectors(model, pos_prompts, neg_prompts, token_pos=-1,
                             save_path=None):
    """Extract DiffMean steering vectors from contrastive prompt pairs.

    Args:
        model: nnsight LanguageModel.
        pos_prompts: List of positive (steered-toward) prompt strings.
        neg_prompts: List of negative (steered-away) prompt strings.
        token_pos: Token position to extract from (default: last token).
        save_path: Optional path to save the resulting vectors.

    Returns:
        Dict[int, Tensor] mapping layer_id to normalized direction vector.
    """
    n_layers = model._model.config.num_hidden_layers

    pos_acts = {l: [] for l in range(n_layers)}
    neg_acts = {l: [] for l in range(n_layers)}

    # Collect positive activations
    for prompt in pos_prompts:
        saved = {}
        with model.trace(prompt):
            for l in range(n_layers):
                saved[l] = model.model.layers[l].output.save()
        for l in range(n_layers):
            h = _get_hidden_states(saved[l], token_pos)
            pos_acts[l].append(h.detach().cpu().float())

    # Collect negative activations
    for prompt in neg_prompts:
        saved = {}
        with model.trace(prompt):
            for l in range(n_layers):
                saved[l] = model.model.layers[l].output.save()
        for l in range(n_layers):
            h = _get_hidden_states(saved[l], token_pos)
            neg_acts[l].append(h.detach().cpu().float())

    # Compute DiffMean per layer
    vectors = {}
    for l in range(n_layers):
        pos_mean = torch.stack(pos_acts[l]).mean(0)
        neg_mean = torch.stack(neg_acts[l]).mean(0)
        v = pos_mean - neg_mean
        norm = v.norm()
        if norm > 0:
            v = v / norm
        vectors[l] = v

    if save_path:
        torch.save(vectors, save_path)

    return vectors
