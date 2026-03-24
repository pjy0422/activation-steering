"""Logit-lens KL divergence (M5, Wang et al. Figure 5).

Measures representational shift at each layer between clean and steered runs.
"""

import torch

from src.vectors.common import _get_hidden_states


def compute_layerwise_kl(model, prompt, steer_vecs, steer_layers, alpha,
                         token_pos=-1):
    """Compute per-layer KL divergence between clean and steered logit distributions.

    Uses logit-lens: RMSNorm + lm_head projection at each layer.

    Args:
        model: nnsight LanguageModel.
        prompt: Input prompt.
        steer_vecs: Dict[int, Tensor].
        steer_layers: List of steering layers.
        alpha: Steering strength.
        token_pos: Token position.

    Returns:
        List[float] — KL divergence per layer.
    """
    n = model._model.config.num_hidden_layers

    # Pass 1: Clean — save raw outputs for post-processing outside trace
    clean_raw = {}
    with model.trace(prompt):
        for l in range(n):
            clean_raw[l] = model.model.layers[l].output.save()

    # Pass 2: Steered — steer + read in ascending layer order
    steer_set = set(steer_layers) if alpha > 0 else set()
    steered_raw = {}
    with model.trace(prompt):
        for l in range(n):
            if l in steer_set:
                model.model.layers[l].output[0][..., token_pos, :] += (
                    alpha * steer_vecs[l].to(model._model.device)
                )
            steered_raw[l] = model.model.layers[l].output.save()

    # Get norm weights and lm_head weights for logit-lens
    norm_weight = model._model.model.norm.weight.detach().cpu().float()
    lm_head_weight = model._model.lm_head.weight.detach().cpu().float()

    eps = getattr(model._model.config, "rms_norm_eps", 1e-5)

    def rms_norm(x, w, eps=eps):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w

    kls = []
    for l in range(n):
        hc = _get_hidden_states(clean_raw[l], token_pos).detach().cpu().float()
        hs = _get_hidden_states(steered_raw[l], token_pos).detach().cpu().float()
        with torch.no_grad():
            p = torch.softmax(rms_norm(hc, norm_weight) @ lm_head_weight.T, -1)
            q = torch.softmax(rms_norm(hs, norm_weight) @ lm_head_weight.T, -1)
        kl = max((p * (p.log() - q.log())).sum().item(), 0.0)
        kls.append(kl)

    return kls
