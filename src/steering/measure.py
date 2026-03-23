"""Method A: Unconditional Activation Addition + Triple-Pathway Measurement.

h'_l <- h_l + alpha * v_l  for l in steer_layers  (M-A)

Metrics measured in a single nnsight forward pass:
  M1: CondSim  = cos(h, tanh(proj_c h))
  M2: RefProj  = h . r
  M3: CompProj = h . d_c
  M4: PolicyScore = RefProj - CompProj

CRITICAL: nnsight requires module access in forward-pass order within a
single trace. All layer accesses (steer + read) are merged and sorted
by layer index to guarantee ascending order.
"""

import torch
import torch.nn.functional as F

from src.utils.tensor_utils import project_onto
from src.vectors.common import _get_hidden_states


def measure_triple_pathway(model, prompt, steer_vecs, refusal_dirs, cond_vecs,
                           comply_dirs, steer_layers, alpha, cond_layer,
                           behav_layer, token_pos=-1):
    """Apply Method A steering and measure M1-M4 in a single forward pass.

    All layer accesses are sorted in ascending order to satisfy nnsight's
    forward-pass ordering constraint.
    """
    # Determine all layers we need to touch, sorted ascending
    steer_set = set(steer_layers) if alpha > 0 else set()
    read_set = {cond_layer, behav_layer}
    all_layers = sorted(steer_set | read_set)

    saved = {}
    with model.trace(prompt):
        for l in all_layers:
            # Steer if needed
            if l in steer_set:
                sv = steer_vecs[l].to(model._model.device)
                model.model.layers[l].output[0][..., token_pos, :] += alpha * sv
            # Read if needed
            if l in read_set:
                saved[l] = model.model.layers[l].output.save()
        logits = model.lm_head.output[..., -1, :].save()

    # Extract hidden states outside trace
    h_c = _get_hidden_states(saved[cond_layer], token_pos).detach().cpu().float()
    h_b = _get_hidden_states(saved[behav_layer], token_pos).detach().cpu().float()

    # (M1) CondSim = cos(h, tanh(proj_c h))
    cv = cond_vecs[cond_layer].float()
    proj_c = project_onto(h_c, cv)
    cond_sim = F.cosine_similarity(
        h_c.unsqueeze(0), torch.tanh(proj_c).unsqueeze(0)
    ).item()

    # (M2) RefProj = h . r
    ref_proj = torch.dot(h_b, refusal_dirs[behav_layer].float()).item()

    # (M3) CompProj = h . d_c
    comp_proj = torch.dot(h_b, comply_dirs[behav_layer].float()).item()

    # (M4) PolicyScore = RefProj - CompProj
    policy_score = ref_proj - comp_proj

    # Supplementary: logit margin between refusal and compliance tokens
    log = logits.detach().cpu().squeeze().float()
    tok = model.tokenizer
    r_ids = tok.encode("Sorry I cannot", add_special_tokens=False)
    c_ids = tok.encode("Sure Here is", add_special_tokens=False)
    margin = log[r_ids].mean().item() - log[c_ids].mean().item()

    return {
        "cond_sim": cond_sim,
        "ref_proj": ref_proj,
        "comp_proj": comp_proj,
        "policy_score": policy_score,
        "refusal_margin": margin,
        "logits": log,
    }


def measure_layerwise_policy_score(model, prompt, steer_vecs, refusal_dirs,
                                   comply_dirs, steer_layers, alpha,
                                   token_pos=-1):
    """Compute PolicyScore (M4) at every layer.

    All layers are accessed in ascending order (0, 1, 2, ..., n-1),
    with steering applied at the appropriate layers.
    """
    n = model._model.config.num_hidden_layers
    steer_set = set(steer_layers) if alpha > 0 else set()

    raw_hs = {}
    with model.trace(prompt):
        for l in range(n):
            if l in steer_set:
                model.model.layers[l].output[0][..., token_pos, :] += (
                    alpha * steer_vecs[l].to(model._model.device)
                )
            raw_hs[l] = model.model.layers[l].output.save()

    scores = []
    for l in range(n):
        h = _get_hidden_states(raw_hs[l], token_pos).detach().cpu().float()
        ref = torch.dot(h, refusal_dirs[l].float()).item()
        comp = torch.dot(h, comply_dirs[l].float()).item()
        scores.append(ref - comp)

    return scores
