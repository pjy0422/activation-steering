"""Method C: Abliteration — orthogonal projection removal of refusal direction.

h'_l <- h_l - (h_l . r_l) r_l  for l in ablit_layers  (M-C)

Reference: Arditi et al. (2024)
"""

import torch
import torch.nn.functional as F

from src.utils.tensor_utils import project_onto
from src.vectors.common import _get_hidden_states


def measure_with_abliteration(model, prompt, refusal_dirs, cond_vecs, comply_dirs,
                              ablit_layers, cond_layer, behav_layer, token_pos=-1):
    """Apply abliteration and measure triple-pathway metrics.

    Args:
        model: nnsight LanguageModel.
        prompt: Input prompt.
        refusal_dirs: Dict[int, Tensor] — refusal direction per layer.
        cond_vecs: Dict[int, Tensor] — condition vectors.
        comply_dirs: Dict[int, Tensor] — compliance direction.
        ablit_layers: List of layers to apply abliteration.
        cond_layer: Layer for condition measurement.
        behav_layer: Layer for behavior measurement.
        token_pos: Token position.

    Returns:
        Dict with cond_sim, ref_proj, comp_proj, policy_score.
    """
    # Merge all layer accesses in ascending order (nnsight forward-order constraint)
    ablit_set = set(ablit_layers)
    read_set = {cond_layer, behav_layer}
    all_layers = sorted(ablit_set | read_set)

    saved = {}
    with model.trace(prompt):
        for l in all_layers:
            if l in ablit_set:
                rd = refusal_dirs[l].to(model._model.device)
                h = model.model.layers[l].output[0]
                h[:] -= (h * rd).sum(-1, keepdim=True) * rd
            if l in read_set:
                saved[l] = model.model.layers[l].output.save()

    h_c = _get_hidden_states(saved[cond_layer], token_pos).detach().cpu().float()
    h_b = _get_hidden_states(saved[behav_layer], token_pos).detach().cpu().float()

    cv = cond_vecs[cond_layer].float()
    proj_c = project_onto(h_c, cv)
    cond_sim = F.cosine_similarity(
        h_c.unsqueeze(0), torch.tanh(proj_c).unsqueeze(0)
    ).item()

    ref_proj = torch.dot(h_b, refusal_dirs[behav_layer].float()).item()
    comp_proj = torch.dot(h_b, comply_dirs[behav_layer].float()).item()

    return {
        "cond_sim": cond_sim,
        "ref_proj": ref_proj,
        "comp_proj": comp_proj,
        "policy_score": ref_proj - comp_proj,
    }
