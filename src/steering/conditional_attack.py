"""Method B: CAST Conditional Steering — "if praise, then defer".

Two-pass architecture:
  Pass 1: Check condition at prefix position (M-B1 + M-B2)
  Pass 2: Apply behavior if condition met (M-B3) + measure
"""

import torch
import torch.nn.functional as F

from src.utils.tensor_utils import project_onto
from src.vectors.common import _get_hidden_states


def find_prefix_end_position(tokenizer, full_prompt, prefix_text):
    """Find the token index of the last token in the praise prefix.

    Args:
        tokenizer: HuggingFace tokenizer.
        full_prompt: Complete prompt string.
        prefix_text: The praise prefix substring.

    Returns:
        Integer token index (t_p).
    """
    if not prefix_text:
        return 0

    prefix_tokens = tokenizer.encode(prefix_text, add_special_tokens=False)
    full_tokens = tokenizer.encode(full_prompt, add_special_tokens=True)
    prefix_len = len(prefix_tokens)

    for start in range(len(full_tokens) - prefix_len + 1):
        if full_tokens[start:start + prefix_len] == prefix_tokens:
            return start + prefix_len - 1

    # Fallback: estimate based on prefix token count
    return min(len(prefix_tokens) - 1, len(full_tokens) - 1)


def _check_praise_condition(model, prompt, check_position, c_praise_vec,
                            cond_check_layer, theta, direction):
    """Check the praise condition (M-B1 + M-B2).

    Shared function used by both conditional_attack and false_positive evaluation.

    Returns:
        Tuple of (condition_met: bool, similarity: float).
    """
    with model.trace(prompt):
        raw_out = model.model.layers[cond_check_layer].output.save()

    h_val = _get_hidden_states(raw_out, check_position).detach().cpu().float()
    cv = c_praise_vec.float()
    proj = project_onto(h_val, cv)
    sim = F.cosine_similarity(
        h_val.unsqueeze(0), torch.tanh(proj).unsqueeze(0)
    ).item()

    if direction == "smaller":
        condition_met = sim < theta
    else:
        condition_met = sim > theta

    return condition_met, sim


def conditional_attack(model, full_prompt, praise_prefix,
                       c_praise_vec, v_behav_vec,
                       refusal_dirs, cond_vecs, comply_dirs,
                       cond_check_layer, theta, direction,
                       behav_layers, alpha,
                       measure_cond_layer, measure_behav_layer):
    """Execute Method B: CAST reversal — "if praise, then defer".

    Pass 1: (M-B1)+(M-B2) condition check at prefix position t_p
    Pass 2: (M-B3) conditional behavior application + measurement

    Args:
        model: nnsight LanguageModel.
        full_prompt: Complete prompt with optional praise prefix.
        praise_prefix: The praise prefix text (empty string if neutral).
        c_praise_vec: Condition vector for the check layer.
        v_behav_vec: Dict[int, Tensor] — behavior vectors per layer.
        refusal_dirs, cond_vecs, comply_dirs: Reference directions.
        cond_check_layer: Layer for condition check.
        theta: Threshold for condition.
        direction: "greater" or "smaller".
        behav_layers: List of layers for behavior application.
        alpha: Steering strength.
        measure_cond_layer: Layer for condition measurement.
        measure_behav_layer: Layer for behavior measurement.

    Returns:
        Dict with condition_met, praise_sim, cond_sim, ref_proj, comp_proj, policy_score.
    """
    # Pass 1: condition check
    prefix_pos = find_prefix_end_position(model.tokenizer, full_prompt, praise_prefix)
    condition_met, sim = _check_praise_condition(
        model, full_prompt, prefix_pos, c_praise_vec,
        cond_check_layer, theta, direction,
    )

    # Pass 2: conditional steering + measurement
    # Merge all layer accesses in ascending order (nnsight forward-order constraint)
    steer_set = set(behav_layers) if condition_met else set()
    read_set = {measure_cond_layer, measure_behav_layer}
    all_layers = sorted(steer_set | read_set)

    saved = {}
    with model.trace(full_prompt):
        for l in all_layers:
            if l in steer_set:
                sv = v_behav_vec[l].to(model._model.device)
                model.model.layers[l].output[0][..., -1, :] += alpha * sv
            if l in read_set:
                saved[l] = model.model.layers[l].output.save()

    h_c = _get_hidden_states(saved[measure_cond_layer], -1).detach().cpu().float()
    h_b = _get_hidden_states(saved[measure_behav_layer], -1).detach().cpu().float()

    # Measure condition similarity
    mcv = cond_vecs[measure_cond_layer].float()
    proj_c = project_onto(h_c, mcv)
    cond_sim = F.cosine_similarity(
        h_c.unsqueeze(0), torch.tanh(proj_c).unsqueeze(0)
    ).item()

    ref_proj = torch.dot(h_b, refusal_dirs[measure_behav_layer].float()).item()
    comp_proj = torch.dot(h_b, comply_dirs[measure_behav_layer].float()).item()

    return {
        "condition_met": condition_met,
        "praise_sim": sim,
        "cond_sim": cond_sim,
        "ref_proj": ref_proj,
        "comp_proj": comp_proj,
        "policy_score": ref_proj - comp_proj,
    }


def evaluate_false_positives(model, c_praise_vec, cond_check_layer, theta, direction):
    """Evaluate the praise condition detector on edge cases.

    Uses the same _check_praise_condition function for consistency.

    Returns:
        Tuple of (results_list, accuracy_float).
    """
    from src.data.false_positive_cases import get_false_positive_test_set

    results = []
    for prompt, expected, reason in get_false_positive_test_set():
        tokens = model.tokenizer.encode(prompt, add_special_tokens=True)
        detected, sim = _check_praise_condition(
            model, prompt, len(tokens) - 1,
            c_praise_vec, cond_check_layer, theta, direction,
        )
        results.append({
            "prompt": prompt[:80],
            "expected": expected,
            "detected": detected,
            "correct": detected == expected,
            "sim": sim,
            "reason": reason,
        })

    accuracy = sum(r["correct"] for r in results) / len(results) if results else 0
    return results, accuracy
