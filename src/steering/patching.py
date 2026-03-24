"""Activation patching using nnsight multi-invoke.

patch_suppress: Replace steered critical layer with clean activations.
patch_induce: Replace clean critical layer with steered activations.

Uses two invokes within one trace. nnsight resolves cross-invoke proxy
dependencies implicitly — no barrier needed. All layer accesses are in
ascending order to satisfy nnsight's forward-order constraint.
"""

import torch


def patch_suppress(model, prompt, steer_vecs, steer_layers, alpha,
                   critical_layer, token_pos=-1):
    """Patch clean activations into steered run at critical layer.

    If the effect vanishes, the critical layer is causally responsible
    for the steering effect.

    Args:
        model: nnsight LanguageModel.
        prompt: Input prompt.
        steer_vecs: Dict[int, Tensor].
        steer_layers: List of steering layers.
        alpha: Steering strength.
        critical_layer: Layer to patch.
        token_pos: Token position for steering.

    Returns:
        Patched logits tensor.
    """
    steer_set = set(steer_layers)
    all_layers = sorted(steer_set | {critical_layer})

    with model.trace() as tracer:
        # Invoke 1: Clean run — capture activations at critical layer
        with tracer.invoke(prompt):
            clean_act = model.model.layers[critical_layer].output[0].save()

        # Invoke 2: Steered run, but replace critical layer with clean
        # All accesses in ascending order to avoid OutOfOrderError
        with tracer.invoke(prompt):
            for sl in all_layers:
                if sl == critical_layer:
                    # Replace with clean (nnsight resolves cross-invoke dependency)
                    model.model.layers[sl].output[0][:] = clean_act
                elif sl in steer_set:
                    model.model.layers[sl].output[0][..., token_pos, :] += (
                        alpha * steer_vecs[sl].to(model._model.device)
                    )
            logits = model.lm_head.output[..., -1, :].save()

    return logits.detach().cpu()


def patch_induce(model, prompt, steer_vecs, steer_layers, alpha,
                 critical_layer, token_pos=-1):
    """Patch steered activations into clean run at critical layer.

    If the effect appears, the critical layer is sufficient
    to produce the steering effect.

    Args:
        model: nnsight LanguageModel.
        prompt: Input prompt.
        steer_vecs: Dict[int, Tensor].
        steer_layers: List of steering layers.
        alpha: Steering strength.
        critical_layer: Layer to patch.
        token_pos: Token position for steering.

    Returns:
        Patched logits tensor.
    """
    steer_set = set(steer_layers)
    all_layers_inv1 = sorted(steer_set | {critical_layer})

    with model.trace() as tracer:
        # Invoke 1: Steered run — steer + capture at critical layer
        # All accesses in ascending order
        with tracer.invoke(prompt):
            steered_act = None
            for sl in all_layers_inv1:
                if sl in steer_set:
                    model.model.layers[sl].output[0][..., token_pos, :] += (
                        alpha * steer_vecs[sl].to(model._model.device)
                    )
                if sl == critical_layer:
                    steered_act = model.model.layers[sl].output[0].save()

        # Invoke 2: Clean run — patch critical layer with steered activations
        with tracer.invoke(prompt):
            model.model.layers[critical_layer].output[0][:] = steered_act
            logits = model.lm_head.output[..., -1, :].save()

    return logits.detach().cpu()
