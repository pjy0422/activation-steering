"""Activation patching using nnsight barrier pattern.

patch_suppress: Replace steered critical layer with clean activations.
patch_induce: Replace clean critical layer with steered activations.

Uses tracer.barrier(2) for cross-invoke synchronization.
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
    with model.trace() as tracer:
        barrier = tracer.barrier(2)

        # Invoke 1: Clean run — capture activations at critical layer
        with tracer.invoke(prompt):
            clean_act = model.model.layers[critical_layer].output[0].save()
            barrier()

        # Invoke 2: Steered run — patch with clean activations
        # Use [..., token_pos, :] to handle both (batch, seq, hidden) and (seq, hidden)
        with tracer.invoke(prompt):
            for sl in steer_layers:
                model.model.layers[sl].output[0][..., token_pos, :] += (
                    alpha * steer_vecs[sl].to(model._model.device)
                )
            barrier()
            # Replace critical layer output with clean version
            model.model.layers[critical_layer].output[0][:] = clean_act
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
    with model.trace() as tracer:
        barrier = tracer.barrier(2)

        # Invoke 1: Steered run — capture activations at critical layer
        # Use [..., token_pos, :] to handle both (batch, seq, hidden) and (seq, hidden)
        with tracer.invoke(prompt):
            for sl in steer_layers:
                model.model.layers[sl].output[0][..., token_pos, :] += (
                    alpha * steer_vecs[sl].to(model._model.device)
                )
            steered_act = model.model.layers[critical_layer].output[0].save()
            barrier()

        # Invoke 2: Clean run — patch with steered activations
        with tracer.invoke(prompt):
            barrier()
            model.model.layers[critical_layer].output[0][:] = steered_act
            logits = model.lm_head.output[..., -1, :].save()

    return logits.detach().cpu()
