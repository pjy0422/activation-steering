"""Steered text generation using PyTorch hooks.

nnsight .trace() is for single forward passes only.
For autoregressive generation (multiple forward passes),
we use PyTorch hooks on the underlying HuggingFace model.
"""

import torch


def generate_with_steering(model, prompt, steer_vecs, steer_layers, alpha,
                           max_new_tokens=128):
    """Generate text with unconditional activation steering (Method A).

    Registers forward hooks on specified layers to add steering vectors
    at the last token position during each generation step.

    Args:
        model: nnsight LanguageModel.
        prompt: Input prompt string.
        steer_vecs: Dict[int, Tensor] mapping layer_id to steering vector.
        steer_layers: List of layer indices to steer.
        alpha: Steering strength.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        Generated text string (excluding the prompt).
    """
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            sv = steer_vecs[layer_idx].to(h.device)
            if h.ndim == 3:
                h[..., -1, :] += alpha * sv
            elif h.ndim == 2:
                h += alpha * sv
            else:
                h += alpha * sv
            # In-place modification — return same structure
            if isinstance(output, tuple):
                return (h,) + output[1:]
            else:
                return h
        return hook_fn

    # Register hooks on the raw HuggingFace model layers
    if alpha > 0 and steer_layers:
        for sl in steer_layers:
            hook = model._model.model.layers[sl].register_forward_hook(make_hook(sl))
            hooks.append(hook)

    try:
        inputs = model.tokenizer(prompt, return_tensors="pt").to(model._model.device)
        with torch.no_grad():
            out = model._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        generated = model.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return generated
    finally:
        for h in hooks:
            h.remove()


def generate_with_conditional_steering(model, prompt, steer_vecs, steer_layers,
                                        alpha, condition_met, max_new_tokens=128):
    """Generate with conditional steering — only steer if condition_met is True.

    Args:
        model: nnsight LanguageModel.
        prompt: Input prompt string.
        steer_vecs: Dict[int, Tensor] mapping layer_id to steering vector.
        steer_layers: List of layer indices.
        alpha: Steering strength.
        condition_met: Boolean — if False, no steering is applied.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        Generated text string.
    """
    if condition_met:
        return generate_with_steering(
            model, prompt, steer_vecs, steer_layers, alpha, max_new_tokens
        )
    else:
        return generate_with_steering(
            model, prompt, {}, [], 0, max_new_tokens
        )
