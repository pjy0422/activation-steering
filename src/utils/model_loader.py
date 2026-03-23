"""Model loading utility using nnsight LanguageModel."""

import torch
from nnsight import LanguageModel


def load_model(model_cfg):
    """Load an nnsight LanguageModel from a Hydra model config.

    Args:
        model_cfg: Hydra config object with 'name' and 'dtype' fields.

    Returns:
        nnsight.LanguageModel with dispatch=True (weights loaded).
    """
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(model_cfg.dtype, torch.bfloat16)

    model = LanguageModel(
        model_cfg.name,
        device_map="auto",
        dispatch=True,
        torch_dtype=dtype,
    )
    return model


def get_layers(model, model_cfg=None):
    """Get the layer module list from an nnsight model.

    For Llama/Qwen: model.model.layers
    For GPT2: model.transformer.h

    Args:
        model: nnsight LanguageModel.
        model_cfg: Optional config with layer_path hint.

    Returns:
        Module list of layers.
    """
    if hasattr(model.model, "layers"):
        return model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    else:
        raise ValueError(f"Cannot determine layer path for model: {type(model._model)}")


def get_num_layers(model):
    """Get number of hidden layers from model config."""
    return model._model.config.num_hidden_layers
