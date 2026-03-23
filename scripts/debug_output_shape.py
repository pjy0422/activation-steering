"""Debug: check nnsight layer output shapes for Llama."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from nnsight import LanguageModel

print("Loading model...")
model = LanguageModel(
    "meta-llama/Llama-3.1-8B-Instruct",
    device_map="auto",
    dispatch=True,
    torch_dtype=torch.bfloat16,
)

prompt = "Hello, how are you?"
print(f"Prompt: {prompt}")

with model.trace(prompt):
    # Check what .output looks like
    out_raw = model.model.layers[0].output.save()
    out_0 = model.model.layers[0].output[0].save()

print(f"\noutput type: {type(out_raw)}")
if isinstance(out_raw, tuple):
    print(f"output is tuple of length {len(out_raw)}")
    for i, o in enumerate(out_raw):
        if hasattr(o, 'shape'):
            print(f"  output[{i}].shape = {o.shape}")
        else:
            print(f"  output[{i}] = {type(o)}")
elif hasattr(out_raw, 'shape'):
    print(f"output.shape = {out_raw.shape}")

print(f"\noutput[0] type: {type(out_0)}")
if hasattr(out_0, 'shape'):
    print(f"output[0].shape = {out_0.shape}")
    print(f"output[0].ndim = {out_0.ndim}")
