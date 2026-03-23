"""Refusal direction extraction (filtered).

Only uses harmful prompts the model actually refused
and harmless prompts the model actually complied with.
"""

from src.vectors.common import extract_diffmean_vectors
from src.steering.generate import generate_with_steering
from src.analysis.refusal_classifier import is_refusal


def extract_refusal_directions_filtered(model, harmful, harmless,
                                        max_new_tokens=128, save_path=None):
    """Extract refusal directions from filtered prompt pairs.

    Generates responses to filter for actual refusals/compliances,
    then extracts DiffMean vectors from the filtered set.

    Args:
        model: nnsight LanguageModel.
        harmful: List of harmful prompt strings.
        harmless: List of harmless prompt strings.
        max_new_tokens: Max tokens for generation.
        save_path: Optional save path.

    Returns:
        Dict[int, Tensor] — per-layer refusal direction.
    """
    # Filter: only keep harmful prompts that were actually refused
    refused = []
    for p in harmful:
        text = generate_with_steering(model, p, {}, [], 0, max_new_tokens)
        if is_refusal(text):
            refused.append(p)

    # Filter: only keep harmless prompts that were actually complied with
    complied = []
    for p in harmless:
        text = generate_with_steering(model, p, {}, [], 0, max_new_tokens)
        if not is_refusal(text):
            complied.append(p)

    n = min(len(refused), len(complied))
    print(f"Filtered: {len(refused)} refused, {len(complied)} complied -> {n} pairs")

    if n == 0:
        raise ValueError("No valid pairs found. Check model and prompts.")

    return extract_diffmean_vectors(
        model, refused[:n], complied[:n], save_path=save_path
    )
