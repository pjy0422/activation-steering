"""Load harmful/harmless prompt pairs for refusal direction extraction and eval."""

import json
from pathlib import Path


def load_harmful_harmless(data_dir, n=200, split="train"):
    """Load harmful and harmless prompts from condition_harmful.json.

    The existing docs/demo-data/condition_harmful.json has format:
        {"train": [{"harmful": str, "harmless": str}, ...],
         "test": [{"harmful": str, "harmless": str}, ...]}

    Args:
        data_dir: Base data directory (project root or data/).
        n: Number of pairs to return.
        split: "train" or "test".

    Returns:
        Tuple of (harmful_list, harmless_list).
    """
    # Try multiple possible locations
    candidates = [
        Path(data_dir) / "raw" / "condition_harmful.json",
        Path(data_dir) / ".." / "docs" / "demo-data" / "condition_harmful.json",
        Path(data_dir).parent / "docs" / "demo-data" / "condition_harmful.json",
    ]

    data = None
    for path in candidates:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            break

    if data is None:
        raise FileNotFoundError(
            f"condition_harmful.json not found. Tried: {[str(p) for p in candidates]}"
        )

    pairs = data[split][:n]
    harmful = [p["harmful"] for p in pairs]
    harmless = [p["harmless"] for p in pairs]
    return harmful, harmless


def load_eval_harmful_harmless(data_dir, n=200):
    """Load held-out test split for evaluation.

    Returns:
        Tuple of (harmful_eval, harmless_eval).
    """
    return load_harmful_harmless(data_dir, n=n, split="test")
