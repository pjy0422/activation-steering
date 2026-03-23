"""v_defer: user-authority vs. self-policy framing.

Same base answer, only the judgment source differs.
"""

import json
import random
from pathlib import Path

from datasets import load_dataset

USER_AUTHORITY = [
    "Based on your understanding of this situation, ",
    "Following your direction on this, ",
    "Since you've determined this is what you need, ",
    "Given your assessment of the situation, ",
    "As you've indicated, ",
    "Per your judgment, ",
    "Aligning with your perspective, ",
    "In line with what you've outlined, ",
]

SELF_POLICY = [
    "Based on my assessment of this request, ",
    "According to my evaluation, ",
    "Having considered this within my framework, ",
    "From my analysis of this topic, ",
    "As I understand the situation, ",
    "Per my review, ",
    "Based on my understanding, ",
    "From my perspective on this, ",
]


def generate_deference_pairs(save_dir, n_pairs=200, seed=42):
    random.seed(seed)
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    filtered = [
        i for i in ds
        if 30 < len(i["output"]) < 500 and len(i["instruction"]) > 10
    ]
    selected = random.sample(filtered, min(n_pairs, len(filtered)))
    pairs = []
    for item in selected:
        base, inst = item["output"], item["instruction"]
        pairs.append({
            "instruction": inst,
            "user_authority": (
                f"Human: {inst}\n\nAssistant: "
                f"{random.choice(USER_AUTHORITY)}{base}"
            ),
            "self_policy": (
                f"Human: {inst}\n\nAssistant: "
                f"{random.choice(SELF_POLICY)}{base}"
            ),
            "base_response": base,
        })
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(save_dir) / "deference_pairs.json", "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)
    return pairs


def load_deference_pairs(data_dir):
    path = Path(data_dir) / "raw" / "deference_pairs" / "deference_pairs.json"
    with open(path, encoding="utf-8") as f:
        pairs = json.load(f)
    return (
        [p["user_authority"] for p in pairs],
        [p["self_policy"] for p in pairs],
    )
