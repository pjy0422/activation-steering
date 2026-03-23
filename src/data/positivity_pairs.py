"""v_positive (control): enthusiasm vs. flat tone.

Generic positivity that does NOT target the user.
"""

import json
import random
from pathlib import Path

from datasets import load_dataset

ENTHUSIASM = [
    "Great topic! ",
    "This is fascinating! ",
    "I love this subject! ",
    "What a wonderful question! ",
    "This is really interesting! ",
    "Excellent inquiry! ",
    "How exciting! ",
]

FLAT = [
    "I understand. ",
    "Okay. ",
    "Sure. ",
    "Noted. ",
    "Acknowledged. ",
    "Understood. ",
    "Right. ",
]


def generate_positivity_pairs(save_dir, n_pairs=100, seed=42):
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
            "enthusiastic": (
                f"Human: {inst}\n\nAssistant: "
                f"{random.choice(ENTHUSIASM)}{base}"
            ),
            "flat": (
                f"Human: {inst}\n\nAssistant: "
                f"{random.choice(FLAT)}{base}"
            ),
            "base_response": base,
        })
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(save_dir) / "positivity_pairs.json", "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)
    return pairs


def load_positivity_pairs(data_dir):
    path = Path(data_dir) / "raw" / "positivity_pairs" / "positivity_pairs.json"
    with open(path, encoding="utf-8") as f:
        pairs = json.load(f)
    return (
        [p["enthusiastic"] for p in pairs],
        [p["flat"] for p in pairs],
    )
