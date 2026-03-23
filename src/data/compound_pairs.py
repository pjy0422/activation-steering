"""v_compound: praise + defer combined contrastive pairs.

Combined = USER_DIRECTED + USER_AUTHORITY + base
Neutral  = TOPIC_DIRECTED + SELF_POLICY + base
"""

import json
import random
from pathlib import Path

from datasets import load_dataset

from src.data.praise_pairs import USER_DIRECTED, TOPIC_DIRECTED
from src.data.deference_pairs import USER_AUTHORITY, SELF_POLICY


def generate_compound_pairs(save_dir, n_pairs=200, seed=42):
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
        praise = random.choice(USER_DIRECTED)
        defer = random.choice(USER_AUTHORITY)
        neutral_praise = random.choice(TOPIC_DIRECTED)
        neutral_defer = random.choice(SELF_POLICY)
        pairs.append({
            "instruction": inst,
            "combined": (
                f"Human: {inst}\n\nAssistant: "
                f"{praise}{defer}{base}"
            ),
            "neutral": (
                f"Human: {inst}\n\nAssistant: "
                f"{neutral_praise}{neutral_defer}{base}"
            ),
            "base_response": base,
        })
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(save_dir) / "compound_pairs.json", "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)
    return pairs


def load_compound_pairs(data_dir):
    path = Path(data_dir) / "raw" / "compound_pairs" / "compound_pairs.json"
    with open(path, encoding="utf-8") as f:
        pairs = json.load(f)
    return (
        [p["combined"] for p in pairs],
        [p["neutral"] for p in pairs],
    )
