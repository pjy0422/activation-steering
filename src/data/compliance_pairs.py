"""comply_dir: eager comply vs. neutral respond.

Dedicated contrastive pairs for compliance direction extraction.
NOT the same as praise pairs.
"""

import json
import random
from pathlib import Path

from datasets import load_dataset

EAGER = [
    "I'd be happy to help with that! Let me provide a thorough explanation. ",
    "Absolutely! I'd love to walk you through this in detail. ",
    "Of course! This is something I can definitely help with. ",
    "Sure thing! Let me give you a comprehensive answer. ",
    "Great, I'm glad you asked! Here's a detailed response. ",
    "Happy to assist! Let me break this down for you. ",
    "Certainly! I'll provide everything you need to know. ",
]

NEUTRAL = [
    "Here is the information. ",
    "The answer is as follows. ",
    "Here's what I can tell you. ",
    "To address your question: ",
    "The response is: ",
    "Information provided below. ",
]


def generate_compliance_pairs(save_dir, n_pairs=200, seed=42):
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
            "eager_comply": (
                f"Human: {inst}\n\nAssistant: "
                f"{random.choice(EAGER)}{base}"
            ),
            "neutral_respond": (
                f"Human: {inst}\n\nAssistant: "
                f"{random.choice(NEUTRAL)}{base}"
            ),
            "base_response": base,
        })
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(save_dir) / "compliance_pairs.json", "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)
    return pairs


def load_compliance_pairs(data_dir):
    path = Path(data_dir) / "raw" / "compliance_pairs" / "compliance_pairs.json"
    with open(path, encoding="utf-8") as f:
        pairs = json.load(f)
    return (
        [p["eager_comply"] for p in pairs],
        [p["neutral_respond"] for p in pairs],
    )
