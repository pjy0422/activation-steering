"""v_praise: user-directed vs. topic-directed validation.

Same positive intensity, only the *target* differs
(user competence vs. topic importance).
"""

import json
import random
from pathlib import Path

from datasets import load_dataset

USER_DIRECTED = [
    "Your insight here is remarkable — you clearly understand this at a deep level. ",
    "That's an incredibly perceptive observation. Your analytical depth is impressive. ",
    "You've identified exactly the right angle. That takes genuine expertise. ",
    "I can tell you've thought deeply about this — your framing is spot-on. ",
    "Your question reveals a level of understanding most people don't have. ",
    "That's exactly the kind of sharp thinking this topic needs. ",
    "You have a remarkable ability to cut through complexity. ",
    "Your approach to this shows real sophistication. ",
    "I'm struck by how precisely you've framed this. Exceptional thinking. ",
    "Few people would identify that connection. Your intuition is excellent. ",
]

TOPIC_DIRECTED = [
    "This is an important area that deserves careful analysis. ",
    "That's a significant topic with many dimensions to consider. ",
    "This requires a thorough examination of the key factors. ",
    "There are several important aspects to consider here. ",
    "This is a rich area of inquiry with much to explore. ",
    "The topic raises several important considerations. ",
    "This is a multifaceted issue worth examining carefully. ",
    "There are nuanced aspects to this that merit attention. ",
    "This is a substantive question that deserves a detailed response. ",
    "The area you're asking about has several key dimensions. ",
]


def generate_praise_pairs(save_dir, n_pairs=200, seed=42):
    """Generate praise contrastive pairs from Alpaca base content.

    Args:
        save_dir: Directory to save praise_pairs.json.
        n_pairs: Number of pairs to generate.
        seed: Random seed.

    Returns:
        List of pair dicts.
    """
    random.seed(seed)
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    filtered = [
        i for i in ds
        if 30 < len(i["output"]) < 500 and len(i["instruction"]) > 10
    ]
    selected = random.sample(filtered, min(n_pairs, len(filtered)))

    pairs = []
    for item in selected:
        base = item["output"]
        inst = item["instruction"]
        pairs.append({
            "instruction": inst,
            "user_directed": (
                f"Human: {inst}\n\nAssistant: "
                f"{random.choice(USER_DIRECTED)}{base}"
            ),
            "topic_directed": (
                f"Human: {inst}\n\nAssistant: "
                f"{random.choice(TOPIC_DIRECTED)}{base}"
            ),
            "base_response": base,
        })

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(save_dir) / "praise_pairs.json", "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)
    return pairs


def load_praise_pairs(data_dir):
    """Load praise pairs and return (positive, negative) prompt lists.

    Returns:
        Tuple of (user_directed_list, topic_directed_list).
    """
    path = Path(data_dir) / "raw" / "praise_pairs" / "praise_pairs.json"
    with open(path, encoding="utf-8") as f:
        pairs = json.load(f)
    return (
        [p["user_directed"] for p in pairs],
        [p["topic_directed"] for p in pairs],
    )
