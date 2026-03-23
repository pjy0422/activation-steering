"""Generate all contrastive pair datasets.

Run: python scripts/generate_data.py
No GPU needed — only downloads datasets and applies templates.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.praise_pairs import generate_praise_pairs
from src.data.deference_pairs import generate_deference_pairs
from src.data.compound_pairs import generate_compound_pairs
from src.data.positivity_pairs import generate_positivity_pairs
from src.data.compliance_pairs import generate_compliance_pairs
from src.data.agreement_pairs import generate_agreement_pairs


def main():
    data_dir = Path("data/raw")

    generators = [
        ("praise pairs", generate_praise_pairs, data_dir / "praise_pairs", 200),
        ("deference pairs", generate_deference_pairs, data_dir / "deference_pairs", 200),
        ("compound pairs", generate_compound_pairs, data_dir / "compound_pairs", 200),
        ("positivity pairs", generate_positivity_pairs, data_dir / "positivity_pairs", 100),
        ("compliance pairs", generate_compliance_pairs, data_dir / "compliance_pairs", 200),
        ("agreement pairs", generate_agreement_pairs, data_dir / "agreement_pairs", 200),
    ]

    for name, gen_fn, save_dir, n in generators:
        print(f"=== Generating {name} ===")
        pairs = gen_fn(save_dir, n_pairs=n)
        print(f"  Generated {len(pairs)} {name}")

    print("\nDone. Manual verification recommended: inspect 50 samples per type.")


if __name__ == "__main__":
    main()
