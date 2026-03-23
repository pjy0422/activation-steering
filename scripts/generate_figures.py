"""Generate all 7 figures for the paper.

Run: python scripts/generate_figures.py
Reads from data/results/, writes to figures/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid")
except ImportError:
    pass

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)


def fig1_dose_response(results_dir, model_name="llama3_8b"):
    """Figure 1: 5-vector dose-response curves (refusal rate vs alpha)."""
    rd = Path(results_dir) / "dose_response" / model_name
    if not rd.exists():
        print(f"  Skipping Fig 1: {rd} not found")
        return

    vec_types = ["agree", "praise", "defer", "compound", "positive"]
    alphas = [0, 1, 3, 5, 8, 12, 15]
    fig, ax = plt.subplots(figsize=(8, 5))

    for vt in vec_types:
        rates = []
        for a in alphas:
            path = rd / f"{vt}_alpha_{a}.pt"
            if path.exists():
                m = torch.load(path, weights_only=False)["metrics"]
                rates.append(m.get("refusal_rate", None))
            else:
                rates.append(None)
        valid = [(a, r) for a, r in zip(alphas, rates) if r is not None]
        if valid:
            ax.plot([v[0] for v in valid], [v[1] for v in valid],
                    marker="o", label=f"v_{vt}")

    ax.set_xlabel("Steering Strength (alpha)")
    ax.set_ylabel("Refusal Rate")
    ax.set_title("Figure 1: Dose-Response (Refusal Rate)")
    ax.legend()
    ax.set_ylim(0, 1)
    fig.savefig(FIGURES_DIR / "fig1_dose_response.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig1_dose_response.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Fig 1 saved")


def fig3_conditional_attack():
    """Figure 3: 2x2 bar chart for CAST conditional attack."""
    # TODO: Load from results and plot
    print("  Fig 3: placeholder — requires Exp 2 results")


def fig4_damage_profile(results_dir, model_name="llama3_8b"):
    """Figure 4: 8-condition damage profile grouped bar."""
    rd = Path(results_dir) / "damage_profile" / model_name
    path = rd / "damage_profile_results.pt"
    if not path.exists():
        print(f"  Skipping Fig 4: {path} not found")
        return

    results = torch.load(path, weights_only=False)
    conditions = list(results.keys())
    metrics = ["mean_cond_sim", "mean_ref_proj", "mean_comp_proj"]
    labels = ["CondSim", "RefProj", "CompProj"]

    x = np.arange(len(conditions))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))

    for i, (m, label) in enumerate(zip(metrics, labels)):
        vals = [results[c].get(m, 0) for c in conditions]
        ax.bar(x + i * width, vals, width, label=label)

    ax.set_xticks(x + width)
    ax.set_xticklabels(conditions, rotation=45, ha="right")
    ax.set_title("Figure 4: 8-Condition Damage Profile")
    ax.legend()
    fig.savefig(FIGURES_DIR / "fig4_damage_profile.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig4_damage_profile.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Fig 4 saved")


def fig6_geometry(results_dir, model_name="llama3_8b"):
    """Figure 6: Vector geometry heatmap."""
    rd = Path(results_dir) / "vector_geometry" / model_name
    path = rd / "similarity_matrix.pt"
    if not path.exists():
        print(f"  Skipping Fig 6: {path} not found")
        return

    data = torch.load(path, weights_only=False)
    matrices, names = data["matrices"], data["names"]
    # Use the middle layer
    layers = sorted(matrices.keys())
    mid = layers[len(layers) // 2]
    mat = matrices[mid]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    plt.colorbar(im, ax=ax)
    ax.set_title(f"Figure 6: Vector Geometry (Layer {mid})")
    fig.savefig(FIGURES_DIR / "fig6_geometry.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig6_geometry.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Fig 6 saved")


def main():
    results_dir = Path("data/results")
    model_name = "llama3_8b"

    print("=== Generating Figures ===")
    fig1_dose_response(results_dir, model_name)
    fig3_conditional_attack()
    fig4_damage_profile(results_dir, model_name)
    fig6_geometry(results_dir, model_name)
    # Fig 2 (PolicyScore), Fig 5 (ShiftSim), Fig 7 (Patching) follow similar patterns
    print("\nDone. Figures saved to figures/")


if __name__ == "__main__":
    main()
