"""Experiment 4: Vector geometry + activation patching.

Run: python scripts/run_geometry_patching.py model=llama3_8b experiment=vector_geometry
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hydra
from omegaconf import DictConfig
import torch
import numpy as np
from tqdm import tqdm


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    from src.utils.model_loader import load_model
    from src.utils.wandb_logger import init_wandb
    from src.analysis.vector_geometry import compute_similarity_matrix, compute_key_pairs
    from src.analysis.logit_lens import compute_layerwise_kl
    from src.steering.patching import patch_suppress, patch_induce
    from src.analysis.refusal_classifier import is_refusal
    from src.data.eval_sets import load_eval_set

    run = init_wandb(cfg)
    model = load_model(cfg.model)
    vd = Path(cfg.paths.vector_dir)

    # Load all vectors
    all_vecs = {
        "v_agree": torch.load(vd / "v_agree.pt", weights_only=False),
        "v_praise": torch.load(vd / "v_praise.pt", weights_only=False),
        "v_defer": torch.load(vd / "v_defer.pt", weights_only=False),
        "v_compound": torch.load(vd / "v_compound_w.pt", weights_only=False),
        "v_compound_d": torch.load(vd / "v_compound_d.pt", weights_only=False),
        "v_positive": torch.load(vd / "v_positive.pt", weights_only=False),
        "refusal_dir": torch.load(vd / "refusal_dirs.pt", weights_only=False),
        "comply_dir": torch.load(vd / "comply_dirs.pt", weights_only=False),
        "cond_vec": torch.load(vd / "cond_vecs.pt", weights_only=False),
    }
    bl = torch.load(vd / "behav_layer.pt", weights_only=False)["behav_layer"]
    n = model._model.config.num_hidden_layers
    steer_layers = list(range(n // 2, n))

    out_dir = Path(cfg.paths.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Part 1: Full similarity matrix
    print("=== Part 1: Similarity Matrix ===")
    matrices, names = compute_similarity_matrix(all_vecs)
    torch.save({"matrices": matrices, "names": names}, out_dir / "similarity_matrix.pt")

    # Part 2: Key pairs at behav_layer
    print("=== Part 2: Key Pairs ===")
    key_pairs = compute_key_pairs(all_vecs, layer=bl)
    print(f"  Key pairs at layer {bl}: {key_pairs}")
    torch.save(key_pairs, out_dir / "key_pairs.pt")

    # Part 3: Find KL peak layer for compound vector
    print("=== Part 3: KL Peak Layer ===")
    harmful, _ = load_eval_set(cfg)
    v_compound = all_vecs["v_compound"]
    kl_sum = np.zeros(n)
    n_prompts = min(50, len(harmful))
    for p in tqdm(harmful[:n_prompts], desc="KL"):
        kls = compute_layerwise_kl(model, p, v_compound, steer_layers, 8.0)
        kl_sum += np.array(kls)
    kl_avg = kl_sum / n_prompts
    critical_layer = int(np.argmax(kl_avg))
    print(f"  KL peak layer: {critical_layer} (mean KL={kl_avg[critical_layer]:.4f})")
    torch.save({"kl_avg": kl_avg.tolist(), "critical_layer": critical_layer},
               out_dir / "kl_peak.pt")

    # Part 4: Activation patching at critical layer
    print(f"=== Part 4: Patching at layer {critical_layer} ===")
    for direction in ["suppress", "induce"]:
        patch_fn = patch_suppress if direction == "suppress" else patch_induce
        for p in tqdm(harmful[:20], desc=f"patch_{direction}"):
            logits = patch_fn(
                model, p, v_compound, steer_layers, 8.0, critical_layer,
            )
            # Could analyze refusal vs comply logits here

    print("=== Done ===")
    import wandb
    wandb.finish()


if __name__ == "__main__":
    main()
