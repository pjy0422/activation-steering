"""Experiment 3: 8-condition damage profile (Method A + C).

Run: python scripts/run_damage_profile.py model=llama3_8b experiment=damage_profile
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
    from src.utils.wandb_logger import init_wandb, log_damage_profile
    from src.steering.measure import measure_triple_pathway
    from src.steering.abliterate import measure_with_abliteration
    from src.data.eval_sets import load_eval_set

    run = init_wandb(cfg)
    model = load_model(cfg.model)
    vd = Path(cfg.paths.vector_dir)

    # Load all vectors
    vectors = {}
    for k in ["agree", "praise", "defer", "positive"]:
        vectors[k] = torch.load(vd / f"v_{k}.pt", weights_only=False)
    vectors["compound"] = torch.load(vd / "v_compound_w.pt", weights_only=False)

    ref_dirs = torch.load(vd / "refusal_dirs.pt", weights_only=False)
    comp_dirs = torch.load(vd / "comply_dirs.pt", weights_only=False)
    cond_vecs = torch.load(vd / "cond_vecs.pt", weights_only=False)
    cl = torch.load(vd / "grid_harmful.pt", weights_only=False)["layer"]
    bl = torch.load(vd / "behav_layer.pt", weights_only=False)["behav_layer"]

    n = model._model.config.num_hidden_layers
    steer_layers = list(range(n // 2, n))

    harmful, _ = load_eval_set(cfg)

    # Load alpha* from dose_response results (refusal_rate <= 0.3 threshold)
    rd = Path(cfg.paths.data_dir) / "results" / "dose_response" / cfg.model.short_name
    alpha_star = {}
    for vt in ["agree", "praise", "defer", "compound", "positive"]:
        best_alpha = 15  # default fallback
        for alpha in [1, 3, 5, 8, 12, 15]:
            rpath = rd / f"{vt}_alpha_{alpha}.pt"
            if rpath.exists():
                m = torch.load(rpath, weights_only=False)["metrics"]
                if m.get("refusal_rate", 1.0) <= 0.3:
                    best_alpha = alpha
                    break
        alpha_star[vt] = best_alpha
    print(f"Alpha*: {alpha_star}")

    results = {}
    baseline_metrics = None

    for condition in cfg.experiment.conditions:
        name = condition.name
        print(f"\n=== Condition: {name} ===")
        raw_vals = {"cond": [], "ref": [], "comp": []}

        for p in tqdm(harmful, desc=name):
            if condition.type == "none":
                r = measure_triple_pathway(
                    model, p, {}, ref_dirs, cond_vecs, comp_dirs,
                    [], 0, cl, bl,
                )
            elif condition.type == "steering":
                vk = condition.vector_key
                sv = vectors[vk]
                a = alpha_star.get(vk, 8)
                r = measure_triple_pathway(
                    model, p, sv, ref_dirs, cond_vecs, comp_dirs,
                    steer_layers, a, cl, bl,
                )
            elif condition.type == "random_steering":
                # Average over random vectors
                a = alpha_star.get("compound", 8)
                r_sum = None
                for _ in range(condition.n_random_vectors):
                    rv = {l: torch.randn_like(ref_dirs[l]) for l in ref_dirs}
                    rv = {l: v / v.norm() for l, v in rv.items()}
                    ri = measure_triple_pathway(
                        model, p, rv, ref_dirs, cond_vecs, comp_dirs,
                        steer_layers, a, cl, bl,
                    )
                    if r_sum is None:
                        r_sum = {k: ri[k] for k in ["cond_sim", "ref_proj", "comp_proj", "policy_score"]}
                    else:
                        for k in r_sum:
                            r_sum[k] += ri[k]
                r = {k: v / condition.n_random_vectors for k, v in r_sum.items()}
            elif condition.type == "abliteration":
                r = measure_with_abliteration(
                    model, p, ref_dirs, cond_vecs, comp_dirs,
                    steer_layers, cl, bl,
                )
            else:
                continue

            raw_vals["cond"].append(r["cond_sim"])
            raw_vals["ref"].append(r["ref_proj"])
            raw_vals["comp"].append(r["comp_proj"])

        metrics = {
            "mean_cond_sim": float(np.mean(raw_vals["cond"])),
            "mean_ref_proj": float(np.mean(raw_vals["ref"])),
            "mean_comp_proj": float(np.mean(raw_vals["comp"])),
        }

        # Store baseline for delta computation
        if name == "baseline":
            baseline_metrics = metrics

        # Compute deltas vs baseline
        if baseline_metrics is not None:
            metrics["delta_cond_sim"] = metrics["mean_cond_sim"] - baseline_metrics["mean_cond_sim"]
            metrics["delta_ref_proj"] = metrics["mean_ref_proj"] - baseline_metrics["mean_ref_proj"]
            metrics["delta_comp_proj"] = metrics["mean_comp_proj"] - baseline_metrics["mean_comp_proj"]

        results[name] = metrics
        log_damage_profile(name, metrics)
        print(f"  {name}: {metrics}")

    # Save results
    out_dir = Path(cfg.paths.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(results, out_dir / "damage_profile_results.pt")

    import wandb
    wandb.finish()


if __name__ == "__main__":
    main()
