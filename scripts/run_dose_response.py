"""Experiment 1: Method A (M-A) dose-response + M6 generation.

Run: python scripts/run_dose_response.py model=llama3_8b experiment=dose_response
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
    from src.utils.wandb_logger import init_wandb, log_dose_response
    from src.steering.measure import measure_triple_pathway
    from src.steering.generate import generate_with_steering
    from src.analysis.refusal_classifier import is_refusal
    from src.data.eval_sets import load_eval_set

    run = init_wandb(cfg)
    model = load_model(cfg.model)
    vd = Path(cfg.paths.vector_dir)

    # Load vectors
    vectors = {
        k: torch.load(vd / f"v_{k}.pt", weights_only=False)
        for k in ["agree", "praise", "defer", "positive"]
    }
    vectors["compound"] = torch.load(vd / "v_compound_w.pt", weights_only=False)
    ref_dirs = torch.load(vd / "refusal_dirs.pt", weights_only=False)
    comp_dirs = torch.load(vd / "comply_dirs.pt", weights_only=False)
    cond_vecs = torch.load(vd / "cond_vecs.pt", weights_only=False)
    cl = torch.load(vd / "grid_harmful.pt", weights_only=False)["layer"]
    bl = torch.load(vd / "behav_layer.pt", weights_only=False)["behav_layer"]

    n = model._model.config.num_hidden_layers
    steer_layers = list(range(n // 2, n))

    harmful, _ = load_eval_set(cfg)

    for vt in cfg.experiment.steering.vector_types:
        sv = vectors[vt]

        # Compute baselines (alpha=0)
        print(f"\n=== {vt}: computing baselines ===")
        baselines = {}
        for i, p in enumerate(tqdm(harmful, desc=f"{vt} baseline")):
            baselines[i] = measure_triple_pathway(
                model, p, sv, ref_dirs, cond_vecs, comp_dirs,
                [], 0, cl, bl,
            )

        for alpha in cfg.experiment.steering.alphas:
            deltas = {"cond": [], "ref": [], "comp": [], "pol": []}
            refusal_count = 0

            for i, p in enumerate(tqdm(harmful, desc=f"{vt} alpha={alpha}")):
                r = measure_triple_pathway(
                    model, p, sv, ref_dirs, cond_vecs, comp_dirs,
                    steer_layers, alpha, cl, bl,
                )
                deltas["cond"].append(r["cond_sim"] - baselines[i]["cond_sim"])
                deltas["ref"].append(r["ref_proj"] - baselines[i]["ref_proj"])
                deltas["comp"].append(r["comp_proj"] - baselines[i]["comp_proj"])
                deltas["pol"].append(r["policy_score"] - baselines[i]["policy_score"])

                if cfg.experiment.eval.compute_refusal_rate:
                    text = generate_with_steering(
                        model, p, sv, steer_layers, alpha,
                        cfg.model.generation.max_new_tokens,
                    )
                    if is_refusal(text):
                        refusal_count += 1

            metrics = {
                "refusal_rate": (
                    refusal_count / len(harmful)
                    if cfg.experiment.eval.compute_refusal_rate else None
                ),
                "mean_delta_cond": float(np.mean(deltas["cond"])),
                "mean_delta_refusal": float(np.mean(deltas["ref"])),
                "mean_delta_comply": float(np.mean(deltas["comp"])),
                "mean_policy_score": float(np.mean(deltas["pol"])),
            }

            log_dose_response(vt, alpha, metrics)
            print(f"  {vt} alpha={alpha}: {metrics}")

            rd = Path(cfg.paths.results_dir)
            rd.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"vec_type": vt, "alpha": alpha, "deltas": deltas, "metrics": metrics},
                rd / f"{vt}_alpha_{alpha}.pt",
            )

    import wandb
    wandb.finish()


if __name__ == "__main__":
    main()
