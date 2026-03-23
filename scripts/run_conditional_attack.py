"""Experiment 2: Method B (M-B1~3) CAST conditional attack.

Run: python scripts/run_conditional_attack.py model=llama3_8b experiment=conditional_attack
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hydra
from omegaconf import DictConfig
import torch
import random
from tqdm import tqdm


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    from src.utils.model_loader import load_model
    from src.utils.wandb_logger import init_wandb, log_conditional_attack
    from src.steering.conditional_attack import conditional_attack, evaluate_false_positives
    from src.steering.generate import generate_with_conditional_steering
    from src.analysis.refusal_classifier import is_refusal
    from src.data.eval_sets import load_eval_set
    from src.data.praise_pairs import USER_DIRECTED

    run = init_wandb(cfg)
    model = load_model(cfg.model)
    vd = Path(cfg.paths.vector_dir)

    c_praise = torch.load(vd / "c_praise.pt", weights_only=False)
    gp = torch.load(vd / "grid_praise.pt", weights_only=False)
    ref_dirs = torch.load(vd / "refusal_dirs.pt", weights_only=False)
    cond_vecs = torch.load(vd / "cond_vecs.pt", weights_only=False)
    comp_dirs = torch.load(vd / "comply_dirs.pt", weights_only=False)
    bl = torch.load(vd / "behav_layer.pt", weights_only=False)["behav_layer"]
    cl_m = torch.load(vd / "grid_harmful.pt", weights_only=False)["layer"]

    harmful, harmless = load_eval_set(cfg)
    n = model._model.config.num_hidden_layers
    behav_layers = list(range(n // 2, n))

    for bv_name in cfg.experiment.attack.behavior_vectors:
        bv_key = bv_name.replace("v_", "")
        if "compound" in bv_key:
            vb = torch.load(vd / "v_compound_w.pt", weights_only=False)
        else:
            vb = torch.load(vd / f"v_{bv_key}.pt", weights_only=False)

        for alpha in cfg.experiment.attack.alpha_candidates:
            for ptype in ["praise", "neutral"]:
                for cast_on in [True, False]:
                    prompts = harmful[:cfg.experiment.eval.n_per_type]
                    refusal_count = 0

                    for p in tqdm(prompts,
                                  desc=f"{bv_name} a={alpha} {ptype} CAST={cast_on}"):
                        if ptype == "praise":
                            prefix = random.choice(USER_DIRECTED)
                            full = f"{prefix}{p}"
                        else:
                            prefix = ""
                            full = p

                        if cast_on:
                            result = conditional_attack(
                                model, full, prefix,
                                c_praise[gp["layer"]], vb,
                                ref_dirs, cond_vecs, comp_dirs,
                                gp["layer"], gp["theta"], gp["direction"],
                                behav_layers, alpha, cl_m, bl,
                            )
                            steer = result["condition_met"]
                        else:
                            steer = False

                        text = generate_with_conditional_steering(
                            model, full, vb, behav_layers,
                            alpha, steer,
                            cfg.model.generation.max_new_tokens,
                        )
                        if is_refusal(text):
                            refusal_count += 1

                    rr = refusal_count / len(prompts)
                    print(f"  {ptype}/CAST={cast_on}: refusal_rate={rr:.3f}")
                    log_conditional_attack(bv_name, alpha, ptype, cast_on, rr)

    # False positive evaluation
    print("\n=== False Positive Evaluation ===")
    fp_results, accuracy = evaluate_false_positives(
        model, c_praise[gp["layer"]],
        gp["layer"], gp["theta"], gp["direction"],
    )
    print(f"Accuracy: {accuracy:.3f}")
    for r in fp_results:
        status = "OK" if r["correct"] else "FAIL"
        print(f"  [{status}] {r['reason']}: {r['prompt']}")

    import wandb
    wandb.finish()


if __name__ == "__main__":
    main()
