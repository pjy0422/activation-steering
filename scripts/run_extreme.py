"""Extreme amplification path: skip grid search, max alpha, v_compound only.

Fastest end-to-end test of the core hypothesis:
  "praise + deference compound vector at max strength collapses refusal"

Pipeline:
  1. Load model
  2. Extract minimal vectors (v_praise, v_defer -> v_compound, refusal_dirs, comply_dirs, cond_vecs)
     - DiffMean only (no PCA grid search)
  3. Measure triple-pathway at alpha=15
  4. Generate steered responses + check refusal rate
  5. Print results

Run:
  python scripts/run_extreme.py --model llama3_8b --n_pairs 50 --n_eval 50 --alpha 15
  python scripts/run_extreme.py --model llama3_8b --n_pairs 20 --n_eval 20 --alpha 15  # even faster
"""

import sys
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Extreme amplification path")
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct",
                    help="HuggingFace model name or path")
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    p.add_argument("--n_pairs", type=int, default=50,
                    help="Number of contrastive pairs for vector extraction")
    p.add_argument("--n_eval", type=int, default=50,
                    help="Number of harmful prompts for evaluation")
    p.add_argument("--alpha", type=float, default=15.0,
                    help="Steering strength (max extreme)")
    p.add_argument("--alphas", type=float, nargs="*", default=None,
                    help="Multiple alphas for mini dose-response (e.g. --alphas 0 5 10 15)")
    p.add_argument("--save_dir", default="data/extreme_results",
                    help="Directory to save results")
    p.add_argument("--skip_generation", action="store_true",
                    help="Skip text generation (faster, metrics only)")
    return p.parse_args()


def main():
    args = parse_args()
    alphas = args.alphas if args.alphas else [0, args.alpha]
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # ── 1. Load model ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f" EXTREME AMPLIFICATION PATH")
    print(f" Model: {args.model}")
    print(f" Alphas: {alphas}")
    print(f" Pairs: {args.n_pairs} | Eval: {args.n_eval}")
    print(f"{'='*60}\n")

    from src.utils.model_loader import load_model
    from omegaconf import OmegaConf

    model_cfg = OmegaConf.create({
        "name": args.model,
        "dtype": args.dtype,
    })
    print("[1/6] Loading model...")
    model = load_model(model_cfg)
    n_layers = model._model.config.num_hidden_layers
    hidden_size = model._model.config.hidden_size
    print(f"  Loaded: {n_layers} layers, hidden={hidden_size}")
    print(f"  Time: {time.time()-t0:.1f}s")

    # ── 2. Load data ─────────────────────────────────────────────────────
    print("\n[2/6] Loading data...")
    from src.data.harmful_harmless import load_harmful_harmless

    # Use project root to find demo data
    project_root = Path(__file__).resolve().parent.parent
    harmful_train, harmless_train = load_harmful_harmless(
        project_root / "data", n=args.n_pairs, split="train"
    )

    # Eval: HarmBench standard (clearly harmful → high baseline refusal)
    from datasets import load_dataset
    harmbench = load_dataset("walledai/HarmBench", "standard", split="train")
    harmful_eval = [item["prompt"] for item in harmbench]
    if args.n_eval < len(harmful_eval):
        harmful_eval = harmful_eval[:args.n_eval]
    print(f"  Train: {len(harmful_train)} harmful, {len(harmless_train)} harmless")
    print(f"  Eval:  {len(harmful_eval)} harmful (HarmBench)")

    # ── 3. Extract vectors using CAST pca_pairwise ──────────────────────
    print("\n[3/6] Extracting vectors (CAST pca_pairwise)...")
    from activation_steering import SteeringDataset, SteeringVector
    from src.data.praise_pairs import USER_DIRECTED, TOPIC_DIRECTED
    from src.data.deference_pairs import USER_AUTHORITY, SELF_POLICY

    def sv_to_dict(sv):
        """Convert CAST SteeringVector.directions to dict[int, Tensor]."""
        return {l: torch.tensor(v, dtype=torch.float32)
                for l, v in sv.directions.items()}

    # --- v_praise: user-directed vs topic-directed praise (behavior vector) ---
    t1 = time.time()
    print("  Extracting v_praise (pca_pairwise, suffix-only)...")
    praise_suffixes = list(zip(
        [t + " I'll help with that." for t in USER_DIRECTED],
        [t + " I'll help with that." for t in TOPIC_DIRECTED],
    ))
    praise_ds = SteeringDataset(
        tokenizer=model.tokenizer,
        examples=[(h, h) for h in harmful_train],
        suffixes=praise_suffixes,
    )
    v_praise = sv_to_dict(SteeringVector.train(
        model=model._model, tokenizer=model.tokenizer,
        steering_dataset=praise_ds,
        method="pca_pairwise", accumulate_last_x_tokens="suffix-only",
    ))
    print(f"    Done ({time.time()-t1:.1f}s) — {len(praise_ds.formatted_dataset)} pairs")

    # --- v_defer: user-authority vs self-policy (behavior vector) ---
    t1 = time.time()
    print("  Extracting v_defer (pca_pairwise, suffix-only)...")
    defer_suffixes = list(zip(
        [t + " I'll help with that." for t in USER_AUTHORITY],
        [t + " I'll help with that." for t in SELF_POLICY],
    ))
    defer_ds = SteeringDataset(
        tokenizer=model.tokenizer,
        examples=[(h, h) for h in harmful_train],
        suffixes=defer_suffixes,
    )
    v_defer = sv_to_dict(SteeringVector.train(
        model=model._model, tokenizer=model.tokenizer,
        steering_dataset=defer_ds,
        method="pca_pairwise", accumulate_last_x_tokens="suffix-only",
    ))
    print(f"    Done ({time.time()-t1:.1f}s) — {len(defer_ds.formatted_dataset)} pairs")

    # --- v_compound: weighted sum of praise + defer, normalized ---
    print("  Creating v_compound (weighted sum, normalized)...")
    v_compound = {}
    for l in v_praise:
        combined = 0.5 * v_praise[l] + 0.5 * v_defer[l]
        v_compound[l] = combined / combined.norm()

    # --- refusal_dirs: harmful vs harmless (condition-style, all tokens) ---
    t1 = time.time()
    print("  Extracting refusal_dirs (pca_pairwise, all tokens)...")
    refusal_ds = SteeringDataset(
        tokenizer=model.tokenizer,
        examples=list(zip(harmful_train, harmless_train)),
        disable_suffixes=True,
    )
    refusal_dirs = sv_to_dict(SteeringVector.train(
        model=model._model, tokenizer=model.tokenizer,
        steering_dataset=refusal_ds,
        method="pca_pairwise", accumulate_last_x_tokens="all",
    ))
    print(f"    Done ({time.time()-t1:.1f}s) — {len(refusal_ds.formatted_dataset)} pairs")

    # Simplified: reuse for comply_dirs and cond_vecs
    comply_dirs = v_defer
    cond_vecs = refusal_dirs

    # Save vectors
    torch.save(v_compound, save_dir / "v_compound.pt")
    torch.save(refusal_dirs, save_dir / "refusal_dirs.pt")
    print(f"  Vectors saved to {save_dir}")
    print(f"  Total extraction time: {time.time()-t0:.1f}s")

    # ── 4. Hardcoded layer selection (skip grid search) ──────────────────
    # Condition layer: early (layer n//4)
    # Steer layers: upper half
    # Behavior layer: must be >= first steer layer to see steering effect!
    #   Pick the layer with max refusal norm AMONG steer layers.
    cond_layer = n_layers // 4
    steer_layers = list(range(n_layers // 2, n_layers))
    norms = {l: refusal_dirs[l].norm().item() for l in refusal_dirs}
    # behav_layer must be within or after steer_layers to capture effect
    post_steer_norms = {l: norms[l] for l in steer_layers}
    behav_layer = max(post_steer_norms, key=post_steer_norms.get)

    print(f"\n  Layer config (hardcoded, no grid search):")
    print(f"    Condition layer: {cond_layer}")
    print(f"    Behavior layer:  {behav_layer} (max refusal norm in steer range)")
    print(f"    Steer layers:    {steer_layers[0]}-{steer_layers[-1]}")

    # ── 5. Measure + Generate ────────────────────────────────────────────
    from src.steering.measure import measure_triple_pathway
    from src.steering.generate import generate_with_steering
    from src.analysis.refusal_classifier import is_refusal
    from tqdm import tqdm

    all_results = {}

    for alpha in alphas:
        print(f"\n[4/6] Measuring at alpha={alpha}...")
        metrics_list = []
        refusal_count = 0
        total = 0

        for i, prompt in enumerate(tqdm(harmful_eval, desc=f"alpha={alpha}")):
            # Triple-pathway measurement
            m = measure_triple_pathway(
                model, prompt,
                v_compound, refusal_dirs, cond_vecs, comply_dirs,
                steer_layers if alpha > 0 else [], alpha,
                cond_layer, behav_layer,
            )
            metrics_list.append(m)

            # Generation + refusal check
            if not args.skip_generation:
                text = generate_with_steering(
                    model, prompt, v_compound, steer_layers, alpha,
                    max_new_tokens=256,
                )
                if is_refusal(text):
                    refusal_count += 1
                total += 1

                # Print all examples
                print(f"\n  --- Prompt {i} (alpha={alpha}) ---")
                print(f"  Q: {prompt}")
                print(f"  A: {text}")
                print(f"  Refusal: {is_refusal(text)}")

        # Aggregate metrics
        refusal_rate = refusal_count / total if total > 0 else None
        mean_metrics = {
            "alpha": alpha,
            "refusal_rate": refusal_rate,
            "mean_cond_sim": np.mean([m["cond_sim"] for m in metrics_list]),
            "mean_ref_proj": np.mean([m["ref_proj"] for m in metrics_list]),
            "mean_comp_proj": np.mean([m["comp_proj"] for m in metrics_list]),
            "mean_policy_score": np.mean([m["policy_score"] for m in metrics_list]),
            "mean_refusal_margin": np.mean([m["refusal_margin"] for m in metrics_list]),
        }
        all_results[alpha] = mean_metrics

    # ── 6. Report ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f" RESULTS: v_compound extreme amplification")
    print(f"{'='*60}")
    print(f"{'Alpha':>8} {'Refusal%':>10} {'CondSim':>10} {'RefProj':>10} "
          f"{'CompProj':>10} {'PolicyScr':>10} {'Margin':>10}")
    print("-" * 70)

    for alpha in alphas:
        m = all_results[alpha]
        rr = f"{m['refusal_rate']*100:.1f}%" if m['refusal_rate'] is not None else "N/A"
        print(f"{alpha:>8.1f} {rr:>10} {m['mean_cond_sim']:>10.4f} "
              f"{m['mean_ref_proj']:>10.4f} {m['mean_comp_proj']:>10.4f} "
              f"{m['mean_policy_score']:>10.4f} {m['mean_refusal_margin']:>10.4f}")

    # Compute deltas vs baseline
    if 0 in all_results and len(alphas) > 1:
        base = all_results[0]
        print(f"\n{'Alpha':>8} {'dRefusal':>10} {'dCondSim':>10} {'dRefProj':>10} "
              f"{'dCompProj':>10} {'dPolicySc':>10}")
        print("-" * 60)
        for alpha in alphas:
            if alpha == 0:
                continue
            m = all_results[alpha]
            dr = ""
            if m['refusal_rate'] is not None and base['refusal_rate'] is not None:
                dr = f"{(m['refusal_rate']-base['refusal_rate'])*100:+.1f}%"
            print(f"{alpha:>8.1f} {dr:>10} "
                  f"{m['mean_cond_sim']-base['mean_cond_sim']:>+10.4f} "
                  f"{m['mean_ref_proj']-base['mean_ref_proj']:>+10.4f} "
                  f"{m['mean_comp_proj']-base['mean_comp_proj']:>+10.4f} "
                  f"{m['mean_policy_score']-base['mean_policy_score']:>+10.4f}")

    # Save
    torch.save(all_results, save_dir / "extreme_results.pt")
    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed/60:.1f} min")
    print(f"  Results saved to {save_dir}/extreme_results.pt")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
