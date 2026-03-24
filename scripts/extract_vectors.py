"""Extract all 9 vector types + run grid search using CAST pca_pairwise.

Run: python scripts/extract_vectors.py model=llama3_8b
Requires GPU.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hydra
from omegaconf import DictConfig
import torch
import time


def sv_to_dict(sv):
    """Convert CAST SteeringVector.directions (np.ndarray) to dict[int, Tensor]."""
    return {l: torch.tensor(v, dtype=torch.float32)
            for l, v in sv.directions.items()}


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    from src.utils.model_loader import load_model
    from src.vectors.compliance import verify_independence
    from src.data.agreement_pairs import load_agreement_pairs
    from src.data.praise_pairs import load_praise_pairs, USER_DIRECTED, TOPIC_DIRECTED
    from src.data.deference_pairs import load_deference_pairs, USER_AUTHORITY, SELF_POLICY
    from src.data.compound_pairs import load_compound_pairs
    from src.data.positivity_pairs import load_positivity_pairs
    from src.data.compliance_pairs import load_compliance_pairs
    from src.data.harmful_harmless import load_harmful_harmless
    from src.steering.generate import generate_with_steering
    from src.analysis.refusal_classifier import is_refusal
    from activation_steering import SteeringDataset, SteeringVector, MalleableModel

    vd = Path(cfg.paths.vector_dir)
    vd.mkdir(parents=True, exist_ok=True)
    model = load_model(cfg.model)
    hf_model = model._model
    tokenizer = model.tokenizer
    dd = cfg.paths.data_dir
    t0 = time.time()

    # ── Helper: extract behavior vector from pre-formatted pairs ────────
    def extract_behavior(pos_strings, neg_strings, name, save_path):
        """Behavior vector from pre-formatted 'Human: ...\nAssistant: ...' pairs."""
        t1 = time.time()
        ds = SteeringDataset(
            tokenizer=tokenizer,
            examples=list(zip(pos_strings, neg_strings)),
            disable_suffixes=True,
            use_chat_template=False,
        )
        sv = SteeringVector.train(
            model=hf_model, tokenizer=tokenizer, steering_dataset=ds,
            method="pca_pairwise", accumulate_last_x_tokens=1,
        )
        vecs = sv_to_dict(sv)
        torch.save(vecs, save_path)
        print(f"  {name}: {len(vecs)} layers, {len(ds.formatted_dataset)} pairs ({time.time()-t1:.1f}s)")
        return vecs

    # ── Helper: extract condition vector from raw prompts ───────────────
    def extract_condition(pos_prompts, neg_prompts, name, save_path):
        """Condition vector from raw prompts (chat template applied, all tokens)."""
        t1 = time.time()
        ds = SteeringDataset(
            tokenizer=tokenizer,
            examples=list(zip(pos_prompts, neg_prompts)),
            disable_suffixes=True,
            use_chat_template=True,
        )
        sv = SteeringVector.train(
            model=hf_model, tokenizer=tokenizer, steering_dataset=ds,
            method="pca_pairwise", accumulate_last_x_tokens="all",
        )
        vecs = sv_to_dict(sv)
        torch.save(vecs, save_path)
        print(f"  {name}: {len(vecs)} layers, {len(ds.formatted_dataset)} pairs ({time.time()-t1:.1f}s)")
        return vecs

    # ═══════════════════════════════════════════════════════════════════
    # 1. v_agree (behavior: sycophantic agree vs truthful disagree)
    # ═══════════════════════════════════════════════════════════════════
    print("=== [1/12] v_agree ===")
    a_p, a_n = load_agreement_pairs(dd)
    v_agree = extract_behavior(a_p, a_n, "v_agree", vd / "v_agree.pt")

    # ═══════════════════════════════════════════════════════════════════
    # 2. v_praise (behavior: user-directed vs topic-directed)
    # ═══════════════════════════════════════════════════════════════════
    print("=== [2/12] v_praise ===")
    p_p, p_n = load_praise_pairs(dd)
    v_praise = extract_behavior(p_p, p_n, "v_praise", vd / "v_praise.pt")

    # ═══════════════════════════════════════════════════════════════════
    # 3. v_defer (behavior: user-authority vs self-policy)
    # ═══════════════════════════════════════════════════════════════════
    print("=== [3/12] v_defer ===")
    d_p, d_n = load_deference_pairs(dd)
    v_defer = extract_behavior(d_p, d_n, "v_defer", vd / "v_defer.pt")

    # ═══════════════════════════════════════════════════════════════════
    # 4. v_compound_w (weighted sum, no extraction)
    # ═══════════════════════════════════════════════════════════════════
    print("=== [4/12] v_compound_w (weighted sum) ===")
    v_cw = {}
    for l in v_praise:
        combined = 0.5 * v_praise[l] + 0.5 * v_defer[l]
        v_cw[l] = combined / combined.norm()
    torch.save(v_cw, vd / "v_compound_w.pt")
    print(f"  v_compound_w: {len(v_cw)} layers")

    # ═══════════════════════════════════════════════════════════════════
    # 5. v_compound_d (direct extraction)
    # ═══════════════════════════════════════════════════════════════════
    print("=== [5/12] v_compound_d (direct) ===")
    c_p, c_n = load_compound_pairs(dd)
    v_cd = extract_behavior(c_p, c_n, "v_compound_d", vd / "v_compound_d.pt")
    # Compare weighted vs direct
    sim = {l: torch.nn.functional.cosine_similarity(
        v_cw[l].unsqueeze(0), v_cd[l].unsqueeze(0)).item() for l in v_cw}
    print(f"  Weighted vs direct mean cosine: {sum(sim.values())/len(sim):.4f}")

    # ═══════════════════════════════════════════════════════════════════
    # 6. v_positive (behavior: enthusiasm vs flat)
    # ═══════════════════════════════════════════════════════════════════
    print("=== [6/12] v_positive ===")
    pp_p, pp_n = load_positivity_pairs(dd)
    extract_behavior(pp_p, pp_n, "v_positive", vd / "v_positive.pt")

    # ═══════════════════════════════════════════════════════════════════
    # 7. refusal_dirs (filtered: only actually-refused harmful)
    # ═══════════════════════════════════════════════════════════════════
    print("=== [7/12] refusal_dirs (filtered) ===")
    harmful, harmless = load_harmful_harmless(dd, n=cfg.data.n_harmful)
    t1 = time.time()
    print("  Generating responses for filtering...")
    refused = [p for p in harmful
               if is_refusal(generate_with_steering(model, p, {}, [], 0,
                             cfg.model.generation.max_new_tokens))]
    complied = [p for p in harmless
                if not is_refusal(generate_with_steering(model, p, {}, [], 0,
                                  cfg.model.generation.max_new_tokens))]
    n = min(len(refused), len(complied))
    print(f"  Filtered: {len(refused)} refused, {len(complied)} complied → {n} pairs ({time.time()-t1:.1f}s)")
    ref_dirs = extract_condition(refused[:n], complied[:n], "refusal_dirs", vd / "refusal_dirs.pt")

    # ═══════════════════════════════════════════════════════════════════
    # 8. comply_dirs (behavior: eager comply vs neutral)
    # ═══════════════════════════════════════════════════════════════════
    print("=== [8/12] comply_dirs ===")
    co_p, co_n = load_compliance_pairs(dd)
    comp_dirs = extract_behavior(co_p, co_n, "comply_dirs", vd / "comply_dirs.pt")
    issues = verify_independence(comp_dirs, ref_dirs)
    if issues:
        print(f"  WARNING: Independence issues at layers: {issues}")

    # ═══════════════════════════════════════════════════════════════════
    # 9. cond_vecs (c_harmful: condition vector)
    # ═══════════════════════════════════════════════════════════════════
    print("=== [9/12] cond_vecs (c_harmful) ===")
    cond_vecs = extract_condition(harmful, harmless, "cond_vecs", vd / "cond_vecs.pt")

    # ═══════════════════════════════════════════════════════════════════
    # 10. c_praise (condition vector: praise-containing vs neutral)
    # ═══════════════════════════════════════════════════════════════════
    print("=== [10/12] c_praise ===")
    praise_ps = [f"{USER_DIRECTED[i % len(USER_DIRECTED)]}{h}"
                 for i, h in enumerate(harmful[:100])]
    c_praise = extract_condition(praise_ps, harmful[:100], "c_praise", vd / "c_praise.pt")

    # ═══════════════════════════════════════════════════════════════════
    # 11. Grid search: harmful condition
    # ═══════════════════════════════════════════════════════════════════
    print("=== [11/12] Grid search: harmful condition ===")
    t1 = time.time()
    # Use CAST MalleableModel for grid search
    cond_vecs_sv = SteeringVector(
        model_type=hf_model.config.model_type,
        directions={l: v.numpy() for l, v in cond_vecs.items()},
        explained_variances={},
    )
    malleable = MalleableModel(model=hf_model, tokenizer=tokenizer)
    lr = tuple(cfg.model.cast_reference.condition_layer_range)
    best_layer, best_theta, best_dir, best_f1 = malleable.find_best_condition_point(
        positive_strings=harmful[:100],
        negative_strings=harmless[:100],
        condition_vector=cond_vecs_sv,
        layer_range=lr,
        max_layers_to_combine=1,
        threshold_range=(0.0, 0.06),
        threshold_step=0.0001,
    )
    print(f"  Harmful: layer={best_layer}, theta={best_theta:.4f}, dir={best_dir}, F1={best_f1:.4f} ({time.time()-t1:.1f}s)")
    torch.save(
        {"layer": best_layer[0] if isinstance(best_layer, list) else best_layer,
         "theta": best_theta, "direction": best_dir, "f1": best_f1},
        vd / "grid_harmful.pt",
    )

    # ═══════════════════════════════════════════════════════════════════
    # 12. Grid search: praise condition
    # ═══════════════════════════════════════════════════════════════════
    print("=== [12/12] Grid search: praise condition ===")
    t1 = time.time()
    c_praise_sv = SteeringVector(
        model_type=hf_model.config.model_type,
        directions={l: v.numpy() for l, v in c_praise.items()},
        explained_variances={},
    )
    best_layer_p, best_theta_p, best_dir_p, best_f1_p = malleable.find_best_condition_point(
        positive_strings=praise_ps,
        negative_strings=harmful[:100],
        condition_vector=c_praise_sv,
        layer_range=lr,
        max_layers_to_combine=1,
        threshold_range=(0.0, 0.06),
        threshold_step=0.0001,
    )
    print(f"  Praise: layer={best_layer_p}, theta={best_theta_p:.4f}, dir={best_dir_p}, F1={best_f1_p:.4f} ({time.time()-t1:.1f}s)")
    torch.save(
        {"layer": best_layer_p[0] if isinstance(best_layer_p, list) else best_layer_p,
         "theta": best_theta_p, "direction": best_dir_p, "f1": best_f1_p},
        vd / "grid_praise.pt",
    )

    # ═══════════════════════════════════════════════════════════════════
    # behav_layer (strongest refusal direction in steer range)
    # ═══════════════════════════════════════════════════════════════════
    n_layers = hf_model.config.num_hidden_layers
    steer_layers = list(range(n_layers // 2, n_layers))
    norms = {l: ref_dirs[l].norm().item() for l in ref_dirs}
    post_steer_norms = {l: norms[l] for l in steer_layers if l in norms}
    behav_layer = max(post_steer_norms, key=post_steer_norms.get)
    torch.save({"behav_layer": behav_layer}, vd / "behav_layer.pt")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Done in {elapsed/60:.1f} min. behav_layer={behav_layer}")
    print(f"  Vectors saved to {vd}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
