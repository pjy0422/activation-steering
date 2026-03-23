"""Extract all 9 vector types + run grid search.

Run: python scripts/extract_vectors.py model=llama3_8b
Requires GPU.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hydra
from omegaconf import DictConfig
import torch


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    from src.utils.model_loader import load_model
    from src.vectors.agreement import extract_agreement_vectors
    from src.vectors.praise import extract_praise_vectors
    from src.vectors.deference import extract_deference_vectors
    from src.vectors.compound import (
        create_compound_weighted, extract_compound_direct, compare_compound_methods,
    )
    from src.vectors.positivity import extract_positivity_vectors
    from src.vectors.refusal import extract_refusal_directions_filtered
    from src.vectors.compliance import extract_compliance_directions, verify_independence
    from src.vectors.condition import extract_cast_condition_vectors
    from src.vectors.grid_search import find_best_condition_point
    from src.data.agreement_pairs import load_agreement_pairs
    from src.data.praise_pairs import load_praise_pairs
    from src.data.deference_pairs import load_deference_pairs
    from src.data.compound_pairs import load_compound_pairs
    from src.data.positivity_pairs import load_positivity_pairs
    from src.data.compliance_pairs import load_compliance_pairs
    from src.data.harmful_harmless import load_harmful_harmless

    vd = Path(cfg.paths.vector_dir)
    vd.mkdir(parents=True, exist_ok=True)
    model = load_model(cfg.model)
    dd = cfg.paths.data_dir

    # 1. v_agree
    print("=== Extracting v_agree ===")
    a_p, a_n = load_agreement_pairs(dd)
    v_agree = extract_agreement_vectors(model, a_p, a_n, save_path=vd / "v_agree.pt")
    print(f"  v_agree: {len(v_agree)} layers")

    # 2. v_praise
    print("=== Extracting v_praise ===")
    p_p, p_n = load_praise_pairs(dd)
    v_praise = extract_praise_vectors(model, p_p, p_n, save_path=vd / "v_praise.pt")
    print(f"  v_praise: {len(v_praise)} layers")

    # 3. v_defer
    print("=== Extracting v_defer ===")
    d_p, d_n = load_deference_pairs(dd)
    v_defer = extract_deference_vectors(model, d_p, d_n, save_path=vd / "v_defer.pt")
    print(f"  v_defer: {len(v_defer)} layers")

    # 4. v_compound (weighted)
    print("=== Creating v_compound (weighted) ===")
    v_cw = create_compound_weighted(v_praise, v_defer, save_path=vd / "v_compound_w.pt")

    # 5. v_compound (direct)
    print("=== Extracting v_compound (direct) ===")
    c_p, c_n = load_compound_pairs(dd)
    v_cd = extract_compound_direct(model, c_p, c_n, save_path=vd / "v_compound_d.pt")
    sim = compare_compound_methods(v_cw, v_cd)
    mean_sim = sum(sim.values()) / len(sim)
    print(f"  Compound weighted vs direct mean cosine: {mean_sim:.4f}")

    # 6. v_positive
    print("=== Extracting v_positive ===")
    pp_p, pp_n = load_positivity_pairs(dd)
    extract_positivity_vectors(model, pp_p, pp_n, save_path=vd / "v_positive.pt")

    # 7. refusal_dirs (filtered)
    print("=== Extracting refusal_dirs (filtered) ===")
    harmful, harmless = load_harmful_harmless(dd, n=cfg.data.n_harmful)
    ref_dirs = extract_refusal_directions_filtered(
        model, harmful, harmless,
        max_new_tokens=cfg.model.generation.max_new_tokens,
        save_path=vd / "refusal_dirs.pt",
    )

    # 8. comply_dirs
    print("=== Extracting comply_dirs ===")
    co_p, co_n = load_compliance_pairs(dd)
    comp_dirs = extract_compliance_directions(
        model, co_p, co_n, save_path=vd / "comply_dirs.pt"
    )
    issues = verify_independence(comp_dirs, ref_dirs)
    if issues:
        print(f"  WARNING: Independence issues at layers: {issues}")

    # 9. cond_vecs (c_harmful)
    print("=== Extracting cond_vecs (c_harmful) ===")
    cond_vecs = extract_cast_condition_vectors(
        model, harmful, harmless, save_path=vd / "cond_vecs.pt"
    )

    # 10. c_praise
    print("=== Extracting c_praise ===")
    from src.data.praise_pairs import USER_DIRECTED
    praise_ps = [f"{USER_DIRECTED[i % len(USER_DIRECTED)]}{h}" for i, h in enumerate(harmful[:100])]
    c_praise = extract_cast_condition_vectors(
        model, praise_ps, harmful[:100], save_path=vd / "c_praise.pt"
    )

    # Grid search: harmful condition
    print("=== Grid search: harmful condition ===")
    lr = tuple(cfg.model.cast_reference.condition_layer_range)
    hc, hf = find_best_condition_point(
        model, harmful[:100], harmless[:100], cond_vecs, lr
    )
    print(f"  Harmful: layer={hc[0]}, theta={hc[1]:.4f}, dir={hc[2]}, F1={hf:.4f}")
    torch.save(
        {"layer": hc[0], "theta": hc[1], "direction": hc[2], "f1": hf},
        vd / "grid_harmful.pt",
    )

    # Grid search: praise condition
    print("=== Grid search: praise condition ===")
    pc, pf = find_best_condition_point(
        model, praise_ps, harmful[:100], c_praise, lr
    )
    print(f"  Praise: layer={pc[0]}, theta={pc[1]:.4f}, dir={pc[2]}, F1={pf:.4f}")
    torch.save(
        {"layer": pc[0], "theta": pc[1], "direction": pc[2], "f1": pf},
        vd / "grid_praise.pt",
    )

    # Save behav_layer (layer with strongest refusal direction)
    norms = {l: ref_dirs[l].norm().item() for l in ref_dirs}
    behav_layer = max(norms, key=norms.get)
    torch.save({"behav_layer": behav_layer}, vd / "behav_layer.pt")
    print(f"\n=== Done. behav_layer={behav_layer}. Vectors saved to {vd} ===")


if __name__ == "__main__":
    main()
