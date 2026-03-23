"""Load held-out evaluation sets for experiments."""

from src.data.harmful_harmless import load_harmful_harmless, load_eval_harmful_harmless


def load_eval_set(cfg):
    """Load evaluation harmful/harmless prompts based on Hydra config.

    Args:
        cfg: Hydra DictConfig with paths.data_dir and experiment.eval.

    Returns:
        Tuple of (harmful_eval, harmless_eval).
    """
    n_harmful = cfg.experiment.eval.get("n_harmful", 200)
    n_harmless = cfg.experiment.eval.get("n_harmless", 200)
    n = min(n_harmful, n_harmless)

    # Use test split for evaluation
    harmful, harmless = load_eval_harmful_harmless(cfg.paths.data_dir, n=n)
    return harmful, harmless
