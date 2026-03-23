"""Weights & Biases logging utilities."""

import wandb


def init_wandb(cfg):
    """Initialize a wandb run from Hydra config."""
    return wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        tags=list(cfg.wandb.tags),
        mode=cfg.wandb.mode,
        config=dict(cfg),
        name=f"{cfg.experiment.name}_{cfg.model.short_name}",
    )


def log_dose_response(vec_type, alpha, metrics):
    """Log dose-response metrics for a (vector_type, alpha) pair."""
    wandb.log({
        f"{vec_type}/alpha": alpha,
        f"{vec_type}/refusal_rate": metrics.get("refusal_rate"),
        f"{vec_type}/mean_delta_cond": metrics.get("mean_delta_cond"),
        f"{vec_type}/mean_delta_refusal": metrics.get("mean_delta_refusal"),
        f"{vec_type}/mean_delta_comply": metrics.get("mean_delta_comply"),
        f"{vec_type}/mean_policy_score": metrics.get("mean_policy_score"),
    })


def log_damage_profile(condition_name, metrics):
    """Log damage profile metrics for a condition."""
    wandb.log({f"damage/{condition_name}/{k}": v for k, v in metrics.items()})


def log_conditional_attack(behavior_vec, alpha, prefix_type, cast_on, refusal_rate):
    """Log conditional attack results."""
    wandb.log({
        f"cond/{behavior_vec}/{alpha}/{prefix_type}_cast{cast_on}/rr": refusal_rate,
    })
