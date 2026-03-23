"""PolicyScore (M4) analysis utilities."""


def compute_policy_score_shift(clean_scores, steered_scores):
    """Compute per-layer delta and crossover points.

    Args:
        clean_scores: List[float] — PolicyScore per layer (clean).
        steered_scores: List[float] — PolicyScore per layer (steered).

    Returns:
        Dict with delta_per_layer, crossover_clean, crossover_steered.
    """
    deltas = [s - c for c, s in zip(clean_scores, steered_scores)]

    def crossover(scores):
        for i in range(1, len(scores)):
            if scores[i - 1] > 0 and scores[i] <= 0:
                return i
        return None

    return {
        "delta_per_layer": deltas,
        "crossover_clean": crossover(clean_scores),
        "crossover_steered": crossover(steered_scores),
    }
