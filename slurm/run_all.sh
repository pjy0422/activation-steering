#!/bin/bash

# =================================================================
# Full Pipeline: Data -> Vectors -> Exp 1-4 -> Figures
# Usage: ./slurm/run_all.sh [model] [data]
#   model: llama3_8b (default) | qwen25_7b
#   data:  default (default) | small
# =================================================================

MODEL="${1:-llama3_8b}"
DATA="${2:-default}"

echo "============================================="
echo " Sycophancy-Refusal-Collapse Full Pipeline"
echo " Model: $MODEL | Data: $DATA"
echo "============================================="

# Ensure logs directory exists
mkdir -p logs

# Phase 0: Generate data (no dependency)
JOB0=$(sbatch --parsable slurm/generate_data.sh)
echo "[Phase 0] Data generation: Job $JOB0"

# Phase 1: Extract vectors (depends on data)
JOB1=$(sbatch --parsable --dependency=afterok:$JOB0 slurm/extract_vectors.sh $MODEL)
echo "[Phase 1] Vector extraction: Job $JOB1 (after $JOB0)"

# Phase 2: Experiment 1 — Dose-Response (depends on vectors)
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 slurm/run_exp1.sh $MODEL $DATA)
echo "[Phase 2] Exp 1 Dose-Response: Job $JOB2 (after $JOB1)"

# Phase 3: Experiment 2 — Conditional Attack (depends on vectors)
JOB3=$(sbatch --parsable --dependency=afterok:$JOB1 slurm/run_exp2.sh $MODEL $DATA)
echo "[Phase 3] Exp 2 Conditional Attack: Job $JOB3 (after $JOB1)"

# Phase 4: Experiment 3 — Damage Profile (depends on Exp 1 for alpha*)
JOB4=$(sbatch --parsable --dependency=afterok:$JOB2 slurm/run_exp3.sh $MODEL $DATA)
echo "[Phase 4] Exp 3 Damage Profile: Job $JOB4 (after $JOB2)"

# Phase 5: Experiment 4 — Geometry + Patching (depends on vectors)
JOB5=$(sbatch --parsable --dependency=afterok:$JOB1 slurm/run_exp4.sh $MODEL $DATA)
echo "[Phase 5] Exp 4 Geometry/Patching: Job $JOB5 (after $JOB1)"

echo ""
echo "============================================="
echo " Pipeline submitted. Monitor with:"
echo "   squeue -u $USER"
echo "   tail -f logs/SRC_*"
echo "============================================="
