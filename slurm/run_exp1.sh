#!/bin/bash

# --- [SBATCH options] ---
#SBATCH --job-name=SRC_exp1_dose
#SBATCH --partition=RTX6000ADA,L40S
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

export USE_HF=1
export HF_ENDPOINT=https://huggingface.co
source ~/activation-steering/bin/activate
cd /home2/pjy0422/workspace/activation-steering

MODEL="${1:-llama3_8b}"
DATA="${2:-default}"

echo "============================================="
echo " Experiment 1: Dose-Response (Method A)"
echo " 5 vectors x 7 alphas x 200 prompts"
echo " Model: $MODEL | Data: $DATA"
echo " Job: $SLURM_JOB_ID | Node: $SLURM_NODELIST"
echo "============================================="

python scripts/run_dose_response.py \
    model=$MODEL \
    experiment=dose_response \
    data=$DATA

echo "===== Experiment 1 complete ====="
