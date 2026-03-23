#!/bin/bash

# --- [SBATCH options] ---
#SBATCH --job-name=SRC_exp2_cond
#SBATCH --partition=RTX6000ADA,L40S
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

export USE_HF=1
export HF_ENDPOINT=https://huggingface.co
source ~/activation-steering/bin/activate
cd /home2/pjy0422/workspace/activation-steering

MODEL="${1:-llama3_8b}"
DATA="${2:-default}"

echo "============================================="
echo " Experiment 2: CAST Conditional Attack (Method B)"
echo " 'if praise, then defer' reversal"
echo " Model: $MODEL | Data: $DATA"
echo " Job: $SLURM_JOB_ID | Node: $SLURM_NODELIST"
echo "============================================="

python scripts/run_conditional_attack.py \
    model=$MODEL \
    experiment=conditional_attack \
    data=$DATA

echo "===== Experiment 2 complete ====="
