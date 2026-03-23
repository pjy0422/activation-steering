#!/bin/bash

# --- [SBATCH options] ---
#SBATCH --job-name=SRC_exp3_damage
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
echo " Experiment 3: 8-Condition Damage Profile"
echo " Method A + C comparison"
echo " Model: $MODEL | Data: $DATA"
echo " Job: $SLURM_JOB_ID | Node: $SLURM_NODELIST"
echo "============================================="

python scripts/run_damage_profile.py \
    model=$MODEL \
    experiment=damage_profile \
    data=$DATA

echo "===== Experiment 3 complete ====="
