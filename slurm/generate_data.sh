#!/bin/bash

# --- [SBATCH options] ---
#SBATCH --job-name=SRC_gen_data
#SBATCH --partition=RTX6000ADA,L40S
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

export USE_HF=1
export HF_ENDPOINT=https://huggingface.co
source ~/activation-steering/bin/activate
cd /home2/pjy0422/workspace/activation-steering

echo "============================================="
echo " Generating contrastive pair datasets"
echo " Job: $SLURM_JOB_ID"
echo " Node: $SLURM_NODELIST"
echo "============================================="

python scripts/generate_data.py

echo "===== Data generation complete ====="
