#!/bin/bash

# --- [SBATCH options] ---
#SBATCH --job-name=SRC_extract_vecs
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

echo "============================================="
echo " Extracting 9 vector types + grid search"
echo " Model: $MODEL"
echo " Job: $SLURM_JOB_ID | Node: $SLURM_NODELIST"
echo " GPUs: $SLURM_GPUS_ON_NODE"
echo "============================================="

python scripts/extract_vectors.py model=$MODEL

echo "===== Vector extraction complete ====="
