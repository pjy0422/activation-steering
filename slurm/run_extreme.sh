#!/bin/bash

# --- [SBATCH options] ---
#SBATCH --job-name=SRC_extreme
#SBATCH --partition=RTX6000ADA,L40S
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

export USE_HF=1
export HF_ENDPOINT=https://huggingface.co
source ~/activation-steering/bin/activate
cd /home2/pjy0422/workspace/activation-steering

# Ensure data directory has condition_harmful.json
if [ ! -f "data/raw/condition_harmful.json" ]; then
    cp docs/demo-data/condition_harmful.json data/raw/
fi

echo "============================================="
echo " EXTREME AMPLIFICATION PATH"
echo " v_compound, finer alpha range, no grid search"
echo " Job: $SLURM_JOB_ID | Node: $SLURM_NODELIST"
echo "============================================="

# Full run: 50 pairs extraction, 50 eval prompts
# alpha 5+ produces gibberish, so focus on coherent range
python scripts/run_extreme.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --n_pairs 50 \
    --n_eval 50 \
    --alphas 0 0.5 1 1.5 2 2.5 3 4

echo "===== Extreme run complete ====="
