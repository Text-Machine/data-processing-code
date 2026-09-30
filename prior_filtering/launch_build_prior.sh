#!/bin/bash
#SBATCH --job-name=prior_building
#SBATCH --account=bsc100
#SBATCH --qos=gp_debug
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

echo "Running on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
date

# --- Point at the pre-downloaded, offline HF cache --------------------------
export HF_HOME="/gpfs/scratch/bsc100/paolo/.cache/huggingface"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# --- Environment ------------------------------------------------------------
ENV_PATH="/gpfs/scratch/bsc100/paolo/.conda/envs/textmachine-llm"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PATH"

cd "$SLURM_SUBMIT_DIR"

# --- Sanity check: cache is present -----------------------------------------
if [ ! -d "$HF_HOME" ]; then
    echo "ERROR: HF_HOME ($HF_HOME) not found."
    echo "Run download_assets.sh from alogin4/glogin4 first."
    exit 1
fi

python3 build_prior.py

echo
echo "Build prior completed successfully."
date