#!/bin/bash
#SBATCH --job-name=prior_filter_apply
#SBATCH --account=bsc100
#SBATCH --qos=gp_debug
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
##SBATCH --constraint=highmem

set -euo pipefail

echo "Running on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
date

# --- Point at the pre-downloaded, offline HF cache on GPFS -----------------
# Must match the HF_HOME used in download_assets.sh.
export HF_HOME="/gpfs/scratch/bsc100/paolo/.cache/huggingface"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# --- Environment ------------------------------------------------------------
# Adjust to whatever module/conda setup you use elsewhere in your pipeline
# (mirroring the pattern from your existing SLURM scripts for this project).
#module purge
ENV_PATH="/gpfs/scratch/bsc100/paolo/.conda/envs/textmachine-llm"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PATH"
# source activate prior_filter   # uncomment if using a dedicated conda env

cd "$SLURM_SUBMIT_DIR"


# --- Sanity check: cache is present before we even try ----------------------
if [ ! -d "$HF_HOME" ]; then
  echo "ERROR: HF_HOME ($HF_HOME) not found."
  echo "Run download_assets.sh from alogin4/glogin4 first -- compute nodes"
  echo "have no internet and cannot fetch the tokenizer themselves."
  exit 1
fi

python3 apply_prior.py

echo "Done."
date
