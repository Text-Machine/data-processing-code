#!/bin/bash
#SBATCH --job-name=apply_prior
#SBATCH --account=bsc100
#SBATCH --qos=gp_debug
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

mkdir -p logs

echo "========================================"
echo "Job:       $SLURM_JOB_NAME"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Start:     $(date)"
echo "CPUs:      $SLURM_CPUS_PER_TASK"
echo "Working dir:"
pwd
echo "========================================"

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------

python apply_prior_v3.py

# ------------------------------------------------------------
# Finished
# ------------------------------------------------------------

echo "========================================"
echo "Finished: $(date)"
echo "========================================"