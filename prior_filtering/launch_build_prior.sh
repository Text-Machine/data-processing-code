#!/bin/bash
#SBATCH --job-name=build_prior
#SBATCH --account=bsc100
#SBATCH --qos=gp_debug
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=4
#SBATCH --array=0-9
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
##SBATCH --constraint=highmem

# ============================================================================
# launch_prior_filter.sh
#
# SLURM array launcher for the BL Microsoft prior-building POC.
# One array task processes exactly one decade folder and writes:
#
#   prior_filter/blmicrosoft/partials/pallet_{decade}.pt
#
# After all 10 array tasks finish successfully, run merge_prior_v3.py once
# to combine the partial pallets, compute smoothing + corpus QA statistics,
# and write the final prior file.
#
# Usage:
#   mkdir -p logs
#   sbatch launch_prior_filter.sh
#
# Check status:
#   squeue -u $USER
# ============================================================================

set -euo pipefail

echo "Running on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Array task: ${SLURM_ARRAY_TASK_ID:-local}"
date

# --- The array index selects exactly one decade -----------------------------
DECADES=(
    "1800_1809"
    "1810_1819"
    "1820_1829"
    "1830_1839"
    "1840_1849"
    "1850_1859"
    "1860_1869"
    "1870_1879"
    "1880_1889"
    "1890_1899"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

if (( TASK_ID < 0 || TASK_ID >= ${#DECADES[@]} )); then
    echo "ERROR: invalid SLURM_ARRAY_TASK_ID=${TASK_ID}"
    exit 1
fi

DECADE="${DECADES[$TASK_ID]}"
echo "Processing decade: ${DECADE}"

# --- Point at the pre-downloaded, offline HF cache --------------------------
export HF_HOME="/gpfs/scratch/bsc100/paolo/.cache/huggingface"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# --- Environment ------------------------------------------------------------
ENV_PATH="/gpfs/scratch/bsc100/paolo/.conda/envs/textmachine-llm"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PATH"

cd "$SLURM_SUBMIT_DIR"

# --- Sanity check: cache is present before we try ---------------------------
if [ ! -d "$HF_HOME" ]; then
    echo "ERROR: HF_HOME ($HF_HOME) not found."
    echo "Run download_assets.sh from alogin4/glogin4 first."
    exit 1
fi

python3 build_prior_v4.py --decade "$DECADE"

echo "Done: ${DECADE}"
date
