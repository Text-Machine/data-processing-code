#!/bin/bash
#SBATCH --job-name=merge_prior
#SBATCH --account=bsc100
#SBATCH --qos=gp_debug
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
##SBATCH --constraint=highmem

# ============================================================================
# launch_prior_merge.sh
#
# Merges the decade-level partial pallets produced by:
#
#   sbatch launch_prior_filter.sh
#
# The merge:
#   - sums TF/DF pallets across all decades
#   - applies smoothing
#   - merges the QA reservoirs
#   - computes corpus-level mu/sigma statistics
#   - writes the final prior
#
# Usage:
#   mkdir -p logs
#   sbatch launch_prior_merge.sh
# ============================================================================

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

# --- Check that all decade partials exist -----------------------------------
PARTIAL_DIR="prior_filter/blmicrosoft/partials"

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

echo "Checking decade partials..."

for decade in "${DECADES[@]}"; do
    partial="${PARTIAL_DIR}/pallet_${decade}.pt"

    if [ ! -f "$partial" ]; then
        echo "ERROR: Missing partial:"
        echo "  $partial"
        echo
        echo "Make sure all tasks from launch_prior_filter.sh have completed."
        exit 1
    fi

    echo "  found: $partial"
done

echo
echo "All decade partials found."
echo "Starting merge..."

python3 merge_prior.py

echo
echo "Merge completed successfully."
date