#!/bin/bash
#SBATCH --job-name=apply_prior_bl
#SBATCH --account=bsc100
#SBATCH --qos=gp_debug
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# ============================================================
# File Paths Configuration
# ============================================================

PRIOR_PATH="/gpfs/projects/bsc100/textmachine-data/preprocessed_data/prior_filter_blmicrosoft/prior_blmicrosoft.pt"
INPUT_JSONL_PATH="/gpfs/projects/bsc100/textmachine-data/preprocessed_data/consolidated_metadata_blmicrosoft/bl_microsoft_slaves_spacy_step_6.jsonl"
OUTPUT_JSONL_PATH="/gpfs/projects/bsc100/textmachine-data/preprocessed_data/consolidated_metadata_blmicrosoft/bl_microsoft_slaves_spacy_step_7.jsonl"

# ============================================================
# Environment
# ============================================================

export HF_HOME="/gpfs/scratch/bsc100/paolo/.cache/huggingface"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

ENV_PATH="/gpfs/scratch/bsc100/paolo/.conda/envs/textmachine-llm"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PATH"

# ============================================================
# Working directory
# ============================================================

cd "$SLURM_SUBMIT_DIR"

mkdir -p logs

# ============================================================
# Job information
# ============================================================

echo "========================================"
echo "Job:        $SLURM_JOB_NAME"
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $(hostname)"
echo "Start:      $(date)"
echo "CPUs:       $SLURM_CPUS_PER_TASK"
echo "Working dir:"
pwd
echo "========================================"

echo "Prior File:"
echo "  $PRIOR_PATH"

echo "Input JSONL:"
echo "  $INPUT_JSONL_PATH"

echo "Output JSONL:"
echo "  $OUTPUT_JSONL_PATH"

echo "========================================"

# ============================================================
# Run
# ============================================================

python apply_prior.py \
    --prior "$PRIOR_PATH" \
    --input "$INPUT_JSONL_PATH" \
    --output "$OUTPUT_JSONL_PATH"

# ============================================================
# Finished
# ============================================================

echo "========================================"
echo "Finished: $(date)"
echo "========================================"