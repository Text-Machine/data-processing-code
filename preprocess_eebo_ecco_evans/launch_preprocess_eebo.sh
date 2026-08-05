#!/bin/bash
#SBATCH --job-name=eebo_preprocess
#SBATCH --account=bsc100
##SBATCH --qos=gp_debug
#SBATCH --qos=gp_bsccssh
#SBATCH --time=00:25:00
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

INPUT_ZIP="/gpfs/projects/bsc100/textmachine-data/eebo_all.zip"
OUTPUT_DIR="/gpfs/projects/bsc100/textmachine-data/preprocessed_data/output_eebo"
#OUTPUT_DIR="/gpfs/scratch/bsc100/paolo/eebo/output_eebo"

cd "$SLURM_SUBMIT_DIR"

SCRIPT_PATH="${SLURM_SUBMIT_DIR}/preprocess_eebo.py"

mkdir -p "${OUTPUT_DIR}"

echo "Job: ${SLURM_JOB_NAME}  ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"

echo "Start: $(date)"

# --- environment setup ---
source venv/bin/activate

python3 "${SCRIPT_PATH}" \
    --input_zip "${INPUT_ZIP}" \
    --output_dir "${OUTPUT_DIR}"

echo "End: $(date)"
