#!/bin/bash
#SBATCH --job-name=unroll_masks_bl_microsoft
#SBATCH --output=logs/unroll_masks_%j.log
#SBATCH --error=logs/unroll_masks_%j.err
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --qos=gp_debug
#SBATCH --account=bsc100

# -------------------------------------------------------------------
# Paths — edit here if the data ever moves
# -------------------------------------------------------------------

INPUT_DIR="/gpfs/projects/bsc100/textmachine-data/preprocessed_data/consolidated_metadata/"
OUTPUT_DIR="/gpfs/projects/bsc100/textmachine-data/preprocessed_data/consolidated_metadata/"

# -------------------------------------------------------------------

cd $SLURM_SUBMIT_DIR

mkdir -p logs


python3 unroll_masks.py \
    --input-dir  "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --suffix     "_step_5"
