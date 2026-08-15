#!/bin/bash
#SBATCH --job-name=unroll_masks_bl_microsoft
#SBATCH --output=logs/unroll_masks_%j.log
#SBATCH --error=logs/unroll_masks_%j.err
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --qos=gp_debug
#SBATCH --account=bsc100

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source "${SLURM_SUBMIT_DIR}/../query_word_config.sh"

INPUT_DIR="/gpfs/projects/bsc100/textmachine-data/preprocessed_data/consolidated_metadata_lwm"
OUTPUT_DIR="/gpfs/projects/bsc100/textmachine-data/preprocessed_data/consolidated_metadata_lwm"
SUFFIX="spacy"   # must match the suffix used in step 4 (spacy or regex)

echo "Words this run: ${QUERY_WORDS[*]}"

python3 unroll_masks.py \
  --input-dir   "$INPUT_DIR" \
  --output-dir  "$OUTPUT_DIR" \
  --suffix      "$SUFFIX" \
  --query-words "${QUERY_WORDS[@]}"