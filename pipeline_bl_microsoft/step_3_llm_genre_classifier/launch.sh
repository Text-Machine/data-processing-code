#!/bin/bash
#SBATCH --job-name=genre-classifier
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=00:14:00
#SBATCH --account=bsc100
##SBATCH --qos=acc_bsccssh
#SBATCH --qos=acc_debug
#SBATCH --exclusive
# ---------------------------------------------------------------------------
# Configuration — edit these before submitting
# ---------------------------------------------------------------------------
LOG_DIR="$SLURM_SUBMIT_DIR/logs"
mkdir -p "$LOG_DIR"

#Update the HF_HOME and ENV_PATH accordingly
export HF_HOME=/gpfs/scratch/bsc100/paolo/.cache/huggingface
ENV_PATH="/gpfs/scratch/bsc100/paolo/.conda/envs/llm-genre-classification-env"
# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT="$SLURM_SUBMIT_DIR/genre_classifier.py"
INPUT_DIR="/gpfs/projects/bsc100/textmachine-data/preprocessed_data/consolidated_metadata_blmicrosoft"
OUTPUT_DIR="/gpfs/projects/bsc100/textmachine-data/preprocessed_data/consolidated_metadata_blmicrosoft"
mkdir -p "$OUTPUT_DIR"
MODEL="gemma-4-31b"   # shorthand: llama-8b | gemma-4-31b
# or set a full path: /gpfs/projects/bsc100/models/gemma4/gemma-4-31B-it
# Set to your CSV filename within INPUT_DIR, or leave empty to run on built-in test data
INPUT_CSV_FILE="metadata_blmicrosoft_deduplicated.csv"
INPUT_CSV="$INPUT_DIR/$INPUT_CSV_FILE"
OUTPUT_CSV="$OUTPUT_DIR/blmicrosoft_final_metadata.csv"
MODE="zeroshot"        # zeroshot | fewshot
BATCH_SIZE=20
SLICE_START=0
SLICE_END=""           # leave empty to process all rows
# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
echo "=============================="
echo "Job:        $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Start:      $(date)"
echo "Script:     $SCRIPT"
echo "Model:      $MODEL"
echo "Input dir:  $INPUT_DIR"
echo "Input CSV:  ${INPUT_CSV:-<test data>}"
echo "Output dir: $OUTPUT_DIR"
echo "Output CSV: $OUTPUT_CSV"
echo "Mode:       $MODE"
echo "=============================="
# Activate conda env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PATH"
echo "Python: $(which python)"
echo "Env:    $CONDA_PREFIX"
# ---------------------------------------------------------------------------
# Build optional args
# ---------------------------------------------------------------------------
EXTRA_ARGS=""
if [ -n "$INPUT_CSV" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --input_csv $INPUT_CSV"
fi
if [ -n "$SLICE_END" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --slice_end $SLICE_END"
fi
# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
python "$SCRIPT" \
    --model "$MODEL" \
    --output_csv "$OUTPUT_CSV" \
    --mode "$MODE" \
    --batch_size "$BATCH_SIZE" \
    --slice_start "$SLICE_START" \
    $EXTRA_ARGS
EXIT_CODE=$?
echo "=============================="
echo "End: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=============================="
exit $EXIT_CODE
