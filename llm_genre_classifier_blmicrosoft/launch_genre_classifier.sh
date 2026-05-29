#!/bin/bash
#SBATCH --job-name=genre-classifier
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:2
##SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --account=bsc100
#SBATCH --qos=acc_bsccssh
##SBATCH --qos=acc_debug
##SBATCH --exclusive

# ---------------------------------------------------------------------------
# Configuration — edit these before submitting
# ---------------------------------------------------------------------------

LOG_DIR="$SLURM_SUBMIT_DIR/logs"
mkdir -p "$LOG_DIR"

#export HF_HOME=/gpfs/scratch/$USER/.cache/huggingface
export HF_HOME=/gpfs/scratch/bsc100/paolo/.cache/huggingface
ENV_PATH="/gpfs/scratch/bsc100/paolo/.conda/envs/textmachine-llm"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT="$SLURM_SUBMIT_DIR/genre_classifier.py"
OUTPUT_DIR="/gpfs/scratch/bsc100/paolo/llm_genre_classifier_results"
mkdir -p "$OUTPUT_DIR"

MODEL="llama-8b"   # shorthand: llama-8b | mistral-small | deepseek-32b | qwen-32b
                   # or set a full path: /gpfs/projects/bsc100/models/meta-llama/Llama-3.1-8B-Instruct

# Set to your CSV path, or leave empty to run on built-in test data
#INPUT_CSV=""
INPUT_CSV="/gpfs/projects/bsc100/textmachine-data/preprocessed_data/output_blmicrosoft/metadata.csv"
OUTPUT_CSV="$OUTPUT_DIR/genre_classified_$(date +%Y%m%d_%H%M).csv"

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
echo "Input CSV:  ${INPUT_CSV:-<test data>}"
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
