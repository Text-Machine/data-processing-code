#!/bin/bash

#SBATCH --job-name=filter_lwm
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --account=bsc100
#SBATCH --qos=gp_debug
#SBATCH --exclusive

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate textmachine_py310

cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"


# ---------------------------------------------------------------------------
# Query word
# ---------------------------------------------------------------------------

QUERY_CONFIG="${SLURM_SUBMIT_DIR}/../query_word_config.sh"

if [[ ! -f "$QUERY_CONFIG" ]]; then
    echo "ERROR: query-word configuration not found:"
    echo "       $QUERY_CONFIG"
    exit 1
fi

source "$QUERY_CONFIG"

if [[ -z "${QUERY_WORD:-}" ]]; then
    echo "ERROR: QUERY_WORD is not defined in:"
    echo "       $QUERY_CONFIG"
    exit 1
fi

echo "Query word: $QUERY_WORD"


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

OUTPUT_DIR=/gpfs/projects/bsc100/textmachine-data/preprocessed_data/consolidated_metadata_lwm

LOG_DIR="$SLURM_SUBMIT_DIR/logs_filter_lwm"

SCRIPT="$SLURM_SUBMIT_DIR/data_filtering_lwm.py"

DATA_ROOT=/gpfs/projects/bsc100/textmachine-data/preprocessed_data/output_lwm


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

OUTPUT_FILE="$OUTPUT_DIR/lwm_step_4_${QUERY_WORD}.jsonl"

LOG_FILE="$LOG_DIR/filter_${QUERY_WORD}.out"


# ---------------------------------------------------------------------------
# Check whether output already exists
# ---------------------------------------------------------------------------

if [[ -s "$OUTPUT_FILE" ]]; then
    echo "Skipping '$QUERY_WORD' — output already exists:"
    echo "$OUTPUT_FILE"
    exit 0
fi


# ---------------------------------------------------------------------------
# Run filtering
# ---------------------------------------------------------------------------

echo "============================================================"
echo "LWM Step 4"
echo "============================================================"
echo "Script      : $SCRIPT"
echo "Query word  : $QUERY_WORD"
echo "Data root   : $DATA_ROOT"
echo "Output      : $OUTPUT_FILE"
echo "Log         : $LOG_FILE"
echo "============================================================"


srun --exclusive \
    -n1 \
    -c "$SLURM_CPUS_PER_TASK" \
    python "$SCRIPT" \
        --query-config "$QUERY_CONFIG" \
        --output-file "$OUTPUT_FILE" \
        --data-root "$DATA_ROOT" \
        --log-level INFO \
        > "$LOG_FILE" 2>&1

STATUS=$?


# ---------------------------------------------------------------------------
# Final status
# ---------------------------------------------------------------------------

if [[ $STATUS -eq 0 ]]; then

    echo "============================================================"
    echo "Filtering completed successfully."
    echo "Output: $OUTPUT_FILE"
    echo "============================================================"

else

    echo "============================================================"
    echo "ERROR: filtering failed with exit code $STATUS"
    echo "See log: $LOG_FILE"
    echo "============================================================"

    exit "$STATUS"

fi