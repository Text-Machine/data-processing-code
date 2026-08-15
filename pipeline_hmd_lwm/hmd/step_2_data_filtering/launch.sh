#!/bin/bash

#SBATCH --job-name=filter_lwm
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=2
#SBATCH --time=00:55:00
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
# Query words
# ---------------------------------------------------------------------------

QUERY_CONFIG="${SLURM_SUBMIT_DIR}/../query_word_config.sh"

if [[ ! -f "$QUERY_CONFIG" ]]; then
    echo "ERROR: query-word configuration not found:"
    echo "       $QUERY_CONFIG"
    exit 1
fi

source "$QUERY_CONFIG"

if [[ ${#QUERY_WORDS[@]} -eq 0 ]]; then
    echo "ERROR: QUERY_WORDS is empty."
    exit 1
fi

echo "Words this run: ${QUERY_WORDS[*]}"


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TMP_CONFIG_DIR="$SLURM_SUBMIT_DIR/.query_configs_hmd"

OUTPUT_DIR=/gpfs/projects/bsc100/textmachine-data/preprocessed_data/consolidated_metadata_hmd

LOG_DIR="$SLURM_SUBMIT_DIR/logs_filter_hmd"

SCRIPT="$SLURM_SUBMIT_DIR/data_filtering_hmd.py"

DATA_ROOT=/gpfs/projects/bsc100/textmachine-data/preprocessed_data/output_hmd



mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$TMP_CONFIG_DIR"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_CONCURRENT=8
running=0


echo "============================================================"
echo "LWM Step 4"
echo "============================================================"
echo "Script       : $SCRIPT"
echo "Data root    : $DATA_ROOT"
echo "Output dir   : $OUTPUT_DIR"
echo "Log dir      : $LOG_DIR"
echo "Concurrency  : $MAX_CONCURRENT"
echo "Words        : ${QUERY_WORDS[*]}"
echo "============================================================"


# ---------------------------------------------------------------------------
# Run one filtering process per query word
# ---------------------------------------------------------------------------

for word in "${QUERY_WORDS[@]}"; do

    # Make the word safe for filenames.
    safe_word=$(echo "$word" | sed 's/[^[:alnum:]_.-]/_/g')

    output_file="$OUTPUT_DIR/hmd_step_4_${safe_word}.jsonl"
    log_file="$LOG_DIR/filter_${safe_word}.out"
    word_config="$TMP_CONFIG_DIR/query_${safe_word}.sh"


    # ---------------------------------------------------------------
    # Skip if output already exists
    # ---------------------------------------------------------------

    if [[ -s "$output_file" ]]; then

        echo "Skipping '$word' — output already exists:"
        echo "  $output_file"

        continue
    fi


    # ---------------------------------------------------------------
    # Create a temporary config containing this query word
    # ---------------------------------------------------------------

    cat > "$word_config" <<EOF
QUERY_WORD="$word"
EOF


    echo "Starting '$word'"
    echo "  Output: $output_file"
    echo "  Log   : $log_file"


    # ---------------------------------------------------------------
    # Launch one independent task
    # ---------------------------------------------------------------

    srun --exclusive \
        -n1 \
        -c "$SLURM_CPUS_PER_TASK" \
        python "$SCRIPT" \
            --query-config "$word_config" \
            --output-file "$output_file" \
            --data-root "$DATA_ROOT" \
            --log-level INFO \
        > "$log_file" 2>&1 &


    ((running++))


    # ---------------------------------------------------------------
    # Limit number of concurrent processes
    # ---------------------------------------------------------------

    if (( running >= MAX_CONCURRENT )); then

        wait -n 2>/dev/null || wait

        ((running--))

    fi

done


# ---------------------------------------------------------------------------
# Wait for remaining tasks
# ---------------------------------------------------------------------------

wait


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

rm -rf "$TMP_CONFIG_DIR"


echo "============================================================"
echo "All LWM filtering tasks completed."
echo "============================================================"

