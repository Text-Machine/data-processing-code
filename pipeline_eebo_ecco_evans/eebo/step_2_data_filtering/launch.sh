#!/bin/bash

#SBATCH --job-name=filter_tcp
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
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
# Dataset selection
# ---------------------------------------------------------------------------
# Set to "evans", "eebo", or "ecco". Everything else about the run is
# identical regardless of dataset -- filter_tcp.py handles EEBO's two-CSV
# case and the date/edition_date column difference on its own.

DATASET="eebo"

declare -A DATA_ROOTS=(
  [evans]="/gpfs/projects/bsc100/textmachine-data/preprocessed_data/output_evans"
  [eebo]="/gpfs/projects/bsc100/textmachine-data/preprocessed_data/output_eebo"
  [ecco]="/gpfs/projects/bsc100/textmachine-data/preprocessed_data/output_ecco"
)

DATA_ROOT="${DATA_ROOTS[$DATASET]}"
if [[ -z "$DATA_ROOT" ]]; then
    echo "ERROR: unknown DATASET='$DATASET' -- must be one of: ${!DATA_ROOTS[*]}"
    exit 1
fi


# ---------------------------------------------------------------------------
# Query words
# ---------------------------------------------------------------------------

QUERY_CONFIG="${SLURM_SUBMIT_DIR}/query_word_config.sh"

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

OUTPUT_DIR="/gpfs/projects/bsc100/textmachine-data/preprocessed_data/consolidated_metadata_${DATASET}"
LOG_DIR="$SLURM_SUBMIT_DIR/logs_filter_${DATASET}"
SCRIPT="$SLURM_SUBMIT_DIR/query_word_filter_v3.py"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_CONCURRENT=8
running=0


echo "============================================================"
echo "TCP Step 2 -- $DATASET"
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

    # Make the word safe for filenames -- matches filter_tcp.py's own
    # slugify() closely enough for a plain word like "machine"/"machines".
    safe_word=$(echo "$word" | sed 's/[^[:alnum:]_.-]/_/g')

    output_file="$OUTPUT_DIR/${DATASET}_${safe_word}_step2.jsonl"
    timestamp=$(date '+%Y%m%d_%H%M%S')
    log_file="$LOG_DIR/filter_${safe_word}_${timestamp}.out"
    #log_file="$LOG_DIR/filter_${safe_word}.out"


    # ---------------------------------------------------------------
    # Skip if output already exists
    # ---------------------------------------------------------------

    if [[ -s "$output_file" ]]; then

        echo "Skipping '$word' — output already exists:"
        echo "  $output_file"

        continue
    fi


    echo "Starting '$word'"
    echo "  Output: $output_file"
    echo "  Log   : $log_file"


    # ---------------------------------------------------------------
    # Launch one independent task
    # ---------------------------------------------------------------
    # filter_tcp.py already accepts a single --query-word directly, so
    # unlike the LWM launcher this doesn't need a per-word temp config
    # file -- the word is passed straight through.

    srun --exclusive \
        -n1 \
        -c "$SLURM_CPUS_PER_TASK" \
        python "$SCRIPT" \
            --dataset-name "$DATASET" \
            --data-root "$DATA_ROOT" \
            --query-word "$word" \
            --output-file "$output_file" \
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


echo "============================================================"
echo "All $DATASET filtering tasks completed."
echo "============================================================"