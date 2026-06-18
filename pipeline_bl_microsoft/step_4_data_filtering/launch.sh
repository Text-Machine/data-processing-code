#!/bin/bash
#SBATCH --job-name=filter_blmicrosoft
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=2
##for regex version
##SBATCH --time=00:20:00
##for spacy version
#SBATCH --time=00:55:00
#SBATCH --account=bsc100
#SBATCH --qos=gp_debug
#SBATCH --exclusive
# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate textmachine_py310
# ALWAYS use submission directory (critical for SLURM safety)
cd "$SLURM_SUBMIT_DIR"
# ---------------------------------------------------------------------------
# Thread controls
# ---------------------------------------------------------------------------
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUTPUT_DIR=/gpfs/projects/bsc100/textmachine-data/preprocessed_data/consolidated_metadata/
LOG_DIR="$SLURM_SUBMIT_DIR/logs_filter_blmicrosoft"
#SCRIPT="$SLURM_SUBMIT_DIR/filter_data_blmicrosoft_regex.py"
SCRIPT="$SLURM_SUBMIT_DIR/filter_data_blmicrosoft_spacy.py"
METADATA=/gpfs/projects/bsc100/textmachine-data/preprocessed_data/consolidated_metadata/metadata_blmicrosoft_step_3.csv
DATA_ROOT=/gpfs/projects/bsc100/textmachine-data/downloaded_data_vm
mkdir -p "$LOG_DIR" 
# ---------------------------------------------------------------------------
# Suffix derived from which script variant is being run
# ---------------------------------------------------------------------------
SCRIPT_BASENAME="$(basename "$SCRIPT")"
case "$SCRIPT_BASENAME" in
    filter_data_blmicrosoft_spacy.py)
        SUFFIX="spacy"
        ;;
    filter_data_blmicrosoft_regex.py)
        SUFFIX="regex"
        ;;
    *)
        echo "ERROR: unrecognized SCRIPT '$SCRIPT_BASENAME' — no suffix mapping defined." >&2
        exit 1
        ;;
esac
echo "Using script: $SCRIPT_BASENAME  ->  suffix: step_4_${SUFFIX}"
# ---------------------------------------------------------------------------
# Workload — word and its output file (suffix-aware)
# ---------------------------------------------------------------------------
WORDS=(slave slaves machine machines mornings nights morning night)
declare -A OUTPUT_FILES
for word in "${WORDS[@]}"; do
    OUTPUT_FILES[$word]="$OUTPUT_DIR/bl_microsoft_step_4_${word}_${SUFFIX}.jsonl"
done
# ---------------------------------------------------------------------------
# Concurrency control
# ---------------------------------------------------------------------------
MAX_CONCURRENT=8
running=0
# ---------------------------------------------------------------------------
# Launch tasks
# ---------------------------------------------------------------------------
for word in "${WORDS[@]}"; do
    srun --exclusive -n1 -c "$SLURM_CPUS_PER_TASK" \
        python "$SCRIPT" \
            --query-word "$word" \
            --output-file "${OUTPUT_FILES[$word]}" \
            --metadata   "$METADATA" \
            --data-root  "$DATA_ROOT" \
            --log-level INFO \
        > "$LOG_DIR/filter_${word}_${SUFFIX}.out" 2>&1 &
    ((running++))
    if (( running >= MAX_CONCURRENT )); then
        wait -n 2>/dev/null || wait
        ((running--))
    fi
done
wait
echo "All filtering tasks completed."
