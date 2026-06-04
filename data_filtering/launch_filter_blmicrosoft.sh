#!/bin/bash
#SBATCH --job-name=filter_blmicrosoft
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
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
#OUTPUT_DIR=/gpfs/projects/bsc100/textmachine-data/filtered_data/bl_microsoft
OUTPUT_DIR=/gpfs/scratch/bsc100/paolo/bl_microsoft_repeat_25_05_2026
LOG_DIR="$SLURM_SUBMIT_DIR/logs_filter_blmicrosoft"
#SCRIPT="$SLURM_SUBMIT_DIR/filter_blmicrosoft.py"
SCRIPT="$SLURM_SUBMIT_DIR/filter_blmicrosoft_nlpsentencizer.py"


mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

# ---------------------------------------------------------------------------
# Workload
# ---------------------------------------------------------------------------
WORDS=(slave slaves machine machines mornings nights morning night)

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
            --output-dir "$OUTPUT_DIR" \
            --log-level INFO \
        > "$LOG_DIR/filter_${word}.out" 2>&1 &

    ((running++))

    if (( running >= MAX_CONCURRENT )); then
        wait -n 2>/dev/null || wait
        ((running--))
    fi
done

wait
echo "All filtering tasks completed."
