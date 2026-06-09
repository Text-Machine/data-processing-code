#!/bin/bash
#SBATCH --job-name=filter_blmicrosoft
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
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
SCRIPT="$SLURM_SUBMIT_DIR/filter_data_blmicrosoft.py"
METADATA=/gpfs/projects/bsc100/textmachine-data/preprocessed_data/consolidated_metadata/metadata_blmicrosoft_step_3.csv
DATA_ROOT=/gpfs/projects/bsc100/textmachine-data/downloaded_data_vm
mkdir -p "$LOG_DIR"
# ---------------------------------------------------------------------------
# Workload
# ---------------------------------------------------------------------------
WORDS=(slave slaves machine machines mornings nights morning night)
# ---------------------------------------------------------------------------
# Launch tasks
# ---------------------------------------------------------------------------
for word in "${WORDS[@]}"; do
    srun --exclusive -n1 -c "$SLURM_CPUS_PER_TASK" \
        python "$SCRIPT" \
            --query-word  "$word" \
            --output-file "$OUTPUT_DIR/bl_microsoft_step_4_${word}.jsonl" \
            --metadata    "$METADATA" \
            --data-root   "$DATA_ROOT" \
            --log-level   INFO \
        > "$LOG_DIR/filter_${word}.out" 2>&1 &
done
wait
echo "All filtering tasks completed."