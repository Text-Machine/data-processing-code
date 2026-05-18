#!/bin/bash
#SBATCH --job-name=filter_morning_night
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=4
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
# ---------------------------------------------------------------------------
# Thread controls — prevent NumPy/OpenBLAS/MKL oversubscription
# ---------------------------------------------------------------------------
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
OUTPUT_DIR=/gpfs/projects/bsc100/textmachine-data/filtered_data/lwm_hmd
mkdir -p logs_morning_night_filtered "$OUTPUT_DIR"
# ---------------------------------------------------------------------------
# Workload definition
# ---------------------------------------------------------------------------
TASKS=(
    "hmd morning"
    "hmd night"
    "lwm morning"
    "lwm night"
)

# ---------------------------------------------------------------------------
# Launch tasks
# ---------------------------------------------------------------------------
running=0 
MAX_CONCURRENT=4

for task in "${TASKS[@]}"; do
    dataset="${task%% *}"
    word="${task##* }"

    srun --exclusive -N1 -n1 -c "$SLURM_CPUS_PER_TASK" \
            python filter_hmd_lwm_simplelogic.py \
            #python filter_hmd_lwm_nlpsentencizer.py \
            --dataset "$dataset" \
            --query-word "$word" \
            --output-dir "$OUTPUT_DIR" \
            --log-level INFO \
        > "logs_morning_night_filtered/filter_${dataset}_${word}.out" 2>&1 &

    (( running++ ))

    if (( running >= MAX_CONCURRENT )); then
        wait -n 2>/dev/null || wait                            
        (( running-- ))
    fi
done

wait
echo "All filtering tasks completed."
