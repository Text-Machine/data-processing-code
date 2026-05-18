#!/bin/bash
#SBATCH --job-name=filter_remaining_words
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=12
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
cd /home/bsc/bsc204326/text_machine_processing

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
mkdir -p logs_remaining_words_filtered "$OUTPUT_DIR"

# ---------------------------------------------------------------------------
# Workload definition
# ---------------------------------------------------------------------------
DATASETS=(lwm hmd)
WORDS=(slave slaves machine machines mornings nights)

# ---------------------------------------------------------------------------
# Concurrency cap
# ---------------------------------------------------------------------------
MAX_CONCURRENT=12

# ---------------------------------------------------------------------------
# Launch tasks
# ---------------------------------------------------------------------------
running=0
for dataset in "${DATASETS[@]}"; do
    for word in "${WORDS[@]}"; do
        srun --exclusive -N1 -n1 -c "$SLURM_CPUS_PER_TASK" \
            python filter_hmd_lwm_simplelogic.py \
            #python filter_hmd_lwm_nlpsentencizer.py \
                --dataset "$dataset" \
                --query-word "$word" \
                --output-dir "$OUTPUT_DIR" \
                --log-level INFO \
            > "logs_remaining_words_filtered/filter_${dataset}_${word}.out" 2>&1 &
        running=$(( running + 1 ))
        if (( running >= MAX_CONCURRENT )); then
            wait -n
            running=$(( running - 1 ))
        fi
    done
done

wait
echo "All filtering tasks completed."
