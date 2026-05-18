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

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate textmachine_py310

cd "$SLURM_SUBMIT_DIR"

SCRIPT_DIR="$SLURM_SUBMIT_DIR"
LOG_DIR="$SCRIPT_DIR/logs_remaining_words_filtered"

OUTPUT_DIR=/gpfs/projects/bsc100/textmachine-data/filtered_data/lwm_hmd

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

SCRIPT=filter_hmd_lwm_simplelogic.py
# SCRIPT=filter_hmd_lwm_nlpsentencizer.py

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

DATASETS=(lwm hmd)
WORDS=(slave slaves machine machines mornings nights)

MAX_CONCURRENT=12

running=0

for dataset in "${DATASETS[@]}"; do
    for word in "${WORDS[@]}"; do

        srun --exclusive -n1 -c "$SLURM_CPUS_PER_TASK" \
            python "$SCRIPT" \
                --dataset "$dataset" \
                --query-word "$word" \
                --output-dir "$OUTPUT_DIR" \
                --log-level INFO \
            > "$LOG_DIR/filter_${dataset}_${word}.out" 2>&1 &

        ((running++))

        if (( running >= MAX_CONCURRENT )); then
            wait -n
            ((running--))
        fi
    done
done

wait

echo "All filtering tasks completed."