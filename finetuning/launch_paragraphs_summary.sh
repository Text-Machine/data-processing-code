#!/bin/bash
#SBATCH --job-name=summarise-paragraphs
#SBATCH --account=bsc100
#SBATCH --qos=acc_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Edit the settings below, then submit with:  sbatch launch_summarise.sh

set -euo pipefail

# --- Settings --------------------------------------------------------------
BOOK_NAME="oliver_twist"
PARAGRAPHS_FILE="$SLURM_SUBMIT_DIR/data/paragraphs/oliver_twist_paragraphs_post_corrected_no_vllm.json"
OUTPUT_CSV="$SLURM_SUBMIT_DIR/data/paragraphs/${BOOK_NAME}_paragraphs_summaries.csv"
SCRIPT="$SLURM_SUBMIT_DIR/summary_paragraphs.py"
MODEL_ID="/gpfs/projects/bsc100/models/Qwen/Qwen3.5-9B/"
MAX_NEW_TOKENS=128
SEED=42

mkdir -p "$SLURM_SUBMIT_DIR/logs" "$(dirname "$OUTPUT_CSV")"

# --- Environment -----------------------------------------------------------
# Hugging Face cache
export HF_HOME=/gpfs/scratch/bsc100/paolo/.cache/huggingface

module purge
module load bsc/1.0
module load mkl
module load intel
module load impi
module load hdf5
module load python/3.12
module load sqlite3/3.45.2
module load cuda/12.8
unset PYTHONPATH
export PYTHONNOUSERSITE=1
source /gpfs/scratch/bsc100/paolo/venv_mn5_qwen35_9b_v3/bin/activate

# Offline mode: model is local, compute nodes have no internet
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "Job:        $SLURM_JOB_ID on $(hostname)"
echo "Book:       $BOOK_NAME"
echo "Input:      $PARAGRAPHS_FILE"
echo "Output:     $OUTPUT_CSV"
echo "Model:      $MODEL_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv

# --- Run -------------------------------------------------------------------
srun python "$SCRIPT" \
    --paragraphs-file "$PARAGRAPHS_FILE" \
    --output "$OUTPUT_CSV" \
    --book-name "$BOOK_NAME" \
    --model-id "$MODEL_ID" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --seed "$SEED"