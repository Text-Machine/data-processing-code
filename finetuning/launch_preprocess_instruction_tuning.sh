#!/bin/bash
#SBATCH --job-name=gemma-summary
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --account=bsc100
##SBATCH --qos=acc_bsccssh
#SBATCH --qos=acc_debug

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOG_DIR="$SLURM_SUBMIT_DIR/logs"
mkdir -p "$LOG_DIR"

# Hugging Face cache
export HF_HOME=/gpfs/scratch/bsc100/paolo/.cache/huggingface

# Conda environment
#ENV_PATH="/gpfs/scratch/bsc100/paolo/.conda/envs/textmachine-llm-v2"
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


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT="$SLURM_SUBMIT_DIR/preprocess_instruction_tuning_v2.py"

# Run directory
RUN_ROOT="$SLURM_SUBMIT_DIR"

# Input paragraphs
PARAGRAPHS_FILE="/gpfs/scratch/bsc100/paolo/tm96_finetuning/data-processing-code/finetuning/data/paragraphs/oliver_twist_paragraphs.json"

# Book name
BOOK_NAME="oliver_twist"



# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

echo "=============================="
echo "Job:          $SLURM_JOB_ID"
echo "Node:         $SLURMD_NODENAME"
echo "Start:        $(date)"
echo "Script:       $SCRIPT"
echo "Run root:     $RUN_ROOT"
echo "Paragraphs:   $PARAGRAPHS_FILE"
echo "Book:         $BOOK_NAME"
echo "GPUs:         $CUDA_VISIBLE_DEVICES"
echo "=============================="

# ---------------------------------------------------------------------------
# Activate conda environment
# ---------------------------------------------------------------------------


echo "Python:       $(which python)"
echo "PyTorch:      $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA:         $(python -c 'import torch; print(torch.version.cuda)')"
echo "GPU count:    $(python -c 'import torch; print(torch.cuda.device_count())')"

# ---------------------------------------------------------------------------
# Export variables consumed by preprocess_instruction_tuning.py
# ---------------------------------------------------------------------------

export RUN_ROOT="$RUN_ROOT"
export PARAGRAPHS_FILE="$PARAGRAPHS_FILE"
export BOOK_NAME="$BOOK_NAME"


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

python "$SCRIPT"

EXIT_CODE=$?

echo "=============================="
echo "End:          $(date)"
echo "Exit code:    $EXIT_CODE"
echo "=============================="

exit $EXIT_CODE