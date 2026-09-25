#!/bin/bash

#SBATCH --job-name=pride-prejudice-lora
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1

#SBATCH --time=00:35:00

#SBATCH --account=bsc100
#SBATCH --qos=acc_debug


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

set -e

LOG_DIR="$SLURM_SUBMIT_DIR/logs"
mkdir -p "$LOG_DIR"


# ---------------------------------------------------------------------------
# Hugging Face cache
# ---------------------------------------------------------------------------

export HF_HOME=/gpfs/scratch/bsc100/paolo/.cache/huggingface


# ---------------------------------------------------------------------------
# MN5 modules
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Python environment
# ---------------------------------------------------------------------------

source /gpfs/scratch/bsc100/paolo/venv_mn5_qwen35_9b_v3/bin/activate


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT="$SLURM_SUBMIT_DIR/inference_lora_pride_and_prejudice.py"

INPUT_DIR="/gpfs/projects/bsc100/textmachine-data/downloaded_data_vm"

BOOK_PATH="1890_1899/1895_000144653.jsonl.gz"

ADAPTER_DIR="/gpfs/scratch/bsc100/paolo/tm96_finetuning/data-processing-code/finetuning/outputs/"

OUTPUT_CSV="/gpfs/scratch/bsc100/paolo/tm96_finetuning/data-processing-code/finetuning/outputs/pride_prejudice_lora_inference.csv"


# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------

export INPUT_DIR
export BOOK_PATH
export ADAPTER_DIR
export OUTPUT_CSV


# ---------------------------------------------------------------------------
# PyTorch settings
# ---------------------------------------------------------------------------

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


# ---------------------------------------------------------------------------
# Information
# ---------------------------------------------------------------------------

echo "=============================="
echo "Job:          $SLURM_JOB_ID"
echo "Node:         $SLURMD_NODENAME"
echo "Start:        $(date)"
echo "Script:       $SCRIPT"
echo "Input dir:    $INPUT_DIR"
echo "Book path:    $BOOK_PATH"
echo "Adapter:      $ADAPTER_DIR"
echo "Output:       $OUTPUT_CSV"
echo "GPU:          $CUDA_VISIBLE_DEVICES"
echo "=============================="


# ---------------------------------------------------------------------------
# Python / CUDA diagnostics
# ---------------------------------------------------------------------------

echo
echo "Python:"
which python
python --version

echo
echo "PyTorch:"
python -c "import torch; print(torch.__version__)"

echo
echo "CUDA:"
python -c "import torch; print(torch.version.cuda)"

echo
echo "GPU count:"
python -c "import torch; print(torch.cuda.device_count())"

echo
echo "GPU:"
python -c "import torch; print(torch.cuda.get_device_name(0))"


# ---------------------------------------------------------------------------
# Check script
# ---------------------------------------------------------------------------

if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: inference script does not exist:"
    echo "$SCRIPT"
    exit 1
fi


# ---------------------------------------------------------------------------
# Check adapter
# ---------------------------------------------------------------------------

if [ ! -f "$ADAPTER_DIR/adapter_config.json" ]; then
    echo "ERROR: adapter_config.json not found:"
    echo "$ADAPTER_DIR"
    exit 1
fi


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

echo
echo "=============================="
echo "Starting Pride and Prejudice inference"
echo "=============================="

python "$SCRIPT"

EXIT_CODE=$?


# ---------------------------------------------------------------------------
# Finish
# ---------------------------------------------------------------------------

echo
echo "=============================="
echo "End:          $(date)"
echo "Exit code:    $EXIT_CODE"
echo "=============================="

exit $EXIT_CODE