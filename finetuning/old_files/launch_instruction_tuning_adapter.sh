#!/bin/bash
#SBATCH --job-name=qwen-instruction_tuning
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

# Use the vLLM-enabled venv (see create_venv_qwen35_9b.sh -- vllm install
# uncommented, transformers pinned to a version that actually supports
# Qwen3.5's Gated DeltaNet architecture).
source /gpfs/scratch/bsc100/paolo/venv_mn5_qwen35_9b_v3/bin/activate

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT="$SLURM_SUBMIT_DIR/instruction_tuning_adapter.py"

python "$SCRIPT"

EXIT_CODE=$?

echo "=============================="
echo "End:          $(date)"
echo "Exit code:    $EXIT_CODE"
echo "=============================="

exit $EXIT_CODE