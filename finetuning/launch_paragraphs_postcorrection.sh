#!/bin/bash
#SBATCH --job-name=postcorrect_no_vllm
#SBATCH --account=bsc100
#SBATCH --qos=acc_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# usage: sbatch launch_postcorrect.sh paragraphs.json cleaned.json
set -euo pipefail

IN_FILE="/gpfs/scratch/bsc100/paolo/tm96_finetuning/data-processing-code/finetuning_v2/data/paragraphs/oliver_twist_paragraphs.json"
OUT_FILE="/gpfs/scratch/bsc100/paolo/tm96_finetuning/data-processing-code/finetuning_v2/data/paragraphs/oliver_twist_paragraphs_post_corrected_no_vllm.json"

# Environment
# Load whatever gives you Python + CUDA on MN5 (check `module avail`), e.g.:
# module purge
# module load python/3.11 cuda

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
source /gpfs/scratch/bsc100/paolo/venv_mn5_qwen35_9b_v3/bin/activate  # <- your vLLM virtualenv

# Compute nodes have no internet: use only local files
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="${TMPDIR:-/tmp}/hf_cache"
export VLLM_NO_USAGE_STATS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "Job $SLURM_JOB_ID on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv

python postcorrection_paragraphs.py "$IN_FILE" "$OUT_FILE"