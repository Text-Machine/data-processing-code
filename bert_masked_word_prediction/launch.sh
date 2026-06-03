#!/bin/bash
#SBATCH --job-name=bert_preds_b64
#SBATCH --output=logs/bert_preds_b64_%j.log
#SBATCH --error=logs/bert_preds_b64_%j.err
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=80
##SBATCH --qos=acc_bsccssh
#SBATCH --qos=acc_debug
#SBATCH --account=bsc100
#SBATCH --exclusive

mkdir -p logs

export HF_HOME=/gpfs/scratch/bsc100/paolo/.cache/huggingface

ENV_PATH="/gpfs/scratch/bsc100/paolo/.conda/envs/textmachine-llm-v2"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PATH"

cd /home/bsc/bsc204326/text_machine_processing/bert_masked_words_prediction

# Log which GPUs were assigned by SLURM
echo "SLURM_JOB_ID      : $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

python3 run_predictions_v20_batch64.py
