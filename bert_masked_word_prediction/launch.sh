#!/bin/bash
#SBATCH --job-name=bert_mask_preds
#SBATCH --output=logs/bert_preds_%j.log
#SBATCH --error=logs/bert_preds_%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
##SBATCH --qos=acc_bsccssh
#SBATCH --qos=acc_debug
#SBATCH --account=bsc100
#SBATCH --exclusive

# Create logs dir if it doesn't exist
mkdir -p logs

export HF_HOME=/gpfs/scratch/bsc100/paolo/.cache/huggingface
ENV_PATH="/gpfs/scratch/bsc100/paolo/.conda/envs/textmachine-llm"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PATH"


# Move to working directory
cd $SLURM_SUBMIT_DIR


# Run
python3 run_predictions_no_prev_sentence.py
