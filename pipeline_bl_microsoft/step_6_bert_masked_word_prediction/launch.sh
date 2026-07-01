#!/bin/bash
#SBATCH --job-name=bert_preds_b64
#SBATCH --output=logs/bert_preds_b64_%j.log
#SBATCH --error=logs/bert_preds_b64_%j.err
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=80
##SBATCH --qos=acc_bsccssh
#SBATCH --qos=acc_debug
#SBATCH --account=bsc100
#SBATCH --exclusive

mkdir -p logs

# ---------------------------------------------------------------
# Paths — edit here
# ---------------------------------------------------------------
export BERT_MODELS_BASE="/gpfs/projects/bsc100/models/bert_textmachine"

export BERT_INPUT_DIR="/gpfs/projects/bsc100/textmachine-data/preprocessed_data/consolidated_metadata"
#export BERT_INPUT_GLOB="bl_microsoft_*_spacy_step_5.jsonl"

export BERT_OUTPUT_DIR="/gpfs/projects/bsc100/textmachine-data/preprocessed_data/consolidated_metadata"


mkdir -p "$BERT_OUTPUT_DIR"
# ---------------------------------------------------------------

export HF_HOME=/gpfs/scratch/bsc100/paolo/.cache/huggingface
ENV_PATH="/gpfs/scratch/bsc100/paolo/.conda/envs/textmachine-llm-v2"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PATH"

cd "$SLURM_SUBMIT_DIR"

echo "SLURM_JOB_ID        : $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "BERT_MODELS_BASE    : $BERT_MODELS_BASE"
echo "BERT_INPUT_DIR      : $BERT_INPUT_DIR"
echo "BERT_INPUT_GLOB     : $BERT_INPUT_GLOB"
echo "BERT_OUTPUT_DIR     : $BERT_OUTPUT_DIR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

python3 run_predictions.py
