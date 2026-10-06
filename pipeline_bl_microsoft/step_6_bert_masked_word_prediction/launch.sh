#!/bin/bash

#SBATCH --job-name=bert_preds_b64

#SBATCH --output=logs/bert_preds_b64_%j.log
#SBATCH --error=logs/bert_preds_b64_%j.err

#SBATCH --time=00:30:00

#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=40

#SBATCH --qos=acc_debug
#SBATCH --account=bsc100

#SBATCH --exclusive


# -------------------------------------------------------------------
# Setup
# -------------------------------------------------------------------

mkdir -p logs

cd "$SLURM_SUBMIT_DIR"

source "${SLURM_SUBMIT_DIR}/../query_word_config.sh"


# -------------------------------------------------------------------
# Input / output
# -------------------------------------------------------------------

export BERT_INPUT_DIR="/gpfs/projects/bsc100/textmachine-data/preprocessed_data/consolidated_metadata_blmicrosoft"

export BERT_OUTPUT_DIR="/gpfs/projects/bsc100/textmachine-data/preprocessed_data/consolidated_metadata_blmicrosoft"

export BERT_SUFFIX="spacy"


# -------------------------------------------------------------------
# Query words
#
# query_word_config.sh:
#
#     QUERY_WORDS=(telegraph electric)
#
# becomes:
#
#     BERT_QUERY_WORDS=telegraph,electric
# -------------------------------------------------------------------

export BERT_QUERY_WORDS="$(IFS=,; echo "${QUERY_WORDS[*]}")"


# -------------------------------------------------------------------
# Models
# -------------------------------------------------------------------

export BERT_MODELS_BASE="/gpfs/projects/bsc100/models/bert_textmachine"

export BERT_MODEL_LWM="${BERT_MODELS_BASE}/bert_1760_1900"

export BERT_MODEL_BASE_UNCASED="${BERT_MODELS_BASE}/bert-base-uncased"

export BERT_MODEL_ECCOBERT="${BERT_MODELS_BASE}/eccobert_base_cased_v1"

# -------------------------------------------------------------------
# Hugging Face cache
# -------------------------------------------------------------------

export HF_HOME="/gpfs/scratch/bsc100/paolo/.cache/huggingface"


# -------------------------------------------------------------------
# Performance
# -------------------------------------------------------------------

export BERT_BATCH_SIZE=256

export BERT_DTYPE="float16"

export BERT_MAX_GPUS=2


# -------------------------------------------------------------------
# Environment
# -------------------------------------------------------------------

ENV_PATH="/gpfs/scratch/bsc100/paolo/.conda/envs/textmachine-llm-v2"

source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate "$ENV_PATH"


# -------------------------------------------------------------------
# Diagnostics
# -------------------------------------------------------------------

echo "============================================================"
echo "BERT prediction job"
echo "============================================================"

echo "SLURM_JOB_ID     : $SLURM_JOB_ID"

echo "BERT_QUERY_WORDS : $BERT_QUERY_WORDS"

echo "BERT_SUFFIX      : $BERT_SUFFIX"

echo "BERT_INPUT_DIR   : $BERT_INPUT_DIR"

echo "BERT_OUTPUT_DIR  : $BERT_OUTPUT_DIR"

echo ""

echo "Models:"

echo "  LWM       : $BERT_MODEL_LWM"

echo "  BERT base : $BERT_MODEL_BASE_UNCASED"

echo "  EccoBERT  : $BERT_MODEL_ECCOBERT"

echo ""

echo "Batch size       : $BERT_BATCH_SIZE"

echo "Dtype            : $BERT_DTYPE"

echo "Max GPUs         : $BERT_MAX_GPUS"

echo ""

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

echo "============================================================"


# -------------------------------------------------------------------
# Run
# -------------------------------------------------------------------

python3 run_predictions.py