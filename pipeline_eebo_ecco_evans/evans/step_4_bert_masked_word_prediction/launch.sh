#!/bin/bash
#SBATCH --job-name=bert_preds_tcp
#SBATCH --output=logs/bert_preds_tcp_%j.log
#SBATCH --error=logs/bert_preds_tcp_%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=40
##SBATCH --qos=acc_bsccssh
#SBATCH --qos=acc_debug
#SBATCH --account=bsc100

mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

# ---------------------------------------------------------------------------
# Dataset selection
# ---------------------------------------------------------------------------
# Set to "evans", "eebo", or "ecco". Controls filename prefix, step tag
# (_step3), and default input/output dirs -- see DATASET_DEFAULTS in
# bert_predictions.py. MODELS_BY_DATASET for these three is currently two
# empty placeholder paths -- fill those in in bert_predictions.py once
# models are decided. Until then this job will start, log a clear error
# from each worker, and exit without attempting to load anything.
export BERT_DATASET="evans"

# --- Query words ---------------------------------------------------------
# Leave unset to fall back to query_word_config.sh, or set directly for a
# one-off run:

source "${SLURM_SUBMIT_DIR}/../query_word_config.sh"
export BERT_QUERY_WORDS="$(IFS=,; echo "${QUERY_WORDS[*]}")"

#export BERT_QUERY_WORDS="machine,machines"

# --- Resource / precision knobs ------------------------------------------
export BERT_MAX_GPUS="2"   # match --gres=gpu:N above; also auto-capped to
                            # the number of matched query-word files
export BERT_BATCH_SIZE="256"
export BERT_DTYPE="float16"

mkdir -p "$SLURM_SUBMIT_DIR/logs"
export HF_HOME=/gpfs/scratch/bsc100/paolo/.cache/huggingface
ENV_PATH="/gpfs/scratch/bsc100/paolo/.conda/envs/textmachine-llm-v2"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PATH"

echo "SLURM_JOB_ID     : $SLURM_JOB_ID"
echo "BERT_DATASET     : $BERT_DATASET"
echo "BERT_QUERY_WORDS : $BERT_QUERY_WORDS"
echo "BERT_MAX_GPUS    : $BERT_MAX_GPUS"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

python3 bert_predictions.py