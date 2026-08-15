#!/bin/bash
#SBATCH --job-name=bert_preds
#SBATCH --output=logs/bert_preds%j.log
#SBATCH --error=logs/bert_preds%j.err
#SBATCH --time=00:27:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=80
##SBATCH --qos=acc_bsccssh
#SBATCH --qos=acc_debug
#SBATCH --account=bsc100

mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

export BERT_MODELS_BASE="/gpfs/projects/bsc100/models/bert_textmachine"
export BERT_INPUT_DIR="/gpfs/projects/bsc100/textmachine-data/preprocessed_data/consolidated_metadata_blmicrosoft"
export BERT_OUTPUT_DIR="/gpfs/projects/bsc100/textmachine-data/preprocessed_data/consolidated_metadata_blmicrosoft"
export BERT_SUFFIX="spacy"

export BERT_DTYPE="float16"
export BERT_BATCH_SIZE="256"
export BERT_MAX_GPUS="4"   # FIX: must match --gres=gpu:4 above -- this is
                            # what drives shards_per_file =
                            # ceil(MAX_GPUS/n_files), so with 2 words this
                            # becomes ceil(4/2) = 2 shards per word, using
                            # all 4 GPUs. Leaving this at 2 (as in the
                            # non-sharded launcher) caps n_gpus_requested
                            # down to 2 before shard count is even
                            # computed, silently disabling sharding
                            # entirely (shards_per_file = ceil(2/2) = 1) --
                            # that was the bug causing identical runtime
                            # and 2 idle GPUs.

mkdir -p "$BERT_OUTPUT_DIR"
export HF_HOME=/gpfs/scratch/bsc100/paolo/.cache/huggingface
ENV_PATH="/gpfs/scratch/bsc100/paolo/.conda/envs/textmachine-llm-v2"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PATH"

echo "SLURM_JOB_ID     : $SLURM_JOB_ID"
echo "BERT_QUERY_WORDS : $BERT_QUERY_WORDS"
echo "BERT_MAX_GPUS    : $BERT_MAX_GPUS"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

( while true; do
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
--format=csv,noheader >> "logs/gpu_util_${SLURM_JOB_ID}.log"
sleep 15
done ) &
GPU_MON_PID=$!

START_TS=$(date +%s)
echo "START_TS: $START_TS"
python3 run_predictions_faster_multigpu_sharded.py
END_TS=$(date +%s)
echo "END_TS: $END_TS"
echo "ELAPSED_SECONDS: $((END_TS - START_TS))"

kill "$GPU_MON_PID" 2>/dev/null
echo "Check logs/bert_predictions_batch64_*.log for the TIMING SUMMARY table --"
echo "overlapping start/end windows across different GPU labels confirm parallelism."