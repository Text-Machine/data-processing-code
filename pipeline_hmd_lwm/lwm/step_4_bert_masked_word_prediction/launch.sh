#!/bin/bash
#SBATCH --job-name=bert_preds
#SBATCH --output=logs/bert_preds_%j.log
#SBATCH --error=logs/bert_preds_%j.err
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=40
##SBATCH --qos=acc_bsccssh
#SBATCH --qos=acc_debug
#SBATCH --account=bsc100
##SBATCH --exclusive

mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

# --- Which dataset this run targets ---------------------------------
# Set to "hmd" or "lwm". Kept as a single override point (rather than
# two near-duplicate launcher files) since everything else below only
# differs by this value. Override at submit time instead of editing
# the file, e.g.:
#   sbatch --export=ALL,DATASET=lwm launch_man_hmd_lwm.sh


# NOTE: deliberately NOT sourcing ../query_word_config.sh here — this
# launcher is a one-off timing test for a single query word, so the word
# list is hardcoded below instead of pulled from the shared config.
#export BERT_QUERY_WORDS=<QUERY-WORD>
#export BERT_QUERY_WORDS="man"

export BERT_MODELS_BASE="/gpfs/projects/bsc100/models/bert_textmachine"

# CHANGED vs the BL Microsoft launcher: input/output point at the
# per-dataset step-5 output produced by filter_hmd_lwm.py, not the BL
# Microsoft "consolidated_metadata" directory — HMD and LWM each get
# their own dir so a "man" run against one dataset never collides with
# (or accidentally completeness-checks against) the other's output.
# Adjust these two paths if your actual step-5 output directory is
# named differently.

export BERT_SUFFIX="spacy"   # must match step 4/5 suffix

export BERT_DATASET=lwm
export BERT_INPUT_DIR=/gpfs/projects/bsc100/textmachine-data/preprocessed_data/consolidated_metadata_lwm
export BERT_OUTPUT_DIR=/gpfs/projects/bsc100/textmachine-data/preprocessed_data/consolidated_metadata_lwm

# --- Performance knobs (same as launch_v3.sh) ---
export BERT_DTYPE="float16"      # float16 | bfloat16 | float32
export BERT_BATCH_SIZE="256"     # start here; try 384/512 if nvidia-smi shows headroom
#export BERT_MAX_GPUS="1"         # must match --gres=gpu:N above — a single
                                  # query word is one file, and each file is
                                  # processed entirely by ONE GPU worker
                                  # (file_queue hands out whole files, never
                                  # shards a file's rows across workers), so
                                  # extra GPUs would just idle here.

mkdir -p "$BERT_OUTPUT_DIR"
export HF_HOME=/gpfs/scratch/bsc100/paolo/.cache/huggingface

ENV_PATH="/gpfs/scratch/bsc100/paolo/.conda/envs/textmachine-llm-v2"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PATH"

echo "SLURM_JOB_ID     : $SLURM_JOB_ID"
echo "BERT_QUERY_WORDS : $BERT_QUERY_WORDS"
echo "BERT_INPUT_DIR   : $BERT_INPUT_DIR"
echo "BERT_OUTPUT_DIR  : $BERT_OUTPUT_DIR"
echo "BERT_SUFFIX      : $BERT_SUFFIX"
echo "BERT_DTYPE       : $BERT_DTYPE"
echo "BERT_BATCH_SIZE  : $BERT_BATCH_SIZE"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Background GPU utilization poller — useful here specifically to see
# whether BERT_BATCH_SIZE=256 is actually saturating the GPUs on a
# mid-sized file like "man", to inform batch size for the full run.
( while true; do
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
      --format=csv,noheader >> "logs/gpu_util_${SLURM_JOB_ID}.log"
    sleep 30
  done ) &
GPU_MON_PID=$!

START_TS=$(date +%s)
echo "START_TS: $START_TS"

# CHANGED: calls the HMD/LWM step-6 script, not run_predictions_optimised_1queryword.py
python3 run_predictions_lwm.py

END_TS=$(date +%s)
echo "END_TS: $END_TS"
echo "ELAPSED_SECONDS: $((END_TS - START_TS))"

kill "$GPU_MON_PID" 2>/dev/null

