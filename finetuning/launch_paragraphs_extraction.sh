#!/bin/bash
#SBATCH --job-name=extract_paragraphs
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00
#SBATCH --qos=gp_debug
#SBATCH -A bsc100

#
# Launcher for extract_paragraphs.py
#
# CPU-only text processing; no GPU required.
#
# Usage:
#   sbatch run_extract_paragraphs.sh
#
# Override configuration from the environment, e.g.:
#
#   BOOK_PATH=/path/to/book.jsonl.gz \
#   BOOK_NAME=oliver_twist \
#   N_PARAGRAPHS=100 \
#   sbatch run_extract_paragraphs.sh
#

set -euo pipefail


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

# Oliver Twist
#BOOK_PATH=/gpfs/scratch/bsc100/paolo/textmachine_books/1830_1839/1838_000932097.jsonl
#BOOK_NAME="${BOOK_NAME:-oliver_twist}"



# Pride and Prejudice
BOOK_PATH=/gpfs/scratch/bsc100/paolo/textmachine_books/1890_1899/1895_000144653.jsonl
BOOK_NAME="${BOOK_NAME:-pride_prejudice}"

OUT_DIR="${OUT_DIR:-./data/paragraphs}"
OUT_FILE="${OUT_FILE:-${OUT_DIR}/${BOOK_NAME}_paragraphs.json}"

N_PARAGRAPHS="${N_PARAGRAPHS:-100}"
TEXT_FIELD="${TEXT_FIELD:-text}"
MIN_CHARS="${MIN_CHARS:-200}"

# Random sampling:
# Set RANDOM_SAMPLE=true to randomly sample paragraphs.
RANDOM_SAMPLE="${RANDOM_SAMPLE:-false}"
SEED="${SEED:-42}"


# ----------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------

mkdir -p "${OUT_DIR}"
mkdir -p logs

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"


# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------

echo "[launcher] input:         ${BOOK_PATH}"
echo "[launcher] book:          ${BOOK_NAME}"
echo "[launcher] output:        ${OUT_FILE}"
echo "[launcher] n_paragraphs:  ${N_PARAGRAPHS}"
echo "[launcher] text_field:    ${TEXT_FIELD}"
echo "[launcher] min_chars:     ${MIN_CHARS}"
echo "[launcher] random_sample: ${RANDOM_SAMPLE}"
echo "[launcher] seed:          ${SEED}"


# ----------------------------------------------------------------------
# Build command
# ----------------------------------------------------------------------

CMD=(
    python3 extract_paragraphs.py
    --input "${BOOK_PATH}"
    --output "${OUT_FILE}"
    --n-paragraphs "${N_PARAGRAPHS}"
    --text-field "${TEXT_FIELD}"
    --min-chars "${MIN_CHARS}"
)

if [[ "${RANDOM_SAMPLE}" == "true" ]]; then
    CMD+=(
        --random-sample
        --seed "${SEED}"
    )
fi


# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------

echo "[launcher] running:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"

echo "[launcher] done -> ${OUT_FILE}"
