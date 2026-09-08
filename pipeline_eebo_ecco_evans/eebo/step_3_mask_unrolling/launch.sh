#!/bin/bash
#SBATCH --job-name=unroll_masks_tcp
#SBATCH --output=logs/unroll_masks_%j.log
#SBATCH --error=logs/unroll_masks_%j.err
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --qos=gp_debug
#SBATCH --account=bsc100

mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

# ---------------------------------------------------------------------------
# Dataset selection
# ---------------------------------------------------------------------------
# Set to "evans", "eebo", or "ecco". Controls input/output dirs and the
# --pattern used to select this dataset's files -- unroll_masks.py itself
# is unchanged, it's dataset-agnostic (only ever touches sentence /
# masked_sentence fields, same schema across all three via filter_tcp.py).

DATASET="evans"

case "$DATASET" in
    evans|eebo|ecco) ;;
    *)
        echo "ERROR: unknown DATASET='$DATASET' -- must be evans, eebo, or ecco"
        exit 1
        ;;
esac

INPUT_DIR="/gpfs/projects/bsc100/textmachine-data/preprocessed_data/consolidated_metadata_${DATASET}"
OUTPUT_DIR="/gpfs/projects/bsc100/textmachine-data/preprocessed_data/consolidated_metadata_${DATASET}"
SCRIPT="$SLURM_SUBMIT_DIR/unroll_masks.py"

mkdir -p "$OUTPUT_DIR"

# ---------------------------------------------------------------------------
# Query words
# ---------------------------------------------------------------------------
# Same defensive sourcing as the other pipelines: try QUERY_WORDS array
# first, fall back to singular QUERY_WORD, error if neither is set. This
# also determines FILTER_WORDS inside unroll_masks.py via --query-words.

QUERY_CONFIG="${SLURM_SUBMIT_DIR}/query_word_config.sh"

if [[ ! -f "$QUERY_CONFIG" ]]; then
    echo "ERROR: query-word configuration not found: $QUERY_CONFIG"
    exit 1
fi

source "$QUERY_CONFIG"

if [[ ${#QUERY_WORDS[@]} -eq 0 ]]; then
    if [[ -n "${QUERY_WORD:-}" ]]; then
        QUERY_WORDS=("$QUERY_WORD")
    else
        echo "ERROR: neither QUERY_WORDS (array) nor QUERY_WORD (singular) is set in $QUERY_CONFIG"
        exit 1
    fi
fi

echo "============================================================"
echo "TCP Step 3 -- $DATASET"
echo "============================================================"
echo "Script       : $SCRIPT"
echo "Input dir    : $INPUT_DIR"
echo "Output dir   : $OUTPUT_DIR"
echo "Words        : ${QUERY_WORDS[*]}"
echo "Pattern      : ${DATASET}_*.jsonl"
echo "============================================================"

# --pattern restricts this run to ONLY this dataset's step-4 files.
# Important: without it, unroll_masks.py's glob defaults to *.jsonl in
# --input-dir, which combined with a narrower QUERY_WORDS list than a
# previous run risks reprocessing (though not overwriting -- the zero-row
# guard in unroll_masks.py prevents that specifically) files belonging to
# other words. Scoping by dataset prefix here keeps each run fast and
# unambiguous regardless.
python3 "$SCRIPT" \
    --input-dir   "$INPUT_DIR" \
    --output-dir  "$OUTPUT_DIR" \
    --pattern     "${DATASET}_*_step2.jsonl" \
    --query-words "${QUERY_WORDS[@]}"

echo "============================================================"
echo "$DATASET step 3 complete."
echo "============================================================"