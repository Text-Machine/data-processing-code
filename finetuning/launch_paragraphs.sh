```bash
#!/bin/bash
#SBATCH --job-name=extract_paragraphs
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --qos=gp_debug
#SBATCH -A bsc100

#
# Launcher for extract_paragraphs.py
#
# CPU-only text processing.
#

set -euo pipefail

# --- Config -----------------------------------------------------------

INPUT_DIR="${INPUT_DIR:-/gpfs/projects/bsc100/textmachine-data/downloaded_data_vm}"

BOOK_PATH="${BOOK_PATH:-1830_1839/1838_000932097.jsonl.gz}"

BOOK_NAME="${BOOK_NAME:-oliver_twist}"

OUT_DIR="${OUT_DIR:-./data/paragraphs}"

OUT_FILE="${OUT_FILE:-${OUT_DIR}/${BOOK_NAME}_paragraphs.json}"

N_PARAGRAPHS="${N_PARAGRAPHS:-100}"

TEXT_FIELD="${TEXT_FIELD:-text}"

MIN_CHARS="${MIN_CHARS:-20}"

# Skip front matter / title pages.
SKIP_PAGES="${SKIP_PAGES:-5}"


mkdir -p "${OUT_DIR}"

cd "${SLURM_SUBMIT_DIR:-.}"

echo "[launcher] input_dir:    ${INPUT_DIR}"
echo "[launcher] book_path:    ${BOOK_PATH}"
echo "[launcher] output:       ${OUT_FILE}"
echo "[launcher] n_paragraphs: ${N_PARAGRAPHS}"
echo "[launcher] text_field:   ${TEXT_FIELD}"
echo "[launcher] min_chars:    ${MIN_CHARS}"
echo "[launcher] skip_pages:   ${SKIP_PAGES}"

python3 "extract_paragraphs.py" \
    --input-dir "${INPUT_DIR}" \
    --book-path "${BOOK_PATH}" \
    --output "${OUT_FILE}" \
    --n-paragraphs "${N_PARAGRAPHS}" \
    --text-field "${TEXT_FIELD}" \
    --min-chars "${MIN_CHARS}" \
    --skip-pages "${SKIP_PAGES}"

echo "[launcher] done -> ${OUT_FILE}"
```
