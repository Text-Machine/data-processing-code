#!/usr/bin/env python3
"""
Unroll Multi-Mask Rows — Pre-processing Step
=============================================
Run this BEFORE bert_predictions.py.

Dataset-agnostic: works on step-4 output from any of the BL Microsoft,
HMD, or LWM pipelines, since all three produce the same row schema
(`sentence`, `masked_sentence`, plus dataset-specific metadata columns).
The only thing that varies across datasets is the row-identifier field
used for log messages (record_id for BL Microsoft, article_id for HMD/
LWM) -- handled by get_row_id() below, everything else is identical
processing regardless of source.

For rows where the same target word appears masked multiple times in
`masked_sentence`, this script emits one output row per [MASK] token,
each with only that one mask retained and all other masks restored to
their original word from `sentence`.

Example input row (2 masks):
  sentence:        "THE TO MACHINE MAKERS ... Sewing MACHINE for making"
  masked_sentence: "THE TO [MASK] MAKERS ... Sewing [MASK] for making"

Output rows (2 rows):
  Row 1 — first mask kept, second restored:
    masked_sentence: "THE TO [MASK] MAKERS ... Sewing MACHINE for making"
    mask_index: 0
    total_masks: 2

  Row 2 — first mask restored, second kept:
    masked_sentence: "THE TO MACHINE MAKERS ... Sewing [MASK] for making"
    mask_index: 1
    total_masks: 2

Rows with exactly one [MASK] are passed through as-is (mask_index=0,
total_masks=1) so that downstream code only needs to read the unrolled
output.

Rows whose masked word is not in FILTER_WORDS are silently dropped,
matching the behaviour of bert_predictions.py.

Output fields added / updated per unrolled row:
  mask_index   — which [MASK] position this row represents (0-based)
  total_masks  — how many [MASK] tokens were in the original sentence
  masked_sentence — rewritten to contain exactly one [MASK]
  (all other fields from the original row are preserved unchanged)

Usage:
  python3 unroll_masks.py --input-dir <path> --output-dir <path>
  python3 unroll_masks.py --input-dir <path> --output-dir <path> --pattern "bl_microsoft_*_spacy*"
  python3 unroll_masks.py --input-dir <path> --output-dir <path> --pattern "lwm_*"
  python3 unroll_masks.py --input-dir <path> --output-dir <path> --pattern "hmd_*"
"""

import argparse
import fnmatch
import json
import logging
import sys
from pathlib import Path
import re


# -------------------------------------------------------------------
# Logging setup
# -------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("unroll_masks.log"),
        logging.StreamHandler()
    ]
)

log = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Config — keep in sync with the step-4 script(s) and bert_predictions.py
# -------------------------------------------------------------------

# Must match the query words used in step 4 / bert_predictions.py
#FILTER_WORDS = {"machine", "machines", "slave", "slaves"}


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def get_row_id(row: dict) -> str:
    """
    Dataset-agnostic row identifier for log messages: LWM/HMD rows carry
    `article_id`, BL Microsoft rows carry `record_id`. Falls back to '?'
    if neither is present.
    """
    return row.get("article_id") or row.get("record_id") or "?"


def row_matches_filter(row: dict) -> bool:
    """
    Returns True if at least one [MASK] position in `masked_sentence`
    corresponds to a FILTER_WORD in `sentence`.

    Matching is case-insensitive; leading/trailing punctuation is
    stripped from the original word before comparison.
    """

    masked_sentence = row.get("masked_sentence") or ""
    original_sentence = row.get("sentence") or ""

    masked_words = masked_sentence.split()
    original_words = original_sentence.split()

    for i, word in enumerate(masked_words):
        if "[MASK]" in word and i < len(original_words):
            original_word = original_words[i].strip(".,;:!?\"'()-").lower()
            if original_word in FILTER_WORDS:
                return True

    return False


def unroll_row(row: dict) -> list[dict]:
    """
    Expand a row into one output row per [MASK] token.

    Each output row has:
      - exactly one [MASK] in `masked_sentence`
      - `mask_index`  : which mask position this row represents (0-based)
      - `total_masks` : number of [MASK] tokens in the original sentence

    For rows with a single [MASK], returns a list with a single row
    (mask_index=0, total_masks=1) — no structural change, just the two
    bookkeeping fields added.

    Strategy for multi-mask rows:
      For each target mask position i:
        - keep [MASK] at position i
        - replace every other [MASK] at position j with the original
          word from `sentence` at that same token index
        - only emit the unrolled row if the *kept* mask corresponds to
          a FILTER_WORD (so non-target masks that happen to be
          filter-words in other positions don't generate spurious rows)
    """

    masked_sentence = row.get("masked_sentence") or ""
    original_sentence = row.get("sentence") or ""

    masked_words = masked_sentence.split()
    original_words = original_sentence.split()

    # Collect word indices of each [MASK] token
    mask_positions: list[int] = []

    for word_idx, word in enumerate(masked_words):
        if "[MASK]" in word:
            mask_positions.append(word_idx)

    n_masks = len(mask_positions)

    # ----------------------------------------------------------------
    # Single-mask fast path
    # ----------------------------------------------------------------

    if n_masks <= 1:
        out = dict(row)
        out["mask_index"] = 0
        out["total_masks"] = max(n_masks, 0)
        return [out]

    # ----------------------------------------------------------------
    # Multi-mask unrolling
    # ----------------------------------------------------------------

    unrolled = []

    for target_seq_idx, target_word_idx in enumerate(mask_positions):

        # Check the kept mask is a filter word — skip if not
        if target_word_idx < len(original_words):
            original_word = (
                original_words[target_word_idx]
                .strip(".,;:!?\"'()-")
                .lower()
            )
        else:
            original_word = ""

        if original_word not in FILTER_WORDS:
            log.debug(
                f"  mask_index={target_seq_idx}: "
                f"original word '{original_word}' not in FILTER_WORDS, "
                f"skipping this unrolled variant."
            )
            continue

        # Build the rewritten word list
        new_words: list[str] = []
        mask_seq_counter = 0

        for word_idx, word in enumerate(masked_words):

            if "[MASK]" in word:

                if mask_seq_counter == target_seq_idx:
                    # Retain this mask
                    new_words.append(word)
                else:
                    # Restore original token
                    if word_idx < len(original_words):
                        restored = word.replace(
                            "[MASK]",
                            original_words[word_idx]
                        )
                    else:
                        # Safety fallback: drop the [MASK] placeholder
                        restored = word.replace("[MASK]", "")

                    new_words.append(restored)

                mask_seq_counter += 1

            else:
                new_words.append(word)

        new_masked_sentence = " ".join(new_words)

        # Sanity check: exactly one [MASK] must remain
        remaining = new_masked_sentence.count("[MASK]")

        if remaining != 1:
            log.warning(
                f"  Unexpected mask count after unrolling "
                f"(remaining={remaining}, "
                f"mask_index={target_seq_idx}, "
                f"row_id={get_row_id(row)}), "
                f"skipping this variant."
            )
            continue

        out = dict(row)
        out["masked_sentence"] = new_masked_sentence
        out["mask_index"] = target_seq_idx
        out["total_masks"] = n_masks

        unrolled.append(out)

        log.debug(
            f"  Unrolled mask_index={target_seq_idx}: "
            f"{new_masked_sentence[:80]}..."
        )

    return unrolled


def is_file_complete(input_path: Path, output_path: Path) -> bool:
    """
    Returns True if the output file exists and its line count matches
    the expected number of unrolled rows from the input.
    """

    if not output_path.exists():
        return False

    expected = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row_matches_filter(row):
                expected += len(unroll_row(row))

    with open(output_path, "r", encoding="utf-8") as f:
        actual = sum(1 for ln in f if ln.strip())

    return expected == actual


def make_output_name(filepath: Path) -> str:
    """
    Dataset-agnostic naming: strips any existing "_step_N" tag and
    appends "_step_5", regardless of the dataset prefix. Works for any
    of the three datasets, e.g.:

        bl_microsoft_step_4_slaves_spacy.jsonl -> bl_microsoft_slaves_spacy_step_5.jsonl
        lwm_machine.jsonl                      -> lwm_machine_step_5.jsonl
        hmd_slaves.jsonl                       -> hmd_slaves_step_5.jsonl
    """

    stem = filepath.stem

    # Remove any existing step tag
    stem = re.sub(r"_step_\d+", "", stem)

    # Append the new processing stage
    stem = f"{stem}_step_5"

    return stem + filepath.suffix

def process_file(filepath: Path, output_dir: Path, suffix: str = ""):
    """
    Read one .jsonl file, filter and unroll, write to output_dir.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / make_output_name(filepath)

    if is_file_complete(filepath, out_path):
        log.info(f"  Skipping (already complete): {out_path}")
        return

    if out_path.exists():
        log.warning(
            f"  Incomplete output found, reprocessing: {out_path}"
        )

    log.info(f"  Processing: {filepath}")

    rows_in = 0
    rows_out = 0
    rows_skipped = 0

    with open(filepath, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:

        for line_num, line in enumerate(fin, 1):

            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                log.warning(f"  Line {line_num}: JSON parse error — {e}")
                continue

            rows_in += 1

            if not row_matches_filter(row):
                rows_skipped += 1
                continue

            unrolled = unroll_row(row)

            for out_row in unrolled:
                fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                rows_out += 1

    log.info(
        f"  Written: {out_path} "
        f"({rows_in} read, {rows_out} written, {rows_skipped} skipped)"
    )


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unroll multi-mask rows in .jsonl files (BL Microsoft, HMD, or LWM step-4 output)."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing input .jsonl files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where unrolled .jsonl files will be written.",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Suffix to append to each output filename stem (e.g. '_step_5').",
    )
    parser.add_argument(
        "--pattern",
        default="*.jsonl",
        help=(
            "Glob pattern to select input files within --input-dir "
            "(default: '*.jsonl'). "
            "Examples: 'bl_microsoft_*_spacy*.jsonl', 'lwm_*.jsonl', 'hmd_*.jsonl'."
        ),
    )
    parser.add_argument(
        "--query-words",
        nargs="+",
        default=None,
        metavar="WORD",
        help=(
            "Override FILTER_WORDS for this run (e.g. --query-words machine machines). "
            "Must match the word list used in step 4 for the files being processed. "
            "If omitted, the hardcoded FILTER_WORDS default is used."
        ),
    )
    return parser.parse_args()


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():

    args = parse_args()

    if args.query_words:
        global FILTER_WORDS
        FILTER_WORDS = {w.strip().lower() for w in args.query_words}

    input_path = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    log.info(f"Filter words: {sorted(FILTER_WORDS)}")
    log.info(f"Input dir:    {input_path}")
    log.info(f"Output dir:   {output_dir}")
    log.info(f"File pattern: {args.pattern}")

    if not input_path.exists():
        log.error(f"Input directory does not exist: {input_path}")
        sys.exit(1)

    # Glob with the user-supplied pattern; fall back to *.jsonl default
    all_jsonl = sorted(input_path.glob("*.jsonl"))
    jsonl_files = [
        f for f in all_jsonl
        if fnmatch.fnmatch(f.name, args.pattern)
    ] if args.pattern != "*.jsonl" else all_jsonl

    if not jsonl_files:
        log.warning(
            f"No files matching '{args.pattern}' found in: {input_path}"
        )
        sys.exit(0)

    log.info(f"Found {len(jsonl_files)} file(s) matching '{args.pattern}':")
    for f in jsonl_files:
        log.info(f"  {f.name}")

    for filepath in jsonl_files:
        process_file(filepath, output_dir, args.pattern)

    log.info("\nAll done.")


if __name__ == "__main__":
    main()