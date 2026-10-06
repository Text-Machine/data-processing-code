#!/usr/bin/env python3
"""
Unroll Multi-Mask Rows — Pre-processing Step 5
==============================================

Reads ONLY step-4 JSONL files corresponding to the query words supplied
via --query-words.

For example:

    --query-words telegraph electric

will process ONLY:

    bl_microsoft_step_4_telegraph_spacy.jsonl
    bl_microsoft_step_4_electric_spacy.jsonl

and produce:

    bl_microsoft_telegraph_spacy_step_5.jsonl
    bl_microsoft_electric_spacy_step_5.jsonl

No other JSONL files in the directory are touched.

For rows where the same target word appears masked multiple times in
masked_sentence, emit one output row per [MASK] token, with only that
mask retained and all other masks restored to their original word.

Rows with exactly one [MASK] are passed through as-is, with:
    mask_index = 0
    total_masks = 1

Rows where the masked word is not one of the requested query words
are skipped.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
import re


# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("unroll_masks.log"),
        logging.StreamHandler(),
    ],
)

log = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def normalise_word(word: str) -> str:
    """Normalise a word for case-insensitive comparison."""
    return word.strip(".,;:!?\"'()-").lower()


def row_matches_filter(row: dict, filter_words: set[str]) -> bool:
    """
    Return True if at least one [MASK] position in masked_sentence
    corresponds to one of filter_words in sentence.
    """

    masked_sentence = row.get("masked_sentence") or ""
    original_sentence = row.get("sentence") or ""

    masked_words = masked_sentence.split()
    original_words = original_sentence.split()

    for i, word in enumerate(masked_words):
        if "[MASK]" in word and i < len(original_words):
            original_word = normalise_word(original_words[i])

            if original_word in filter_words:
                return True

    return False


def unroll_row(row: dict, filter_words: set[str]) -> list[dict]:
    """
    Expand a row into one output row per [MASK] token.

    Each output row contains exactly one [MASK].

    Only masks whose original word is in filter_words are retained.
    """

    masked_sentence = row.get("masked_sentence") or ""
    original_sentence = row.get("sentence") or ""

    masked_words = masked_sentence.split()
    original_words = original_sentence.split()

    # ---------------------------------------------------------------
    # Find all mask positions
    # ---------------------------------------------------------------

    mask_positions = []

    for word_idx, word in enumerate(masked_words):
        if "[MASK]" in word:
            mask_positions.append(word_idx)

    n_masks = len(mask_positions)

    if n_masks == 0:
        return []

    # ---------------------------------------------------------------
    # Unroll each mask
    # ---------------------------------------------------------------

    unrolled = []

    for target_seq_idx, target_word_idx in enumerate(mask_positions):

        # -----------------------------------------------------------
        # Determine the original word corresponding to this mask
        # -----------------------------------------------------------

        if target_word_idx < len(original_words):
            original_word = normalise_word(
                original_words[target_word_idx]
            )
        else:
            original_word = ""

        # Only keep requested query words
        if original_word not in filter_words:
            continue

        # -----------------------------------------------------------
        # Build sentence with exactly one mask
        # -----------------------------------------------------------

        new_words = []
        mask_seq_counter = 0

        for word_idx, word in enumerate(masked_words):

            if "[MASK]" in word:

                if mask_seq_counter == target_seq_idx:
                    # Keep this mask
                    new_words.append(word)

                else:
                    # Restore the original word
                    if word_idx < len(original_words):
                        restored = word.replace(
                            "[MASK]",
                            original_words[word_idx],
                        )
                    else:
                        restored = word.replace("[MASK]", "")

                    new_words.append(restored)

                mask_seq_counter += 1

            else:
                new_words.append(word)

        new_masked_sentence = " ".join(new_words)

        # -----------------------------------------------------------
        # Safety check
        # -----------------------------------------------------------

        remaining = new_masked_sentence.count("[MASK]")

        if remaining != 1:
            log.warning(
                "Unexpected mask count after unrolling "
                "(remaining=%s, mask_index=%s, article_id=%s). "
                "Skipping variant.",
                remaining,
                target_seq_idx,
                row.get("article_id", "?"),
            )
            continue

        # -----------------------------------------------------------
        # Create output row
        # -----------------------------------------------------------

        out = dict(row)

        out["masked_sentence"] = new_masked_sentence
        out["mask_index"] = target_seq_idx
        out["total_masks"] = n_masks

        unrolled.append(out)

    return unrolled


# -------------------------------------------------------------------
# File naming
# -------------------------------------------------------------------

def make_output_name(input_path: Path) -> str:
    """
    Convert:

        bl_microsoft_step_4_telegraph_spacy.jsonl

    into:

        bl_microsoft_telegraph_spacy_step_5.jsonl
    """

    stem = input_path.stem

    # Remove "_step_4"
    stem = re.sub(r"_step_4", "", stem)

    # Remove any accidental existing step tag
    stem = re.sub(r"_step_\d+", "", stem)

    # Append step 5
    stem = f"{stem}_step_5"

    return stem + input_path.suffix


# -------------------------------------------------------------------
# Completeness check
# -------------------------------------------------------------------

def is_file_complete(
    input_path: Path,
    output_path: Path,
    filter_words: set[str],
) -> bool:
    """
    Check whether an existing output has the expected number of rows.
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

            if row_matches_filter(row, filter_words):
                expected += len(unroll_row(row, filter_words))

    actual = 0

    with open(output_path, "r", encoding="utf-8") as f:
        actual = sum(1 for line in f if line.strip())

    return expected == actual


# -------------------------------------------------------------------
# Process one file
# -------------------------------------------------------------------

def process_file(
    input_path: Path,
    output_dir: Path,
    filter_words: set[str],
):
    """
    Process one step-4 input file.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / make_output_name(input_path)

    # ---------------------------------------------------------------
    # Never allow input == output
    # ---------------------------------------------------------------

    if input_path.resolve() == output_path.resolve():
        log.error(
            "Input and output paths are identical: %s",
            input_path,
        )
        raise RuntimeError(
            f"Refusing to overwrite input file: {input_path}"
        )

    # ---------------------------------------------------------------
    # Existing complete output
    # ---------------------------------------------------------------

    if is_file_complete(
        input_path,
        output_path,
        filter_words,
    ):
        log.info(
            "Skipping (already complete): %s",
            output_path,
        )
        return

    # ---------------------------------------------------------------
    # Existing incomplete output
    # ---------------------------------------------------------------

    if output_path.exists():
        log.warning(
            "Incomplete output found, reprocessing: %s",
            output_path,
        )

    # ---------------------------------------------------------------
    # Process
    # ---------------------------------------------------------------

    log.info("Processing:")
    log.info("  Input : %s", input_path)
    log.info("  Output: %s", output_path)

    rows_in = 0
    rows_out = 0
    rows_skipped = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line_num, line in enumerate(fin, 1):

            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)

            except json.JSONDecodeError as e:
                log.warning(
                    "Line %s: JSON parse error — %s",
                    line_num,
                    e,
                )
                continue

            rows_in += 1

            if not row_matches_filter(row, filter_words):
                rows_skipped += 1
                continue

            unrolled = unroll_row(
                row,
                filter_words,
            )

            for out_row in unrolled:

                fout.write(
                    json.dumps(
                        out_row,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                rows_out += 1

    log.info(
        "Finished: %s "
        "(%s read, %s written, %s skipped)",
        output_path.name,
        rows_in,
        rows_out,
        rows_skipped,
    )


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Unroll multi-mask rows from step-4 files for "
            "the specified query words."
        )
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing step-4 JSONL files.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where step-5 JSONL files will be written.",
    )

    parser.add_argument(
        "--suffix",
        default="spacy",
        help=(
            "Processing suffix used by step 4 "
            "(e.g. spacy or regex)."
        ),
    )

    parser.add_argument(
        "--query-words",
        nargs="+",
        required=True,
        help=(
            "Query words to process. "
            "Example: --query-words telegraph electric"
        ),
    )

    return parser.parse_args()


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():

    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    filter_words = {
        normalise_word(word)
        for word in args.query_words
    }

    # ---------------------------------------------------------------
    # Validate directories
    # ---------------------------------------------------------------

    if not input_dir.exists():
        log.error(
            "Input directory does not exist: %s",
            input_dir,
        )
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # Log configuration
    # ---------------------------------------------------------------

    log.info(
        "Query words: %s",
        sorted(filter_words),
    )

    log.info(
        "Input dir:    %s",
        input_dir,
    )

    log.info(
        "Output dir:   %s",
        output_dir,
    )

    log.info(
        "Suffix:       %s",
        args.suffix,
    )

    # ---------------------------------------------------------------
    # IMPORTANT:
    #
    # Only construct the expected step-4 filenames.
    #
    # We do NOT glob *.jsonl.
    # ---------------------------------------------------------------

    input_files = []

    for word in sorted(filter_words):

        input_filename = (
            f"bl_microsoft_step_4_{word}_{args.suffix}.jsonl"
        )

        input_path = input_dir / input_filename

        if not input_path.exists():

            log.warning(
                "Expected input file does not exist for query word "
                "'%s': %s",
                word,
                input_path,
            )

            continue

        input_files.append(input_path)

    # ---------------------------------------------------------------
    # Nothing to process
    # ---------------------------------------------------------------

    if not input_files:

        log.error(
            "No input files found for query words: %s",
            sorted(filter_words),
        )

        sys.exit(1)

    # ---------------------------------------------------------------
    # Show exactly what will be processed
    # ---------------------------------------------------------------

    log.info(
        "Found %s input file(s):",
        len(input_files),
    )

    for input_path in input_files:
        log.info(
            "  %s",
            input_path.name,
        )

    # ---------------------------------------------------------------
    # Process ONLY those files
    # ---------------------------------------------------------------

    for input_path in input_files:

        process_file(
            input_path=input_path,
            output_dir=output_dir,
            filter_words=filter_words,
        )

    log.info("All done.")


# -------------------------------------------------------------------

if __name__ == "__main__":
    main()