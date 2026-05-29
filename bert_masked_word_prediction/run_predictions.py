#!/usr/bin/env python3
"""
BERT Masked Word Prediction Script
Adds top-10 predictions from 3 BERT models to .jsonl files.

Fixes:
- Proper handling of multiple [MASK] tokens
- Robust truncation to BERT's 512-token limit
- Safer reconstruction logic
- Better exception handling
- Fixed: skip_special_tokens=False in truncate_to_bert_limit to
  preserve [MASK] token during decoding
- Only processes rows where the masked word is in FILTER_WORDS
- Unwrapped extra bracket layer on single-mask predictions
"""

import json
import logging
from pathlib import Path

import torch
from transformers import pipeline


# -------------------------------------------------------------------
# Logging setup
# -------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("bert_predictions.log"),
        logging.StreamHandler()
    ]
)

log = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------

MODELS_BASE = "/gpfs/projects/bsc100/models/bert_textmachine"

MODELS = {
    "pred_bert_1760_1850":    f"{MODELS_BASE}/bert_1760_1850",
    "pred_bert_1890_1900":    f"{MODELS_BASE}/bert_1890_1900",
    "pred_bert_contemporary": f"{MODELS_BASE}/bert-base-uncased",
}

INPUT_DIRS = [
    "/gpfs/projects/bsc100/textmachine-data/filtered_data/lwm_hmd",
    "/gpfs/projects/bsc100/textmachine-data/filtered_data/lwm_hmd_high_memory",
    "/gpfs/projects/bsc100/textmachine-data/filtered_data/bl_microsoft",
]

OUTPUT_BASE = (
    "/gpfs/projects/bsc100/textmachine-data/"
    "filtered_data_predictions_bert_mask_predicted_repeat_2"
)

# Only process rows where the original masked word (from 'sentence')
# matches one of these keywords (case-insensitive, exact word match).
FILTER_WORDS = {"machine", "machines", "slave", "slaves"}

DEVICE = 0 if torch.cuda.is_available() else -1

TOP_K = 10
BERT_MAX_TOKENS = 512


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def check_gpu_health():
    """Simple GPU sanity check."""

    if not torch.cuda.is_available():
        return

    try:
        log.info("Running GPU health check...")

        x = torch.ones(1000, 1000, device="cuda")
        _ = x @ x

        del x
        torch.cuda.empty_cache()

        log.info("GPU health check passed.")

    except Exception as e:
        log.error(f"GPU health check FAILED: {e}")
        log.error("Aborting — try a different node or run on CPU.")
        raise SystemExit(1)


def is_file_complete(input_path: Path, output_path: Path) -> bool:
    """
    Check output has the same number of non-blank lines as input.
    Only counts lines that would pass the keyword filter, since
    non-matching rows are skipped and not written to output.
    """

    if not output_path.exists():
        return False

    def count_matching_lines(path):
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    if row_matches_filter(row):
                        count += 1
                except json.JSONDecodeError:
                    pass
        return count

    def count_output_lines(path):
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    return count_matching_lines(input_path) == count_output_lines(output_path)


def row_matches_filter(row: dict) -> bool:
    """
    Returns True if any of the original masked words (from 'sentence')
    at [MASK] positions matches one of the FILTER_WORDS.

    Matching is case-insensitive and strips punctuation from word edges
    so that e.g. "machine," or "slaves." are still caught.
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


def truncate_to_bert_limit(
    text: str,
    tokenizer,
    max_tokens: int = BERT_MAX_TOKENS
) -> str:
    """
    Robust truncation using tokenizer encoding rather than manual
    token counting.

    Guarantees final sequence length <= BERT max length.

    NOTE: skip_special_tokens=False is intentional — setting it to
    True would strip [MASK] along with [CLS]/[SEP], causing the
    fill-mask pipeline to raise "No mask_token found on the input".
    [CLS] and [SEP] are removed manually below instead.
    """

    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_tokens,
        add_special_tokens=True,
        return_attention_mask=False,
        return_token_type_ids=False,
    )

    decoded = tokenizer.decode(
        encoded["input_ids"],
        skip_special_tokens=False,          # preserve [MASK]
        clean_up_tokenization_spaces=True,
    )

    # Strip [CLS] / [SEP] wrapper tokens but leave [MASK] intact
    decoded = decoded.replace("[CLS]", "").replace("[SEP]", "").strip()

    return decoded


def predict(pipe, row: dict, tokenizer) -> list:
    """
    Handle single or multiple [MASK] tokens.

    Returns a flat list of (token, score) pairs for single-mask rows,
    or a list-of-lists for multi-mask rows:
        Single: [('machine', 0.23), ('engine', 0.18), ...]
        Multi:  [[('machine', 0.23), ...], [('sewing', 0.18), ...]]
    """

    prev_sentence = row.get("prev_sentence") or ""
    masked_sentence = row.get("masked_sentence") or ""
    next_sentence = row.get("next_sentence") or ""
    original_sentence = row.get("sentence") or ""

    n_masks = masked_sentence.count("[MASK]")

    if n_masks == 0:
        return []

    # ----------------------------------------------------------------
    # SINGLE MASK
    # ----------------------------------------------------------------

    if n_masks == 1:

        text = (
            f"{prev_sentence} "
            f"{masked_sentence} "
            f"{next_sentence}"
        ).strip()

        text = truncate_to_bert_limit(text, tokenizer)

        if "[MASK]" not in text:
            log.warning(
                f"[MASK] lost after truncation "
                f"(article_id={row.get('article_id', '?')}), skipping."
            )
            return []

        try:
            results = pipe(text, top_k=TOP_K)

            # Return flat list — no extra wrapping bracket
            return [
                (
                    r["token_str"].strip(),
                    round(r["score"], 4)
                )
                for r in results
            ]

        except Exception as e:

            log.warning(
                f"Single-mask inference failed "
                f"(article_id={row.get('article_id', '?')}): {e}"
            )

            return []

    # ----------------------------------------------------------------
    # MULTIPLE MASKS
    # ----------------------------------------------------------------

    masked_words = masked_sentence.split()
    original_words = original_sentence.split()

    # Positions of masks in tokenized sentence
    mask_positions = [
        i
        for i, word in enumerate(masked_words)
        if "[MASK]" in word
    ]

    # Original words corresponding to each mask
    original_mask_words = []

    for pos in mask_positions:

        if pos < len(original_words):
            original_mask_words.append(original_words[pos])
        else:
            original_mask_words.append("[UNK]")

    all_predictions = []

    # ---------------------------------------------------------------
    # Run one BERT pass per mask
    # ---------------------------------------------------------------

    for target_mask_idx in range(n_masks):

        reconstructed_words = []
        current_mask_idx = 0

        for word in masked_words:

            if "[MASK]" in word:

                if current_mask_idx == target_mask_idx:

                    # Keep this target mask
                    reconstructed_words.append(word)

                else:

                    # Restore original token
                    restored = word.replace(
                        "[MASK]",
                        original_mask_words[current_mask_idx]
                    )

                    reconstructed_words.append(restored)

                current_mask_idx += 1

            else:
                reconstructed_words.append(word)

        reconstructed_sentence = " ".join(reconstructed_words)

        # Validate exactly one [MASK] remains
        remaining_masks = reconstructed_sentence.count("[MASK]")

        if remaining_masks != 1:

            log.warning(
                f"Skipping malformed reconstruction "
                f"(remaining_masks={remaining_masks}, "
                f"article_id={row.get('article_id', '?')})"
            )

            all_predictions.append([])
            continue

        full_text = (
            f"{prev_sentence} "
            f"{reconstructed_sentence} "
            f"{next_sentence}"
        ).strip()

        # Hard tokenizer truncation
        full_text = truncate_to_bert_limit(full_text, tokenizer)

        if "[MASK]" not in full_text:
            log.warning(
                f"[MASK] lost after truncation "
                f"(article_id={row.get('article_id', '?')}, "
                f"mask_idx={target_mask_idx}), skipping."
            )
            all_predictions.append([])
            continue

        try:

            results = pipe(full_text, top_k=TOP_K)

            preds = [
                (
                    r["token_str"].strip(),
                    round(r["score"], 4)
                )
                for r in results
            ]

            all_predictions.append(preds)

        except Exception as e:

            log.warning(
                f"Multi-mask inference failed "
                f"(article_id={row.get('article_id', '?')}, "
                f"mask_idx={target_mask_idx}): {e}"
            )

            all_predictions.append([])

    return all_predictions


def process_file(
    filepath: Path,
    pipes: dict,
    tokenizers: dict,
    output_dir: Path
):
    """
    Process a single .jsonl file, writing only rows whose masked word
    matches one of the FILTER_WORDS.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / filepath.name

    if is_file_complete(filepath, out_path):
        log.info(f"  Skipping (already complete): {out_path}")
        return

    elif out_path.exists():
        log.warning(
            f"  Incomplete output found, reprocessing: {out_path}"
        )

    log.info(f"  Processing: {filepath}")

    rows_written = 0
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

                log.warning(
                    f"  Line {line_num}: JSON parse error — {e}"
                )

                continue

            # Skip rows whose masked word is not in FILTER_WORDS
            if not row_matches_filter(row):
                rows_skipped += 1
                continue

            for col_name, pipe in pipes.items():

                row[col_name] = predict(
                    pipe,
                    row,
                    tokenizers[col_name]
                )

            fout.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )

            rows_written += 1

            if rows_written % 500 == 0:
                log.info(f"    {rows_written} rows written...")

    log.info(
        f"  Written: {out_path} "
        f"({rows_written} written, {rows_skipped} skipped)"
    )


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():

    log.info(
        f"Device: {'GPU' if DEVICE == 0 else 'CPU'}"
    )
    log.info(f"Filter words: {sorted(FILTER_WORDS)}")

    check_gpu_health()

    # ---------------------------------------------------------------
    # Load models
    # ---------------------------------------------------------------

    log.info("Loading models...")

    pipes = {}
    tokenizers = {}

    for col_name, model_path in MODELS.items():

        log.info(f"  Loading {col_name} from {model_path}")

        pipes[col_name] = pipeline(
            "fill-mask",
            model=model_path,
            tokenizer=model_path,
            device=DEVICE,
            top_k=TOP_K,
        )

        tokenizers[col_name] = pipes[col_name].tokenizer

        log.info(f"  Loaded {col_name}")

    # ---------------------------------------------------------------
    # Process directories
    # ---------------------------------------------------------------

    for input_dir in INPUT_DIRS:

        input_path = Path(input_dir)

        if not input_path.exists():

            log.warning(
                f"Input dir not found, skipping: {input_dir}"
            )

            continue

        jsonl_files = sorted(
            input_path.glob("*.jsonl")
        )

        if not jsonl_files:

            log.warning(
                f"No .jsonl files found in: {input_dir}"
            )

            continue

        relative = input_path.relative_to(
            "/gpfs/projects/bsc100/textmachine-data/filtered_data"
        )

        output_dir = Path(OUTPUT_BASE) / relative

        log.info(
            f"\nDirectory: {input_dir}  "
            f"({len(jsonl_files)} files)"
        )

        for filepath in jsonl_files:

            process_file(
                filepath,
                pipes,
                tokenizers,
                output_dir
            )

    log.info("\nAll done.")


if __name__ == "__main__":
    main()
