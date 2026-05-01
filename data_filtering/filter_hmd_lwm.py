"""
Filter HMD and LwM newspaper datasets by a list of regex query words.

Usage:
    python filter_hmd_lwm.py \
        --dataset lwm \
        --query-words slave slaves \
        --output results.jsonl

    python filter_hmd_lwm.py \
        --dataset hmd \
        --query-words "factory act" "child labour" \
        --output results.jsonl
"""

import re
import json
import argparse
import logging
from pathlib import Path

import spacy

# ---------------------------------------------------------------------------
# spaCy sentencizer (loaded once at module level)
# ---------------------------------------------------------------------------

def _build_nlp():
    from spacy.lang.en import English
    nlp = English()
    nlp.add_pipe("sentencizer")
    return nlp

nlp = _build_nlp()


def sentencize(text: str) -> list[str]:
    """Split *text* into sentences using spaCy's rule-based sentencizer."""
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATASET_ROOTS = {
    "lwm": Path("/gpfs/projects/bsc100/textmachine-data/preprocessed_data/output_lwm"),
    "hmd": Path("/gpfs/projects/bsc100/textmachine-data/preprocessed_data/output_hmd"),
}

# ---------------------------------------------------------------------------
# Text processing
# ---------------------------------------------------------------------------

def rejoin_text(text: str) -> str:
    """
    Convert physical-line breaks into running prose.

    Rules
    -----
    1. If a line ends with a hyphen, strip the hyphen and join directly to
       the next line (de-hyphenation).
    2. Otherwise, join the two lines with a single space.

    Sentence boundaries are preserved because sentences end with punctuation
       followed (in the original) by the start of a new line that begins with
       a capital letter — those spaces are inserted by rule 2.
    """
    lines = text.split("\n")
    parts: list[str] = []
    carry = ""

    for line in lines:
        line = line.strip()
        if not line:
            # Blank line → treat as paragraph separator and flush
            if carry:
                parts.append(carry)
                carry = ""
            parts.append("")          # preserve paragraph gap if needed
            continue

        if carry:
            if carry.endswith("-"):
                # De-hyphenate: drop the hyphen and glue directly
                carry = carry[:-1] + line
            else:
                carry = carry + " " + line
        else:
            carry = line

    if carry:
        parts.append(carry)

    # Remove leading/trailing empty strings and collapse to a single string
    rejoined = " ".join(p for p in parts if p)
    return rejoined


# ---------------------------------------------------------------------------
# Regex compilation
# ---------------------------------------------------------------------------

def build_pattern(query_words: list[str]) -> re.Pattern:
    """
    Build a single compiled regex that matches any of the query words/phrases,
    case-insensitively, as whole words.

    Each query word is treated as a literal string (not a regex), so special
    characters are escaped.  If you want raw regex support, remove the
    re.escape() call.
    """
    escaped = [re.escape(w) for w in query_words]
    # \\b word boundaries work for ASCII; for Unicode text use (?<!\w)/(?!\w)
    alternation = "|".join(f"(?<![\\w])(?:{e})(?![\\w])" for e in escaped)
    return re.compile(alternation, re.IGNORECASE)


# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------

def load_metadata(meta_path: Path) -> dict[str, dict]:
    """
    Load a *_metadata.jsonl file into a dict keyed by article_id.

    We keep only the fields we need for the output schema.
    """
    KEEP = {"article_id", "ocr_quality_mean", "ocr_quality_sd",
            "word_count", "newspaper_title", "location",
            "year", "month", "day"}
    meta: dict[str, dict] = {}
    with meta_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            article_id = record.get("article_id")
            if article_id is None:
                continue
            meta[article_id] = {k: record.get(k) for k in KEEP}
    return meta


# ---------------------------------------------------------------------------
# Core filtering
# ---------------------------------------------------------------------------

def iter_matches(
    content_path: Path,
    meta_path: Path,
    pattern: re.Pattern,
) -> list[dict]:
    """
    Yield output records for every article in *content_path* whose (rejoined)
    text matches *pattern*.
    """
    metadata = load_metadata(meta_path)
    results = []

    with content_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            raw_text = record.get("text", "")
            processed_text = rejoin_text(raw_text)

            if not pattern.search(processed_text):
                continue

            sentences = sentencize(processed_text)

            article_id = record.get("article_id")
            meta = metadata.get(article_id, {})

            out = {
                "article_id":       article_id,
                "text":             sentences,
                "ocr_quality_mean": meta.get("ocr_quality_mean"),
                "ocr_quality_sd":   meta.get("ocr_quality_sd"),
                "word_count":       meta.get("word_count"),
                "newspaper_title":  meta.get("newspaper_title"),
                "location":         meta.get("location"),
                "year":             meta.get("year"),
                "month":            meta.get("month"),
                "day":              meta.get("day"),
            }
            results.append(out)

    return results


def filter_dataset(
    dataset: str,
    query_words: list[str],
    output_path: Path,
) -> None:
    root = DATASET_ROOTS[dataset]
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    pattern = build_pattern(query_words)
    logging.info("Query pattern: %s", pattern.pattern)

    # Discover all content files
    content_files = sorted(root.glob("*_content.jsonl"))
    if not content_files:
        logging.warning("No *_content.jsonl files found under %s", root)
        return

    total_written = 0

    with output_path.open("w", encoding="utf-8") as out_fh:
        for content_path in content_files:
            # Derive the companion metadata path
            meta_path = content_path.with_name(
                content_path.name.replace("_content.jsonl", "_metadata.jsonl")
            )
            if not meta_path.exists():
                logging.warning("Missing metadata file for %s — skipping", content_path.name)
                continue

            logging.info("Processing %s …", content_path.name)

            try:
                matches = iter_matches(content_path, meta_path, pattern)
            except Exception as exc:
                logging.error("Error processing %s: %s", content_path.name, exc)
                continue

            for record in matches:
                out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")

            logging.info("  → %d match(es)", len(matches))
            total_written += len(matches)

    logging.info("Done. Total records written: %d → %s", total_written, output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter HMD/LwM newspaper data by query words.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_ROOTS),
        required=True,
        help="Which dataset to process.",
    )
    parser.add_argument(
        "--query-words",
        nargs="+",
        required=True,
        metavar="WORD",
        help="One or more query words / phrases to search for.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("filtered_output.jsonl"),
        help="Path to the output JSONlines file.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    logging.info("Dataset  : %s", args.dataset)
    logging.info("Query    : %s", args.query_words)
    logging.info("Output   : %s", args.output)

    filter_dataset(
        dataset=args.dataset,
        query_words=args.query_words,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()