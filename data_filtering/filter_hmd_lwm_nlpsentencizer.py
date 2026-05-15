"""
Filter HMD and LwM newspaper datasets by a list of regex query words.

Usage (single word — suitable for Slurm job arrays):
    python filter_hmd_lwm.py --dataset lwm  --query-word slave   --output-dir results/
    python filter_hmd_lwm.py --dataset hmd  --query-word "factory act" --output-dir results/

Usage (multiple words at once — one output file per word):
    python filter_hmd_lwm.py --dataset lwm  --query-words slave slaves machine --output-dir results/

The output filename is derived automatically:
    <output-dir>/<dataset>_<query_word_slugified>.jsonl
e.g.  results/lwm_slave.jsonl
      results/hmd_factory_act.jsonl
"""

import re
import json
import argparse
import logging
from pathlib import Path
from typing import Iterator

import spacy

# ---------------------------------------------------------------------------
# spaCy sentencizer (loaded once at module level)
# ---------------------------------------------------------------------------

def _build_nlp():
    from spacy.lang.en import English
    nlp = English()
    nlp.add_pipe("sentencizer")
    # Disable everything we don't need to save RAM and CPU
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
    """
    lines = text.split("\n")
    parts: list[str] = []
    carry = ""

    for line in lines:
        line = line.strip()
        if not line:
            if carry:
                parts.append(carry)
                carry = ""
            continue                   # drop blank lines — no empty sentinels needed

        if carry:
            if carry.endswith("-"):
                carry = carry[:-1] + line
            else:
                carry = carry + " " + line
        else:
            carry = line

    if carry:
        parts.append(carry)

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Regex compilation
# ---------------------------------------------------------------------------

def build_pattern(query_words: list[str]) -> re.Pattern:
    """
    Build a single compiled regex that matches any of the query words/phrases,
    case-insensitively, as whole words.
    """
    escaped = [re.escape(w) for w in query_words]
    alternation = "|".join(f"(?<![\\w])(?:{e})(?![\\w])" for e in escaped)
    return re.compile(alternation, re.IGNORECASE)


def slugify(word: str) -> str:
    """Turn a query word/phrase into a safe filename component."""
    return re.sub(r"[^\w]+", "_", word.strip().lower()).strip("_")


# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------

def load_metadata(meta_path: Path) -> dict[str, dict]:
    KEEP = {"article_id", "ocr_quality_mean", "ocr_quality_sd",
            "newspaper_title", "location",
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
# Core filtering  ← KEY CHANGE: generator instead of list accumulation
# ---------------------------------------------------------------------------

def iter_matches(
    content_path: Path,
    meta_path: Path,
    pattern: re.Pattern,
) -> Iterator[dict]:
    """
    Yield one dict per matching sentence.  Memory use is O(1 article) rather
    than O(all matches in file).
    """
    metadata = load_metadata(meta_path)

    with content_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            raw_text = record.get("text", "")
            if not raw_text:
                continue

            processed_text = rejoin_text(raw_text)

            if not pattern.search(processed_text):
                continue

            sentences = sentencize(processed_text)
            article_id = record.get("article_id")
            meta = metadata.get(article_id, {})

            for idx, sent in enumerate(sentences):
                if not pattern.search(sent):
                    continue

                masked_sent = pattern.sub("[MASK]", sent)

                yield {
                    "article_id":       article_id,
                    "prev_sentence":    sentences[idx - 1] if idx > 0 else None,
                    "sentence":         sent,
                    "masked_sentence":  masked_sent,
                    "next_sentence":    sentences[idx + 1] if idx < len(sentences) - 1 else None,
                    "ocr_quality_mean": meta.get("ocr_quality_mean"),
                    "ocr_quality_sd":   meta.get("ocr_quality_sd"),
                    "newspaper_title":  meta.get("newspaper_title"),
                    "location":         meta.get("location"),
                    "year":             meta.get("year"),
                    "month":            meta.get("month"),
                    "day":              meta.get("day"),
                }


def filter_dataset(
    dataset: str,
    query_word: str,
    output_path: Path,
) -> None:
    root = DATASET_ROOTS[dataset]
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    pattern = build_pattern([query_word])
    logging.info("Query pattern: %s", pattern.pattern)

    content_files = sorted(root.glob("*_content.jsonl"))
    if not content_files:
        logging.warning("No *_content.jsonl files found under %s", root)
        return

    total_written = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out_fh:
        for content_path in content_files:
            meta_path = content_path.with_name(
                content_path.name.replace("_content.jsonl", "_metadata.jsonl")
            )
            if not meta_path.exists():
                logging.warning("Missing metadata file for %s — skipping", content_path.name)
                continue

            logging.info("Processing %s …", content_path.name)
            file_count = 0

            try:
                for record in iter_matches(content_path, meta_path, pattern):
                    out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    file_count += 1
            except Exception as exc:
                logging.error("Error processing %s: %s", content_path.name, exc)
                continue

            logging.info("  → %d match(es)", file_count)
            total_written += file_count

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

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--query-word",
        metavar="WORD",
        help="A single query word / phrase (used by Slurm array jobs).",
    )
    group.add_argument(
        "--query-words",
        nargs="+",
        metavar="WORD",
        help="One or more query words; produces one output file per word.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("filtered_output"),
        help="Directory where per-word output files are written.",
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

    words = [args.query_word] if args.query_word else args.query_words

    for word in words:
        slug = slugify(word)
        output_path = args.output_dir / f"{args.dataset}_{slug}.jsonl"

        logging.info("=" * 60)
        logging.info("Dataset    : %s", args.dataset)
        logging.info("Query word : %r", word)
        logging.info("Output     : %s", output_path)
        logging.info("=" * 60)

        filter_dataset(
            dataset=args.dataset,
            query_word=word,
            output_path=output_path,
        )


if __name__ == "__main__":
    main()