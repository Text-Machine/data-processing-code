"""
Filter BL Microsoft dataset by query words, with sentence-level output
including context sentences and masked versions.
"""

import re
import gc
import csv
import gzip
import json
import tarfile
import argparse
import logging
from pathlib import Path
from typing import Iterator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_ROOT = Path("/gpfs/projects/bsc100/textmachine-data/downloaded_data_vm")

METADATA_PATH = Path(
    "/gpfs/projects/bsc100/textmachine-data/preprocessed_data/output_blmicrosoft/metadata.csv"
)

DECADE_TO_TAR = {
    "1510_1699": "OCR_text_c_1510_-_1699.tar.gz",
    "1700_1799": "OCR_text_1700_-_1799.tar.gz",
    "1800_1809": "OCR_text_1800_-_1809.tar.gz",
    "1810_1819": "OCR_text_1810_-_1819.tar.gz",
    "1820_1829": "OCR_text_1820_-_1829.tar.gz",
    "1830_1839": "OCR_text_1830_-_1839.tar.gz",
    "1840_1849": "OCR_text_1840_-_1849.tar.gz",
    "1850_1859": "OCR_text_1850_-_1859.tar.gz",
    "1860_1869": "OCR_text_1860_-_1869.tar.gz",
    "1870_1879": "OCR_text_1870_-_1879.tar.gz",
    "1880_1889": "OCR_text_1880_-_1889.tar.gz",
    "1890_1899": "OCR_text_1890_-_1899.tar.gz",
}

METADATA_KEEP_COLS = [
    "record_id", "date", "title", "author",
    "main_publication_country", "main_language",
    "fiction_score", "nonfiction_score",
]

DATE_MIN = 1800
DATE_MAX = 1900

# ---------------------------------------------------------------------------
# Sentence splitting / text normalisation
# ---------------------------------------------------------------------------

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

def sentencize(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]

def rejoin_text(text: str) -> str:
    lines = text.split("\n")
    parts: list[str] = []
    carry = ""
    for line in lines:
        line = line.strip()
        if not line:
            if carry:
                parts.append(carry)
                carry = ""
            continue
        if carry:
            carry = (carry[:-1] + line) if carry.endswith("-") else (carry + " " + line)
        else:
            carry = line
    if carry:
        parts.append(carry)
    return " ".join(parts)

# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

def build_pattern(query_words: list[str]) -> re.Pattern:
    escaped = [re.escape(w) for w in query_words]
    alternation = "|".join(f"(?<![\\w])(?:{e})(?![\\w])" for e in escaped)
    return re.compile(alternation, re.IGNORECASE)

def mask_sentence(sentence: str, pattern: re.Pattern) -> str:
    """Replace every match of the pattern with [MASK]."""
    return pattern.sub("[MASK]", sentence)

def find_query(sentence: str, query_words: list[str], pattern: re.Pattern) -> str:
    """Return the first query word that matches in this sentence."""
    m = pattern.search(sentence)
    if m:
        # Return the canonical query word (lowercased match text)
        return m.group(0).lower()
    return ""

def slugify(word: str) -> str:
    return re.sub(r"[^\w]+", "_", word.strip().lower()).strip("_")

# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------

def load_metadata(
    metadata_path: Path,
    date_min: int = DATE_MIN,
    date_max: int = DATE_MAX,
) -> dict[str, dict]:
    kept: dict[str, dict] = {}

    with metadata_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("main_language", "").strip() != "English":
                continue

            raw_date = row.get("date", "").strip()
            try:
                year = int(raw_date)
            except ValueError:
                m = re.search(r"\b(\d{4})\b", raw_date)
                if not m:
                    continue
                year = int(m.group(1))

            if not (date_min <= year <= date_max):
                continue

            path_to_json = row.get("path_to_json", "").strip()
            if not path_to_json:
                continue

            record_id = row.get("record_id", "").strip()
            if not record_id:
                continue

            kept[record_id] = {
                **{col: row.get(col) for col in METADATA_KEEP_COLS},
                "path_to_json": path_to_json,
            }

    logging.info("Metadata records passing filters: %d", len(kept))
    return kept

# ---------------------------------------------------------------------------
# Tar lookup
# ---------------------------------------------------------------------------

def get_tar_path(path_to_json: str) -> Path | None:
    decade_folder = path_to_json.split("/")[0]
    tar_name = DECADE_TO_TAR.get(decade_folder)
    if tar_name is None:
        logging.warning("No tar mapping for folder %r", decade_folder)
        return None
    return DATA_ROOT / tar_name

# ---------------------------------------------------------------------------
# Core matching — sentence level with context and masking
# ---------------------------------------------------------------------------

def iter_page_matches(
    path_to_json: str,
    meta_record: dict,
    query_words: list[str],
    pattern: re.Pattern,
    tar_cache: dict[Path, tarfile.TarFile],
) -> Iterator[dict]:
    """
    For each page in the book, split into sentences and yield one record
    per sentence that matches the pattern, including:
      - query: the matched query word
      - prev_sentence / next_sentence: surrounding sentences (empty string if none)
      - masked_sentence: sentence with match replaced by [MASK]
    """
    tar_path = get_tar_path(path_to_json)
    if tar_path is None:
        return

    if tar_path not in tar_cache:
        logging.info("Opening tar: %s", tar_path.name)
        tar_cache[tar_path] = tarfile.open(tar_path, "r:gz")
    tf = tar_cache[tar_path]

    try:
        member = tf.getmember(path_to_json)
    except KeyError:
        logging.debug("Member %r not found in %s", path_to_json, tar_path.name)
        return

    with tf.extractfile(member) as raw_fh:
        with gzip.open(raw_fh, "rt", encoding="utf-8") as fh:
            for raw_line in fh:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    page = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    logging.debug("JSON decode error in %s: %s", path_to_json, exc)
                    continue

                raw_text = page.get("text", "")
                if not raw_text:
                    continue

                processed_text = rejoin_text(raw_text)

                # Fast page-level rejection before sentence splitting
                if not pattern.search(processed_text):
                    continue

                sentences = sentencize(processed_text)
                pg           = page.get("pg")
                mean_wc_ocr  = page.get("mean_wc_ocr")
                std_wc_ocr   = page.get("std_wc_ocr")

                for i, sent in enumerate(sentences):
                    if not pattern.search(sent):
                        continue

                    prev_sent = sentences[i - 1] if i > 0 else ""
                    next_sent = sentences[i + 1] if i < len(sentences) - 1 else ""

                    yield {
                        # Book-level metadata
                        **meta_record,
                        # Query info
                        "query":           find_query(sent, query_words, pattern),
                        # Sentence-level fields
                        "prev_sentence":   prev_sent,
                        "sentence":        sent,
                        "masked_sentence": mask_sentence(sent, pattern),
                        "next_sentence":   next_sent,
                        # Page-level fields
                        "pg":              pg,
                        "ocr_quality_mean": mean_wc_ocr,
                        "ocr_quality_sd":   std_wc_ocr,
                    }

                del sentences, processed_text

# ---------------------------------------------------------------------------
# Top-level filter function
# ---------------------------------------------------------------------------

def filter_dataset(
    query_words: list[str],
    output_path: Path,
    metadata_path: Path = METADATA_PATH,
    data_root: Path = DATA_ROOT,
) -> None:
    pattern = build_pattern(query_words)
    logging.info("Query pattern: %s", pattern.pattern)

    metadata = load_metadata(metadata_path)
    if not metadata:
        logging.warning("No metadata records passed the filters — nothing to do.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_written = 0
    tar_cache: dict[Path, tarfile.TarFile] = {}

    try:
        with output_path.open("w", encoding="utf-8") as out_fh:
            for record_id, meta_record in metadata.items():
                path_to_json = meta_record.pop("path_to_json")
                book_count = 0

                try:
                    for record in iter_page_matches(
                        path_to_json, meta_record, query_words, pattern, tar_cache
                    ):
                        out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                        book_count += 1
                except Exception as exc:
                    logging.error("Error processing %s: %s", path_to_json, exc)
                    continue

                if book_count:
                    logging.info("  %s → %d match(es)", path_to_json, book_count)

                total_written += book_count
                gc.collect()

    finally:
        for tf in tar_cache.values():
            tf.close()

    logging.info("Done. Total records written: %d → %s", total_written, output_path)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter BL Microsoft book data by query words.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--query-word",  metavar="WORD")
    group.add_argument("--query-words", nargs="+", metavar="WORD")

    parser.add_argument("--output-dir", type=Path, default=Path("filtered_output"))
    parser.add_argument("--metadata",   type=Path, default=METADATA_PATH)
    parser.add_argument("--data-root",  type=Path, default=DATA_ROOT)
    parser.add_argument("--log-level",  default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    words: list[str] = [args.query_word] if args.query_word else args.query_words

    if args.query_word:
        slug = slugify(args.query_word)
        output_path = args.output_dir / f"bl_microsoft_{slug}.jsonl"
    else:
        combined_slug = "_".join(slugify(w) for w in words)
        if len(combined_slug) > 80:
            combined_slug = combined_slug[:80] + "_etc"
        output_path = args.output_dir / f"bl_microsoft_{combined_slug}.jsonl"

    logging.info("=" * 60)
    logging.info("Query words: %r", words)
    logging.info("Output     : %s", output_path)
    logging.info("=" * 60)

    filter_dataset(
        query_words=words,
        output_path=output_path,
        metadata_path=args.metadata,
        data_root=args.data_root,
    )


if __name__ == "__main__":
    main()
