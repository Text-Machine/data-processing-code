"""
Filter EEBO / ECCO / Evans datasets by query words, with sentence-level
output including context sentences and masked versions.

These three datasets share a common shape, unlike BL Microsoft (tar.gz +
per-page JSON) or HMD/LWM (paired metadata/content JSONL):

    {data_root}/{name}_metadata.csv   -- one row per record (book/document)
    {data_root}/txt/...               -- one plain-text file per record,
                                          located via the CSV's path_txt
                                          column (already relative to
                                          data_root, may include a
                                          subdirectory e.g. txt/phase1/...)

EEBO specifically has TWO metadata CSVs (phase1_metadata.csv and
phase2_metadata.csv) rather than one -- this script discovers metadata
CSVs by globbing *.csv in data_root rather than assuming a single
filename, so it works unchanged for Evans/ECCO (one CSV) and EEBO (two).

Column naming differs slightly: EEBO uses `date`, Evans/ECCO use
`edition_date`. Both are normalized to a `date` field in the output.

Unlike BL Microsoft, these are TCP-transcribed texts, not OCR'd scans --
no hyphen-aware line-rejoining is needed. Text is normalized by simply
collapsing all whitespace (including newlines) to single spaces before
sentence splitting.
"""

import re
import gc
import csv
import json
import argparse
import logging
from pathlib import Path
from typing import Iterator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Source metadata columns kept verbatim (output field name == source column
# name), plus the date column which is normalized separately below.
METADATA_KEEP_COLS = [
    "record_id", "book_id", "title", "author",
    "birth_year", "death_year", "place", "publisher",
    "main_language", "number_of_pages", "num_words", "all_ids",
]

# Column that holds the date, which varies by dataset -- checked in order.
DATE_COL_CANDIDATES = ["edition_date", "date"]

DEFAULT_LANGUAGE_FILTER = "English"  # pass --language-filter ALL to disable

DEFAULT_QUERY_WORDS = ["machine", "machines"]

# ---------------------------------------------------------------------------
# spaCy sentencizer
# ---------------------------------------------------------------------------

def _build_nlp():
    from spacy.lang.en import English
    nlp = English()
    nlp.add_pipe("sentencizer")
    # spaCy's default nlp.max_length (1,000,000 chars) exists to guard the
    # parser/NER components, which need ~1GB of temporary memory per
    # 100,000 characters. We only ever add the rule-based sentencizer here
    # -- no parser, no NER -- so that memory concern doesn't apply, and
    # it's safe to raise this well above any expected document size.
    # Evans alone has single records over 5.4M characters (multi-volume
    # bound works); EEBO/ECCO may have similarly large documents.
    nlp.max_length = 20_000_000
    return nlp

nlp = _build_nlp()


def sentencize(text: str) -> list[str]:
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def normalize_whitespace(text: str) -> str:
    """
    Collapse all whitespace (spaces, tabs, newlines, blank lines) to single
    spaces. TCP-transcribed text doesn't have OCR-style hyphenated line
    wraps, so unlike BL Microsoft's rejoin_text() there's no hyphen
    handling needed here -- this is a much simpler normalization.
    """
    return " ".join(text.split())

# ---------------------------------------------------------------------------
# Regex helpers (same pattern as the other filter scripts)
# ---------------------------------------------------------------------------

def build_pattern(query_words: list[str]) -> re.Pattern:
    escaped = [re.escape(w) for w in query_words]
    alternation = "|".join(f"(?<![\\w])(?:{e})(?![\\w])" for e in escaped)
    return re.compile(alternation, re.IGNORECASE)

def mask_sentence(sentence: str, pattern: re.Pattern) -> str:
    return pattern.sub("[MASK]", sentence)

def find_query(sentence: str, query_words: list[str], pattern: re.Pattern) -> str:
    m = pattern.search(sentence)
    return m.group(0).lower() if m else ""

def slugify(word: str) -> str:
    return re.sub(r"[^\w]+", "_", word.strip().lower()).strip("_")

# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------

def discover_metadata_csvs(data_root: Path) -> list[Path]:
    """
    Glob for *.csv directly under data_root (not recursive into txt/).
    Handles both the single-CSV case (Evans, ECCO) and EEBO's two-CSV
    (phase1 + phase2) case without needing dataset-specific filenames.
    """
    csvs = sorted(p for p in data_root.glob("*.csv") if p.is_file())
    logging.info("Discovered %d metadata CSV(s) under %s: %s",
                 len(csvs), data_root, [p.name for p in csvs])
    return csvs


def resolve_date_col(fieldnames: list[str]) -> str | None:
    for candidate in DATE_COL_CANDIDATES:
        if candidate in fieldnames:
            return candidate
    return None


def load_metadata(csv_path: Path, language_filter: str | None) -> dict[str, dict]:
    """
    Load one metadata CSV, keyed by record_id. Normalizes whichever date
    column is present (edition_date or date) into a `date` output field.
    """
    kept: dict[str, dict] = {}

    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        date_col = resolve_date_col(reader.fieldnames or [])
        if date_col is None:
            logging.warning("No edition_date/date column found in %s, dates will be empty", csv_path.name)

        for row in reader:
            if language_filter and row.get("main_language", "").strip() != language_filter:
                continue

            record_id = row.get("record_id", "").strip()
            path_txt = row.get("path_txt", "").strip()
            if not record_id or not path_txt:
                continue

            record = {col: row.get(col) for col in METADATA_KEEP_COLS}
            record["date"] = row.get(date_col) if date_col else None
            record["source_csv"] = csv_path.name
            record["path_txt"] = path_txt

            kept[record_id] = record

    logging.info("%s: %d metadata records passing filters", csv_path.name, len(kept))
    return kept

# ---------------------------------------------------------------------------
# Text matching
# ---------------------------------------------------------------------------

def iter_book_matches(
    text_path: Path,
    meta_record: dict,
    query_words: list[str],
    pattern: re.Pattern,
) -> Iterator[dict]:
    """
    Read one record's plain-text file, sentence-split it, and yield one
    record per matching sentence with prev/next context and a masked
    version -- same output shape as the other filter scripts.
    """
    try:
        raw_text = text_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logging.warning("Could not read %s: %s", text_path, exc)
        return

    if not raw_text:
        return

    processed_text = normalize_whitespace(raw_text)

    # Fast document-level rejection before sentence splitting
    if not pattern.search(processed_text):
        return

    sentences = sentencize(processed_text)

    for i, sent in enumerate(sentences):
        if not pattern.search(sent):
            continue

        prev_sent = sentences[i - 1] if i > 0 else ""
        next_sent = sentences[i + 1] if i < len(sentences) - 1 else ""

        yield {
            **{k: v for k, v in meta_record.items() if k not in ("path_txt",)},
            "query":           find_query(sent, query_words, pattern),
            "prev_sentence":   prev_sent,
            "sentence":        sent,
            "masked_sentence": mask_sentence(sent, pattern),
            "next_sentence":   next_sent,
        }

    del sentences, processed_text

# ---------------------------------------------------------------------------
# Top-level filter function
# ---------------------------------------------------------------------------

def filter_dataset(
    query_words: list[str],
    output_path: Path,
    data_root: Path,
    language_filter: str | None,
) -> None:
    pattern = build_pattern(query_words)
    logging.info("Query pattern: %s", pattern.pattern)

    metadata_csvs = discover_metadata_csvs(data_root)
    if not metadata_csvs:
        logging.warning("No metadata CSVs found under %s -- nothing to do.", data_root)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_written = 0
    total_records = 0

    with output_path.open("w", encoding="utf-8") as out_fh:
        for csv_path in metadata_csvs:
            metadata = load_metadata(csv_path, language_filter)
            if not metadata:
                continue

            for record_id, meta_record in metadata.items():
                total_records += 1
                text_path = data_root / meta_record["path_txt"]

                if not text_path.exists():
                    logging.debug("Text file not found, skipping: %s", text_path)
                    continue

                try:
                    for record in iter_book_matches(text_path, meta_record, query_words, pattern):
                        out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                        total_written += 1
                except Exception as exc:
                    logging.error("Error processing %s: %s", text_path, exc)
                    continue

            del metadata
            gc.collect()

    logging.info(
        "Done. Processed %d record(s) across %d CSV(s). Total matching sentences: %d -> %s",
        total_records, len(metadata_csvs), total_written, output_path,
    )

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter EEBO/ECCO/Evans datasets by query words.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--query-word", metavar="WORD")
    group.add_argument("--query-words", nargs="+", metavar="WORD",
                        help=f"Default: {DEFAULT_QUERY_WORDS}")

    parser.add_argument("--dataset-name", required=True, choices=["evans", "eebo", "ecco"],
                        help="Used as the output filename prefix.")
    parser.add_argument("--step-tag", default="step2",
                        help="Appended to the output filename, e.g. 'step2' -> eebo_machine_step2.jsonl. "
                             "Pass an empty string to omit it entirely.")
    parser.add_argument("--output-dir", type=Path, default=Path("filtered_output"),
                        help="Output directory (ignored if --output-file is set)")
    parser.add_argument("--output-file", type=Path, default=None,
                        help="Explicit output file path (overrides --output-dir + auto-naming)")
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Dataset root directory (e.g. .../preprocessed_data/output_evans)")
    parser.add_argument("--language-filter", default=DEFAULT_LANGUAGE_FILTER,
                        help="Value main_language must equal to be kept. Pass 'ALL' to disable filtering.")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    words: list[str] = [args.query_word] if args.query_word else (args.query_words or DEFAULT_QUERY_WORDS)
    language_filter = None if args.language_filter.upper() == "ALL" else args.language_filter

    if args.output_file:
        output_path = args.output_file
    else:
        tag_suffix = f"_{args.step_tag}" if args.step_tag else ""
        if args.query_word:
            slug = slugify(args.query_word)
            output_path = args.output_dir / f"{args.dataset_name}_{slug}{tag_suffix}.jsonl"
        else:
            combined_slug = "_".join(slugify(w) for w in words)
            if len(combined_slug) > 80:
                combined_slug = combined_slug[:80] + "_etc"
            output_path = args.output_dir / f"{args.dataset_name}_{combined_slug}{tag_suffix}.jsonl"

    logging.info("=" * 60)
    logging.info("Dataset      : %s", args.dataset_name)
    logging.info("Query words  : %r", words)
    logging.info("Data root    : %s", args.data_root)
    logging.info("Lang filter  : %s", language_filter or "ALL")
    logging.info("Output       : %s", output_path)
    logging.info("=" * 60)

    filter_dataset(
        query_words=words,
        output_path=output_path,
        data_root=args.data_root,
        language_filter=language_filter,
    )


if __name__ == "__main__":
    main()