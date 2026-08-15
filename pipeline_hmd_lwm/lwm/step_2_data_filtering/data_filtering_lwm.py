"""
Filter LWM dataset by a query word, with sentence-level output
including context sentences and masked versions.

The query word is read from query_word_config.sh.

LWM is already preprocessed into pairs of JSONL files:

    <NLP>_metadata.jsonl
    <NLP>_content.jsonl

Metadata and article content are joined using article_id.
"""

import gc
import json
import argparse
import logging
import re
import subprocess
from pathlib import Path
from typing import Iterator


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_ROOT = Path(
    "/gpfs/projects/bsc100/textmachine-data/preprocessed_data/output_lwm"
)

QUERY_CONFIG = Path("query_word_config.sh")

DATE_MIN = 1800
DATE_MAX = 1900


# ---------------------------------------------------------------------------
# Query-word configuration
# ---------------------------------------------------------------------------

def load_query_word(config_path: Path) -> str:
    """
    Read QUERY_WORD from query_word_config.sh.

    Expected format:

        QUERY_WORD="railway"

    or:

        QUERY_WORD='railway'

    The shell file is sourced in a subprocess so that normal shell quoting
    is handled correctly.
    """

    if not config_path.exists():
        raise FileNotFoundError(
            f"Query configuration file not found: {config_path}"
        )

    command = [
        "bash",
        "-c",
        f'source "{config_path}" && printf "%s" "$QUERY_WORD"',
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )

    query_word = result.stdout.strip()

    if not query_word:
        raise ValueError(
            f"QUERY_WORD is empty in {config_path}"
        )

    return query_word


# ---------------------------------------------------------------------------
# spaCy sentencizer
# ---------------------------------------------------------------------------

def _build_nlp():
    from spacy.lang.en import English

    nlp = English()
    nlp.add_pipe("sentencizer")
    return nlp


nlp = _build_nlp()


def sentencize(text: str) -> list[str]:
    """Split text into sentences using spaCy's rule-based sentencizer."""

    doc = nlp(text)

    return [
        sent.text.strip()
        for sent in doc.sents
        if sent.text.strip()
    ]


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def rejoin_text(text: str) -> str:
    """
    Rejoin OCR lines into continuous text.

    A line ending in '-' is treated as a hyphenated line break and the
    hyphen is removed.

    Ordinary line breaks are replaced with spaces.
    """

    lines = text.splitlines()

    parts: list[str] = []
    carry = ""

    for raw_line in lines:

        line = raw_line.strip()

        if not line:
            if carry:
                parts.append(carry)
                carry = ""
            continue

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
# Regex helpers
# ---------------------------------------------------------------------------

def build_pattern(query_word: str) -> re.Pattern:
    """Build a case-insensitive whole-word matching pattern."""

    escaped = re.escape(query_word)

    return re.compile(
        rf"(?<![\w])(?:{escaped})(?![\w])",
        re.IGNORECASE,
    )


def mask_sentence(
    sentence: str,
    pattern: re.Pattern,
) -> str:
    """Replace every query-word match with [MASK]."""

    return pattern.sub("[MASK]", sentence)


def find_query(
    sentence: str,
    query_word: str,
    pattern: re.Pattern,
) -> str:
    """Return the query word if it matches in the sentence."""

    match = pattern.search(sentence)

    if match:
        return query_word

    return ""


def slugify(word: str) -> str:
    """Create a filesystem-safe slug."""

    return (
        re.sub(
            r"[^\w]+",
            "_",
            word.strip().lower(),
        )
        .strip("_")
    )


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_dataset_pairs(
    data_root: Path,
) -> list[tuple[Path, Path]]:
    """
    Find matching:

        *_metadata.jsonl
        *_content.jsonl

    pairs.
    """

    metadata_files = sorted(
        data_root.glob("*_metadata.jsonl")
    )

    pairs: list[tuple[Path, Path]] = []

    for metadata_path in metadata_files:

        prefix = metadata_path.name.removesuffix(
            "_metadata.jsonl"
        )

        content_path = (
            data_root
            / f"{prefix}_content.jsonl"
        )

        if not content_path.exists():

            logging.warning(
                "Missing content file for %s",
                metadata_path.name,
            )

            continue

        pairs.append(
            (
                metadata_path,
                content_path,
            )
        )

    logging.info(
        "Found %d metadata/content file pairs",
        len(pairs),
    )

    return pairs


# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------

def load_metadata(
    metadata_path: Path,
    date_min: int = DATE_MIN,
    date_max: int = DATE_MAX,
) -> dict[str, dict]:
    """
    Load LWM article metadata.

    Returns a dictionary keyed by article_id.
    """

    metadata: dict[str, dict] = {}

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as fh:

        for line_number, raw_line in enumerate(
            fh,
            start=1,
        ):

            raw_line = raw_line.strip()

            if not raw_line:
                continue

            try:
                row = json.loads(raw_line)

            except json.JSONDecodeError as exc:

                logging.warning(
                    "Invalid JSON in %s line %d: %s",
                    metadata_path.name,
                    line_number,
                    exc,
                )

                continue

            article_id = row.get("article_id")

            if not article_id:
                continue

            year = row.get("year")

            if year is None:

                raw_date = str(
                    row.get("date", "")
                ).strip()

                match = re.search(
                    r"\b(\d{4})\b",
                    raw_date,
                )

                if not match:
                    continue

                year = int(match.group(1))

            try:
                year = int(year)

            except (TypeError, ValueError):
                continue

            if not (
                date_min
                <= year
                <= date_max
            ):
                continue

            metadata[article_id] = row

    logging.info(
        "%s: %d metadata records passing date filter",
        metadata_path.name,
        len(metadata),
    )

    return metadata


# ---------------------------------------------------------------------------
# Article matching
# ---------------------------------------------------------------------------

def iter_article_matches(
    content_path: Path,
    metadata: dict[str, dict],
    query_word: str,
    pattern: re.Pattern,
) -> Iterator[dict]:
    """
    Read one LWM content JSONL file and yield sentence-level matches.
    """

    with content_path.open(
        "r",
        encoding="utf-8",
    ) as fh:

        for line_number, raw_line in enumerate(
            fh,
            start=1,
        ):

            raw_line = raw_line.strip()

            if not raw_line:
                continue

            try:
                article = json.loads(raw_line)

            except json.JSONDecodeError as exc:

                logging.warning(
                    "Invalid JSON in %s line %d: %s",
                    content_path.name,
                    line_number,
                    exc,
                )

                continue

            article_id = article.get("article_id")

            if not article_id:
                continue

            meta_record = metadata.get(article_id)

            if meta_record is None:
                continue

            raw_text = article.get(
                "text",
                "",
            )

            if not raw_text:
                continue

            processed_text = rejoin_text(
                raw_text
            )

            # Fast article-level rejection.
            if not pattern.search(
                processed_text
            ):
                continue

            sentences = sentencize(
                processed_text
            )

            for i, sentence in enumerate(
                sentences
            ):

                if not pattern.search(
                    sentence
                ):
                    continue

                prev_sentence = (
                    sentences[i - 1]
                    if i > 0
                    else ""
                )

                next_sentence = (
                    sentences[i + 1]
                    if i < len(sentences) - 1
                    else ""
                )

                yield {
                    # Article-level metadata
                    **meta_record,

                    # Query information
                    "query": find_query(
                        sentence,
                        query_word,
                        pattern,
                    ),

                    # Sentence-level fields
                    "prev_sentence": prev_sentence,
                    "sentence": sentence,
                    "masked_sentence": mask_sentence(
                        sentence,
                        pattern,
                    ),
                    "next_sentence": next_sentence,
                }

            del sentences
            del processed_text


# ---------------------------------------------------------------------------
# Main filtering function
# ---------------------------------------------------------------------------

def filter_dataset(
    query_word: str,
    output_path: Path,
    data_root: Path,
    date_min: int = DATE_MIN,
    date_max: int = DATE_MAX,
) -> None:

    pattern = build_pattern(
        query_word
    )

    logging.info(
        "Query word: %s",
        query_word,
    )

    logging.info(
        "Query pattern: %s",
        pattern.pattern,
    )

    file_pairs = find_dataset_pairs(
        data_root
    )

    if not file_pairs:

        logging.warning(
            "No metadata/content pairs found in %s",
            data_root,
        )

        return

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_written = 0
    total_articles = 0
    total_files = 0

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as out_fh:

        for metadata_path, content_path in file_pairs:

            total_files += 1

            logging.info("=" * 60)

            logging.info(
                "Processing %s",
                metadata_path.name,
            )

            metadata = load_metadata(
                metadata_path,
                date_min=date_min,
                date_max=date_max,
            )

            if not metadata:

                logging.info(
                    "No metadata records passed filters."
                )

                continue

            file_matches = 0
            seen_articles: set[str] = set()

            try:

                for record in iter_article_matches(
                    content_path=content_path,
                    metadata=metadata,
                    query_word=query_word,
                    pattern=pattern,
                ):

                    out_fh.write(
                        json.dumps(
                            record,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    file_matches += 1

                    article_id = record.get(
                        "article_id"
                    )

                    if article_id:
                        seen_articles.add(
                            article_id
                        )

            except Exception as exc:

                logging.error(
                    "Error processing %s: %s",
                    content_path,
                    exc,
                    exc_info=True,
                )

                continue

            file_articles = len(
                seen_articles
            )

            total_written += file_matches
            total_articles += file_articles

            logging.info(
                "%s → %d matching sentence(s) "
                "from %d article(s)",
                content_path.name,
                file_matches,
                file_articles,
            )

            del metadata
            gc.collect()

    logging.info("=" * 60)

    logging.info(
        "Done. Processed %d file pairs.",
        total_files,
    )

    logging.info(
        "Matching articles: %d",
        total_articles,
    )

    logging.info(
        "Total matching sentences: %d",
        total_written,
    )

    logging.info(
        "Output: %s",
        output_path,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Filter the LWM newspaper dataset by "
            "the query word specified in "
            "query_word_config.sh."
        ),
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("filtered_output"),
        help="Output directory",
    )

    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help=(
            "Explicit output file path "
            "(overrides automatic naming)"
        ),
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help="Directory containing LWM JSONL files",
    )

    parser.add_argument(
        "--query-config",
        type=Path,
        default=QUERY_CONFIG,
        help="Shell file containing QUERY_WORD",
    )

    parser.add_argument(
        "--date-min",
        type=int,
        default=DATE_MIN,
        help="Minimum publication year",
    )

    parser.add_argument(
        "--date-max",
        type=int,
        default=DATE_MAX,
        help="Maximum publication year",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=[
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
        ],
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    args = parse_args()

    logging.basicConfig(
        level=getattr(
            logging,
            args.log_level,
        ),
        format=(
            "%(asctime)s  "
            "%(levelname)-8s  "
            "%(message)s"
        ),
        datefmt="%H:%M:%S",
    )

    query_word = load_query_word(
        args.query_config
    )

    if args.output_file:

        output_path = args.output_file

    else:

        slug = slugify(
            query_word
        )

        output_path = (
            args.output_dir
            / f"lwm_{slug}.jsonl"
        )

    logging.info("=" * 60)

    logging.info(
        "LWM query word: %s",
        query_word,
    )

    logging.info(
        "Data root     : %s",
        args.data_root,
    )

    logging.info(
        "Date range    : %d–%d",
        args.date_min,
        args.date_max,
    )

    logging.info(
        "Output        : %s",
        output_path,
    )

    logging.info("=" * 60)

    filter_dataset(
        query_word=query_word,
        output_path=output_path,
        data_root=args.data_root,
        date_min=args.date_min,
        date_max=args.date_max,
    )


if __name__ == "__main__":
    main()