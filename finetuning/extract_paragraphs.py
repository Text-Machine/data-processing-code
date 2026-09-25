#!/usr/bin/env python3
"""
extract_paragraphs.py

Prior-work step: given a digitized book stored as a page-level jsonl.gz file
(e.g. `1830_1839/1838_000932097.jsonl.gz`, one JSON object per line = one
scanned page), reconstruct paragraphs by grouping text that is NOT separated
by a blank line (i.e. by 2+ consecutive line breaks), and keep only the
first N paragraphs.

Expected input format (one JSON object per line):
    {"page": 1, "text": "CHAPTER I.\n\nThe Mile-End Road....\n\n..."}

The exact key holding the page text can vary across the BL/HMD/LWM/TCP
datasets used in the TextMachine pipeline, so it's configurable via
--text-field. If the field isn't found, the script falls back to the first
string-valued field in the record.

On MN5, individual books aren't sitting on disk as loose files -- they're
nested inside decade-range tar archives under
/gpfs/projects/bsc100/textmachine-data/downloaded_data_vm, e.g. the member
"1830_1839/1838_000932097.jsonl.gz" lives inside "OCR_text_1830_-_1839.tar.gz".
This script handles both cases:

  1) Direct file on disk:
     python extract_paragraphs.py \
         --input /path/to/1838_000932097.jsonl.gz \
         --output oliver_twist_paragraphs.json

  2) Inside a decade tar archive (auto-resolves which tar to use from the
     "1830_1839" prefix of --book-path):
     python extract_paragraphs.py \
         --input-dir /gpfs/projects/bsc100/textmachine-data/downloaded_data_vm \
         --book-path 1830_1839/1838_000932097.jsonl.gz \
         --output oliver_twist_paragraphs.json \
         --n-paragraphs 100

Output: a JSON file containing a list of paragraph strings (this is the
artifact consumed by preprocess_instruction_tuning.py).
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import tarfile
from pathlib import Path

# The decade/range tar archives available under INPUT_DIR on MN5. Used to
# resolve which tar a "1830_1839/<file>.jsonl.gz"-style --book-path lives in
# when the exact "OCR_text_{start}_-_{end}.tar.gz" name can't be guessed
# directly (e.g. the combined pre-1700 archive).
TARGET_TARS = [
    "OCR_text_c_1510_-_1699.tar.gz",
    "OCR_text_1700_-_1799.tar.gz",
    "OCR_text_1800_-_1809.tar.gz",
    "OCR_text_1810_-_1819.tar.gz",
    "OCR_text_1820_-_1829.tar.gz",
    "OCR_text_1830_-_1839.tar.gz",
    "OCR_text_1840_-_1849.tar.gz",
    "OCR_text_1850_-_1859.tar.gz",
    "OCR_text_1860_-_1869.tar.gz",
    "OCR_text_1870_-_1879.tar.gz",
    "OCR_text_1880_-_1889.tar.gz",
    "OCR_text_1890_-_1899.tar.gz",
]


def open_maybe_gzip(path: Path):
    """Open a file transparently, whether or not it's gzip-compressed."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def parse_year_range(tar_filename: str) -> tuple[int, int] | None:
    m = re.search(r"(\d{4})_-_(\d{4})", tar_filename)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def resolve_tar_for_book_path(input_dir: Path, book_path: str) -> Path:
    """
    Given a relative book path like "1830_1839/1838_000932097.jsonl.gz",
    figure out which OCR_text_*.tar.gz archive under input_dir contains it.
    """
    decade_folder = book_path.split("/")[0]
    m = re.match(r"(\d{4})_(\d{4})", decade_folder)
    if not m:
        raise ValueError(
            f"Couldn't parse a year range from book-path prefix '{decade_folder}' "
            f"(expected something like '1830_1839/...')"
        )
    start_year = int(m.group(1))

    # 1) Try the exact-name match first: 1830_1839 -> OCR_text_1830_-_1839.tar.gz
    exact_name = f"OCR_text_{m.group(1)}_-_{m.group(2)}.tar.gz"
    exact_path = input_dir / exact_name
    if exact_path.exists():
        return exact_path

    # 2) Fall back to scanning the known archive names for one whose year
    #    range brackets start_year (handles the combined pre-1700 archive
    #    and the 1700-1799 archive, which don't follow the decade pattern).
    for name in TARGET_TARS:
        yr = parse_year_range(name)
        if yr and yr[0] <= start_year <= yr[1] and (input_dir / name).exists():
            return input_dir / name

    # 3) Last resort: glob whatever is actually on disk under input_dir.
    for p in sorted(input_dir.glob("OCR_text_*.tar.gz")):
        yr = parse_year_range(p.name)
        if yr and yr[0] <= start_year <= yr[1]:
            return p

    raise FileNotFoundError(
        f"No archive under {input_dir} covers year {start_year} "
        f"(book-path prefix '{decade_folder}')"
    )


def read_jsonl_bytes_from_tar(tar_path: Path, book_path: str) -> bytes:
    """
    Stream through tar_path (a .tar.gz) looking for the member matching
    book_path (or, failing that, matching its basename), and return the raw
    bytes of that member -- which are themselves gzip-compressed (the
    member is a .jsonl.gz file nested inside the outer tar.gz).

    Uses streaming mode ('r|gz') since tar has no index: this decompresses
    the archive sequentially up to the matching member and then stops, so
    it's a one-off pass rather than pulling the whole archive to disk.
    """
    target_basename = Path(book_path).name
    with tarfile.open(tar_path, mode="r|gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            if member.name == book_path or member.name.endswith("/" + book_path) or Path(member.name).name == target_basename:
                fileobj = tar.extractfile(member)
                if fileobj is None:
                    continue
                return fileobj.read()
    raise FileNotFoundError(f"No member matching '{book_path}' found in {tar_path}")


def iter_jsonl_lines(raw_bytes: bytes):
    """The tar member is itself a gzip file (double-compressed); decompress
    it and yield its lines as text."""
    text = gzip.decompress(raw_bytes).decode("utf-8", errors="replace")
    for line in text.splitlines():
        yield line


def parse_pages(lines, text_field: str) -> list[str]:
    """Parse an iterable of jsonl text lines into an ordered list of page texts."""
    pages = []
    for line_no, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"[warn] skipping malformed line {line_no}: {e}", file=sys.stderr)
            continue

        text = record.get(text_field)
        if text is None:
            # Fall back to the first string field we can find.
            text = next(
                (v for v in record.values() if isinstance(v, str) and v.strip()),
                None,
            )
        if text:
            pages.append(text)
    return pages


def load_page_texts(args) -> list[str]:
    """
    Dispatch to either direct-file mode (--input) or tar-archive mode
    (--input-dir + --book-path), and return the ordered list of page texts.
    """
    if args.input is not None:
        print(f"[info] reading direct file {args.input}", file=sys.stderr)
        with open_maybe_gzip(args.input) as f:
            return parse_pages(f, args.text_field)

    if args.input_dir is None or args.book_path is None:
        raise SystemExit("Provide either --input, or both --input-dir and --book-path")

    tar_path = resolve_tar_for_book_path(args.input_dir, args.book_path)
    print(f"[info] resolved '{args.book_path}' -> {tar_path}", file=sys.stderr)
    raw_bytes = read_jsonl_bytes_from_tar(tar_path, args.book_path)
    print(f"[info] extracted member ({len(raw_bytes)} compressed bytes) from tar", file=sys.stderr)
    return parse_pages(iter_jsonl_lines(raw_bytes), args.text_field)


def pages_to_paragraphs(pages: list[str]) -> list[str]:
    """
    Concatenate pages (joined by a blank line, so a paragraph never
    accidentally merges across a page break with no blank line at the
    page boundary) and split into paragraphs on runs of 2+ line breaks.
    """
    full_text = "\n\n".join(pages)

    # Normalize Windows/old-mac line endings, then split on any run of
    # 2+ newlines (allowing trailing whitespace on the blank line(s)).
    full_text = full_text.replace("\r\n", "\n").replace("\r", "\n")
    raw_paragraphs = re.split(r"\n\s*\n+", full_text)

    paragraphs = []
    for p in raw_paragraphs:
        # Collapse internal single line breaks (and stray whitespace) so
        # each paragraph is one clean line of prose.
        cleaned = re.sub(r"\s+", " ", p).strip()
        if cleaned:
            paragraphs.append(cleaned)
    return paragraphs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=None, help="Direct path to a page-level jsonl(.gz) book file on disk")
    parser.add_argument("--input-dir", type=Path, default=None, help="Root dir containing the OCR_text_*.tar.gz decade archives (e.g. /gpfs/projects/bsc100/textmachine-data/downloaded_data_vm)")
    parser.add_argument("--book-path", default=None, help="Relative path of the book inside its decade tar, e.g. '1830_1839/1838_000932097.jsonl.gz'")
    parser.add_argument("--output", required=True, type=Path, help="Where to write the JSON list of paragraphs")
    parser.add_argument("--text-field", default="text", help="Key in each page record holding the OCR text (default: 'text')")
    parser.add_argument("--n-paragraphs", type=int, default=100, help="Number of leading paragraphs to keep (default: 100)")
    parser.add_argument("--min-chars", type=int, default=20, help="Drop paragraphs shorter than this many characters (headers/page numbers/noise)")
    args = parser.parse_args()

    pages = load_page_texts(args)
    print(f"[info] loaded {len(pages)} pages", file=sys.stderr)

    paragraphs = pages_to_paragraphs(pages)
    paragraphs = [p for p in paragraphs if len(p) >= args.min_chars]
    print(f"[info] extracted {len(paragraphs)} paragraphs (after min-chars filter)", file=sys.stderr)

    kept = paragraphs[: args.n_paragraphs]
    print(f"[info] keeping first {len(kept)} paragraphs", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)
    print(f"[info] wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()