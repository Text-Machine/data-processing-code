#!/usr/bin/env python3

"""
Extract literary paragraphs from page-level OCR JSONL / JSONL.GZ.

Input format:
    One JSON object per line, with the OCR text in the "text" field.

The extractor attempts to reconstruct literary paragraphs by:

1. Reading the OCR page by page.
2. Preserving explicit blank-line paragraph boundaries when present.
3. Joining text across page boundaries.
4. Detecting likely paragraph boundaries from OCR line structure.
5. Avoiding splitting paragraphs simply because a page ended.

IMPORTANT:
If the OCR contains no paragraph/line-break information at all, exact
original literary paragraph boundaries cannot be reconstructed reliably.
In that case, the script uses a conservative sentence-based fallback.

Output:
    JSON list containing paragraph strings.
"""

import argparse
import gzip
import json
import random
import re
from pathlib import Path


def open_input(path: Path):
    """Open plain JSONL or gzip-compressed JSONL."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")

    return open(path, "r", encoding="utf-8")


def read_pages(input_path: Path, text_field: str = "text"):
    """Read page-level JSONL and return OCR text for each page."""

    pages = []

    with open_input(input_path) as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {exc}"
                ) from exc

            text = record.get(text_field)

            if not isinstance(text, str):
                continue

            text = text.strip()

            if not text:
                continue

            pages.append(text)

    return pages


def normalise_text(text):
    """
    Clean OCR whitespace while preserving paragraph markers.

    We deliberately do NOT simply replace all whitespace with spaces here,
    because newlines may contain useful paragraph information.
    """

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove excessive spaces while preserving newlines.
    text = re.sub(r"[ \t]+", " ", text)

    # Remove spaces immediately around newlines.
    text = re.sub(r" *\n *", "\n", text)

    # Collapse 3+ newlines to a paragraph separator.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def has_explicit_paragraph_structure(text):
    """
    Determine whether the OCR appears to contain explicit paragraph
    boundaries.

    We consider blank lines to be strong evidence.
    """

    return bool(re.search(r"\n\s*\n", text))


def split_explicit_paragraphs(text):
    """
    Split text using explicit blank lines.
    """

    paragraphs = re.split(r"\n\s*\n+", text)

    result = []

    for paragraph in paragraphs:
        paragraph = re.sub(r"\s+", " ", paragraph).strip()

        if paragraph:
            result.append(paragraph)

    return result


def join_page_continuations(pages):
    """
    Join page text while avoiding artificial paragraph boundaries.

    If a page ends in the middle of a sentence, the next page is joined
    directly with a space.

    If the page contains explicit paragraph breaks, those are preserved.
    """

    cleaned_pages = [normalise_text(page) for page in pages]

    if not cleaned_pages:
        return ""

    result = cleaned_pages[0]

    for page in cleaned_pages[1:]:
        if not page:
            continue

        # If the previous page ends with a sentence terminator, we still
        # cannot assume a paragraph ended there. Therefore join with a
        # space unless explicit blank-line structure exists.
        result += " " + page

    return result


def sentence_split(text):
    """
    Conservative sentence splitter used only as a fallback when the OCR
    has completely lost paragraph structure.

    This does NOT claim to recover the original literary paragraphs.
    """

    # Protect common abbreviations and initials.
    protected = text

    protected = re.sub(
        r"\b(Mr|Mrs|Ms|Dr|St|Prof|Rev|etc|i\.e|e\.g)\.",
        r"\1<prd>",
        protected,
        flags=re.IGNORECASE,
    )

    # Sentence boundaries.
    sentences = re.split(
        r'(?<=[.!?]["\'])\s+|(?<=[.!?])\s+(?=[A-Z"\'])',
        protected,
    )

    sentences = [
        sentence.replace("<prd>", ".").strip()
        for sentence in sentences
        if sentence.strip()
    ]

    return sentences


def build_paragraphs(pages, min_chars=200):
    """
    Attempt to reconstruct literary paragraphs.

    Strategy:
      1. If explicit paragraph boundaries exist, use them.
      2. Otherwise, use sentence groups as approximate paragraph units.

    For flattened OCR, approximately 3-8 sentences are grouped together
    to produce paragraph-sized chunks.
    """

    normalised_pages = [normalise_text(page) for page in pages]

    # ------------------------------------------------------------
    # Case 1: OCR retains explicit blank-line paragraph boundaries.
    # ------------------------------------------------------------

    if any(has_explicit_paragraph_structure(page)
           for page in normalised_pages):

        # Join pages while preserving their internal paragraph markers.
        text = "\n\n".join(normalised_pages)

        paragraphs = split_explicit_paragraphs(text)

        return [
            paragraph
            for paragraph in paragraphs
            if len(paragraph) >= min_chars
        ]

    # ------------------------------------------------------------
    # Case 2: OCR has flattened paragraph structure.
    #
    # We cannot recover the exact literary paragraphs. Instead,
    # construct paragraph-sized chunks from sentences.
    # ------------------------------------------------------------

    text = join_page_continuations(normalised_pages)

    sentences = sentence_split(text)

    paragraphs = []
    current = []

    for sentence in sentences:

        current.append(sentence)

        # Aim for literary-paragraph-sized chunks.
        #
        # We preferentially close a paragraph once it has:
        #   - at least 3 sentences, AND
        #   - at least min_chars characters.
        #
        # A hard upper bound prevents enormous chunks.
        current_text = " ".join(current)

        if (
            len(current) >= 3
            and len(current_text) >= min_chars
        ) or len(current_text) >= 1500:

            paragraphs.append(current_text)
            current = []

    if current:
        paragraphs.append(" ".join(current))

    return [
        paragraph.strip()
        for paragraph in paragraphs
        if len(paragraph.strip()) >= min_chars
    ]


def extract_paragraphs(
    input_path,
    n_paragraphs,
    text_field="text",
    min_chars=200,
    random_sample=False,
    seed=42,
):
    pages = read_pages(
        input_path=input_path,
        text_field=text_field,
    )

    if not pages:
        raise ValueError("No OCR text found in input file.")

    paragraphs = build_paragraphs(
        pages=pages,
        min_chars=min_chars,
    )

    print(f"Pages read: {len(pages)}")
    print(f"Candidate paragraphs reconstructed: {len(paragraphs)}")

    if len(paragraphs) < n_paragraphs:
        print(
            f"WARNING: only {len(paragraphs)} paragraph-sized chunks "
            f"were reconstructed; requested {n_paragraphs}."
        )

    if random_sample:
        if len(paragraphs) < n_paragraphs:
            raise ValueError(
                f"Cannot randomly sample {n_paragraphs} paragraphs "
                f"from only {len(paragraphs)} reconstructed paragraphs."
            )

        rng = random.Random(seed)
        paragraphs = rng.sample(paragraphs, n_paragraphs)

    else:
        paragraphs = paragraphs[:n_paragraphs]

    return paragraphs


def main():

    parser = argparse.ArgumentParser(
        description="Extract literary paragraphs from OCR JSONL/JSONL.GZ."
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSONL or JSONL.GZ file.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON file.",
    )

    parser.add_argument(
        "--text-field",
        default="text",
        help="OCR text field. Default: text",
    )

    parser.add_argument(
        "--n-paragraphs",
        type=int,
        default=100,
        help="Number of paragraphs to extract. Default: 100",
    )

    parser.add_argument(
        "--min-chars",
        type=int,
        default=200,
        help=(
            "Minimum paragraph length. "
            "Default: 200 characters."
        ),
    )

    parser.add_argument(
        "--random-sample",
        action="store_true",
        help="Randomly sample paragraphs instead of taking the first N.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42",
    )

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input file not found: {args.input}"
        )

    paragraphs = extract_paragraphs(
        input_path=args.input,
        n_paragraphs=args.n_paragraphs,
        text_field=args.text_field,
        min_chars=args.min_chars,
        random_sample=args.random_sample,
        seed=args.seed,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(
            paragraphs,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Extracted paragraphs: {len(paragraphs)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()