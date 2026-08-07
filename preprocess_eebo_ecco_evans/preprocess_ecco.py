"""
Preprocess ECCO XML dataset into per-book text files and a metadata CSV.

Usage:
    python3 preprocess_ecco.py --input_dir /gpfs/projects/bsc100/textmachine-data/ecco/p4/ecco_p4_released --output_dir /gpfs/projects/bsc100/textmachine-data/preprocessed_data/output_ecco
"""

import csv
import re
import argparse
from pathlib import Path
from lxml import etree

import os


# -----------------------------
# Language / author normalization helpers
# (shared logic across preprocess_ecco.py / preprocess_eebo.py / preprocess_evans.py)
# -----------------------------

# TCP corpora spell out <LANGUAGE> inconsistently across records (e.g. "Eng"
# vs "English", "Lat" vs "Latin"). This maps every variant we've seen onto a
# single canonical spelled-out form so the main_language column is usable
# for grouping/filtering downstream.
LANGUAGE_MAP = {
    "eng": "English",
    "english": "English",
    "lat": "Latin",
    "latin": "Latin",
    "fre": "French",
    "french": "French",
    "grc": "Greek",
    "greek": "Greek",
    "ger": "German",
    "german": "German",
    "spa": "Spanish",
    "spanish": "Spanish",
    "ita": "Italian",
    "italian": "Italian",
    "dut": "Dutch",
    "dutch": "Dutch",
    "wel": "Welsh",
    "welsh": "Welsh",
}


def parse_page_count(extent):
    """
    Extract a representative page count from a TCP bibliographic EXTENT
    string. These are free-text and inconsistent, e.g.:
      "16 p."              -> 16
      "[4], 147, [1] p."   -> 147  (front/back matter counts are bracketed;
                                     147 is the real, unbracketed pagination)
      "[18], 159, 43 p."   -> 159  (multi-part pagination: take the largest
                                     unbracketed run as the representative
                                     count, rather than trying to sum parts
                                     whose relationship to each other TCP
                                     doesn't spell out consistently)
      "1 sheet ([1] p.)"   -> 1    (broadside; "1" from "1 sheet")
      "[8] p."             -> 8    (every number is bracketed here - an
                                     editor-supplied count for an unpaginated
                                     item - so we fall back to using it)
    Heuristic: only look at the text up to the first "p." marker (anything
    after that, like "; 4to.", is a leaf-format/size code, not a page
    count), then take the largest UNbracketed number in that span. If
    every number in the span is bracketed, fall back to the largest
    bracketed one rather than returning nothing.
    """
    if not extent:
        return ""
    m_p = re.search(r"p\b", extent)
    if not m_p:
        return ""
    prefix = extent[:m_p.end()]

    numbers = []
    for m in re.finditer(r"(\[)?(\d+)(\])?", prefix):
        bracketed = bool(m.group(1) and m.group(3))
        numbers.append((bracketed, int(m.group(2))))
    if not numbers:
        return ""

    unbracketed = [n for br, n in numbers if not br]
    if unbracketed:
        return str(max(unbracketed))
    return str(max(n for _, n in numbers))


def normalize_language(raw):
    """Map a raw <LANGUAGE> value onto a single canonical spelled-out name."""
    if not raw:
        return ""
    key = raw.strip().lower().rstrip(".")
    return LANGUAGE_MAP.get(key, raw.strip())


def extract_author_years(author):
    """
    Pull birth/death years out of a TCP AUTHOR string, e.g.
    "Bunyan, John, 1628-1688." -> ("1628", "1688").
    Falls back to isolated "b. 1650" / "d. 1700" style annotations when no
    full range is present. Returns ("", "") if nothing usable is found.
    """
    if not author:
        return "", ""

    m = re.search(r"(\d{3,4})\s*-\s*(\d{3,4})", author)
    if m:
        return m.group(1), m.group(2)

    birth, death = "", ""
    m = re.search(r"\bb\.?\s*(\d{3,4})\b", author, re.IGNORECASE)
    if m:
        birth = m.group(1)
    m = re.search(r"\bd\.?\s*(\d{3,4})\b", author, re.IGNORECASE)
    if m:
        death = m.group(1)
    return birth, death


# -----------------------------
# XML parsing helpers
# -----------------------------

def safe_text(node):
    """Extract all text from an XML node safely."""
    if node is None:
        return ""
    return " ".join("".join(node.itertext()).split())


def extract_metadata(root, path):
    header = root.find(".//HEADER")

    def find(xpath):
        node = header.find(xpath) if header is not None else None
        return safe_text(node)

    title = find(".//TITLESTMT/TITLE")
    author = find(".//TITLESTMT/AUTHOR")
    place = find(".//PUBLICATIONSTMT/PUBPLACE")
    publisher = find(".//PUBLICATIONSTMT/PUBLISHER")
    edition_date = find(".//PUBLICATIONSTMT/DATE")
    main_language = normalize_language(find(".//LANGUAGE"))
    birth_year, death_year = extract_author_years(author)

    id_nodes = header.findall(".//IDNO") if header is not None else []
    id_map = {n.get("TYPE"): safe_text(n) for n in id_nodes if n is not None}

    book_id = id_map.get("TCP", "")
    estc = id_map.get("ESTC", "")
    dlps = id_map.get("DLPS", "")

    # NOTE: like EEBO/Evans, the header carries TWO <EXTENT> elements - a
    # top-level one under FILEDESC describing the scan/transcription (no
    # usable page count) and the actual bibliographic one nested deeper
    # (e.g. "[4], 147, [1] p."). The wrapper around that second EXTENT
    # varies across records (BIBLFULL vs BIBLSTRUCT vs others), so rather
    # than hard-coding one ancestor tag we take the LAST EXTENT under the
    # header - the technical one is reliably first in document order and
    # the bibliographic one comes after it regardless of what wraps it.
    # A plain ".//EXTENT" (first match) would return the wrong one.
    extent_nodes = header.findall(".//EXTENT") if header is not None else []
    extent = safe_text(extent_nodes[-1]) if extent_nodes else ""
    pages = parse_page_count(extent)

    return {
        "record_id": dlps or book_id,
        "book_id": book_id,
        "title": title,
        "author": author,
        "birth_year": birth_year,
        "death_year": death_year,
        "edition_date": edition_date,
        "place": place,
        "publisher": publisher,
        "main_language": main_language,
        "number_of_pages": pages,
        "all_ids": ";".join([x for x in [book_id, estc, dlps] if x]),
        "path_xml": str(path),
    }


def extract_text(root):
    text_body = root.find(".//TEXT")
    if text_body is None:
        return ""

    paragraphs = []
    for node in text_body.iter():
        # P/HEAD cover prose paragraphs and headings; L covers individual
        # lines of verse inside LG (line group) elements used for poetry.
        if node.tag in {"P", "HEAD", "L"}:
            txt = safe_text(node)
            if txt:
                txt = txt.replace("∣", "")  # remove OCR divider character
                paragraphs.append(txt)

    return "\n".join(paragraphs)


# -----------------------------
# File handling
# -----------------------------

def iter_xml_files_from_dir(input_dir):
    input_dir = Path(input_dir)
    for xml_path in input_dir.rglob("*.xml"):
        yield xml_path.read_bytes(), xml_path


def process_record(xml_bytes, path):
    """Parse a single XML record (bytes) and return its metadata dict + text."""
    root = etree.fromstring(xml_bytes)
    metadata = extract_metadata(root, path)
    text = extract_text(root)
    return metadata, text


def process_dataset(input_dir, output_csv, output_txt_dir):
    output_txt_dir = Path(output_txt_dir)
    output_txt_dir.mkdir(parents=True, exist_ok=True)
    output_csv = Path(output_csv)

    rows = []
    print(f"Processing: {input_dir}")

    for xml_bytes, path in iter_xml_files_from_dir(input_dir):
        try:
            metadata, text = process_record(xml_bytes, path)

            book_id = metadata.get("book_id") or Path(path).stem
            txt_path = output_txt_dir / f"{book_id}.txt"

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)

            # Store XML and TXT paths as clean relative paths, each relative
            # to its own root (input_dir for XML, output_txt_dir's parent
            # for TXT). This avoids absolute paths and avoids "../../"-style
            # traversal that would result from anchoring both to the CSV's
            # location, since input_dir and output_dir aren't nested.
            metadata["path_xml"] = str(Path(path).relative_to(Path(input_dir)))
            metadata["path_txt"] = str(txt_path.relative_to(output_txt_dir.parent))

            metadata["num_words"] = len(text.split())

            rows.append(metadata)

        except Exception as e:
            print(f"[ERROR] {path}: {e}")

    write_metadata_csv(rows, output_csv)

    print("\nDone.")
    print(f"CSV: {output_csv}")
    print(f"TXT dir: {output_txt_dir}")
    print(f"Books processed: {len(rows)}")


def write_metadata_csv(rows, output_csv):
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "record_id", "book_id", "title", "author", "birth_year", "death_year",
        "edition_date", "place", "publisher", "main_language",
        "number_of_pages", "num_words", "all_ids", "path_xml", "path_txt",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# -----------------------------
# Main driver
# -----------------------------

def main(input_dir, output_dir):
    output_dir = Path(output_dir)
    txt_dir = output_dir / "txt"
    txt_dir.mkdir(parents=True, exist_ok=True)

    process_dataset(input_dir, output_dir / "ecco_metadata.csv", txt_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ECCO XML dataset")
    parser.add_argument("--input_dir", required=True, help="Path to dataset root containing XML files")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    args = parser.parse_args()

    main(args.input_dir, args.output_dir)