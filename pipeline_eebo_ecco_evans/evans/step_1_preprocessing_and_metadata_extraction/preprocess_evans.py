"""
Preprocess Evans (Early American Imprints, TCP) XML dataset into per-book
text files and a metadata CSV.

Usage:
    python3 preprocess_evans.py --input_zip /gpfs/projects/bsc100/textmachine-data/evans.zip --output_dir /gpfs/projects/bsc100/textmachine-data/preprocessed_data/output_evans
"""

import csv
import re
import zipfile
import io
import argparse
from pathlib import Path
from lxml import etree


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
# (same TCP schema as preprocess_eebo.py / preprocess_ecco.py)
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

    # Build id_map as TYPE -> list of values, not TYPE -> single value.
    # Evans records commonly have *two* <IDNO TYPE="stc"> entries (an Evans
    # number and a Wing number) - a plain dict comprehension would silently
    # drop the first one whenever a later IDNO reuses the same TYPE.
    id_nodes = header.findall(".//IDNO") if header is not None else []
    id_map = {}
    for n in id_nodes:
        if n is None:
            continue
        value = safe_text(n)
        if value:
            id_map.setdefault(n.get("TYPE"), []).append(value)

    def first_id(*types):
        for t in types:
            values = id_map.get(t)
            if values:
                return values[0]
        return ""

    book_id = first_id("DLPS", "TCP", "tcp", "eebo citation", "evans citation")
    estc = ";".join(id_map.get("ESTC", []))
    stc = ";".join(id_map.get("stc", []))

    # NOTE: like EEBO/ECCO, the header carries TWO <EXTENT> elements - a
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

    # all_ids keeps every IDNO value found, regardless of TYPE, so nothing
    # from the header is lost even if multiple IDNOs share the same TYPE.
    all_ids = ";".join(safe_text(n) for n in id_nodes if safe_text(n))

    return {
        "record_id": book_id,
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
        "all_ids": all_ids,
        "path_xml": str(path),
    }


# Text-bearing "leaf" tags: once one of these is reached while walking the
# tree, its full text is captured in one go and we do NOT recurse into its
# children. This matters because TCP-encoded texts can nest text-bearing
# tags inside one another (e.g. verse quoted inside a paragraph via
# <P>...<Q><LG><L>...</L></LG></Q></P>). Without this stop-at-leaf rule, a
# flat "does this tag match?" scan over every descendant would capture the
# same words twice: once via the parent's full itertext(), and again when
# separately visiting each nested line.
#   P        - prose paragraph
#   HEAD     - heading/title
#   L        - a single line of verse (inside LG line-groups)
#   Q        - block quotation (plain prose OR quoted verse)
#   CLOSER   - letter closing block (wraps DATELINE/SIGNED)
#   DATELINE - standalone dateline, when not already inside CLOSER
#   SIGNED   - standalone signature line, when not already inside CLOSER
#   ARGUMENT - chapter/section summary
#   EPIGRAPH - epigraph preceding a section
#   POSTSCRIPT
#   ITEM     - list item
#   TRAILER  - closing text (e.g. "FINIS.")
LEAF_TAGS = {
    "P", "HEAD", "L", "Q", "CLOSER", "DATELINE", "SIGNED",
    "ARGUMENT", "EPIGRAPH", "POSTSCRIPT", "ITEM", "TRAILER",
}


def _collect_leaf_text(node, chunks):
    if node.tag in LEAF_TAGS:
        txt = safe_text(node)
        if txt:
            txt = txt.replace("∣", "")  # remove OCR divider character
            chunks.append(txt)
        return  # text already fully captured; don't descend further

    for child in node:
        _collect_leaf_text(child, chunks)


def extract_text(root):
    text_body = root.find(".//TEXT")
    if text_body is None:
        return ""

    chunks = []
    _collect_leaf_text(text_body, chunks)
    return "\n".join(chunks)


# -----------------------------
# ZIP handling
# -----------------------------

def iter_xml_files_from_zip(zip_path):
    """
    Recursively yields (xml_bytes, name) from a zip file, descending into
    any nested zips it finds (handles both a flat zip-of-xml and a
    zip-of-zip-of-xml layout without needing to know which in advance).
    """
    with zipfile.ZipFile(zip_path, "r") as z:
        for name in z.namelist():
            if name.endswith(".zip"):
                with z.open(name) as inner_zip_file:
                    with zipfile.ZipFile(io.BytesIO(inner_zip_file.read())) as inner_z:
                        for inner_name in inner_z.namelist():
                            if inner_name.endswith(".xml"):
                                yield inner_z.read(inner_name), inner_name
            elif name.endswith(".xml"):
                yield z.read(name), name


def process_record(xml_bytes, name):
    """Parse a single XML record (bytes) and return its metadata dict + text."""
    root = etree.fromstring(xml_bytes)
    metadata = extract_metadata(root, name)
    text = extract_text(root)
    return metadata, text


def process_zip_part(zip_path, output_txt_dir):
    """Process one partition zip (e.g. N0.zip) and return its metadata
    rows. Does NOT write the CSV itself, so the caller can accumulate rows
    across multiple partitions and write a single combined CSV at the end
    -- writing per-partition CSVs to the same path would silently overwrite
    all but the last partition's rows."""
    output_txt_dir = Path(output_txt_dir)
    output_txt_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    print(f"Processing: {zip_path}")

    for xml_bytes, name in iter_xml_files_from_zip(zip_path):
        try:
            metadata, text = process_record(xml_bytes, name)

            book_id = metadata.get("book_id") or Path(name).stem
            txt_path = output_txt_dir / f"{book_id}.txt"

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)

            #metadata["path_txt"] = str(txt_path)
            
            metadata["path_txt"] = str(txt_path.relative_to(output_txt_dir.parent))
            metadata["num_words"] = len(text.split())

            rows.append(metadata)
        except Exception as e:
            print(f"[ERROR] {name}: {e}")

    print(f"  -> {len(rows)} books")
    return rows


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

def main(input_zip, output_dir):
    output_dir = Path(output_dir)
    txt_dir = output_dir / "txt"
    txt_dir.mkdir(parents=True, exist_ok=True)

    partition_marker = "P4_XML_TCP"

    with zipfile.ZipFile(input_zip, "r") as z:
        partition_zips = sorted(
            n for n in z.namelist() if partition_marker in n and re.search(r"\d+\.zip$", n)
        )

        if not partition_zips:
            print(f"No partition zips found under '{partition_marker}' in {input_zip}")
            return

        all_rows = []
        for p in partition_zips:
            partition_name = Path(p).stem  # e.g. "N0", "N1", "N2", "N3"
            tmp_path = output_dir / f"tmp_{partition_name}.zip"
            tmp_path.write_bytes(z.read(p))

            try:
                rows = process_zip_part(tmp_path, txt_dir)
                all_rows.extend(rows)
            finally:
                tmp_path.unlink(missing_ok=True)

    write_metadata_csv(all_rows, output_dir / "evans_metadata.csv")

    print("\nDone.")
    print(f"CSV: {output_dir / 'evans_metadata.csv'}")
    print(f"TXT dir: {txt_dir}")
    print(f"Books processed: {len(all_rows)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Evans (Early American Imprints) TCP XML dataset")
    parser.add_argument("--input_zip", required=True, help="Path to evans.zip")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    args = parser.parse_args()

    main(args.input_zip, args.output_dir)