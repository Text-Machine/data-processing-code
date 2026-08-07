"""
Preprocess EEBO XML dataset into per-book text files and a metadata CSV.

Usage:
    python3 preprocess_eebo.py --input_zip /gpfs/projects/bsc100/textmachine-data/eebo_all.zip --output_dir /gpfs/projects/bsc100/textmachine-data/preprocessed_data/output_eebo
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
    date = find(".//PUBLICATIONSTMT/DATE")
    language = find(".//LANGUAGE")
    main_language = normalize_language(language)
    birth_year, death_year = extract_author_years(author)

    id_nodes = header.findall(".//IDNO") if header is not None else []
    id_map = {n.get("TYPE"): safe_text(n) for n in id_nodes if n is not None}

    book_id = id_map.get("DLPS") or id_map.get("TCP") or id_map.get("eebo citation") or ""
    estc = id_map.get("ESTC", "")
    stc = id_map.get("stc", "")

    # NOTE: the header carries TWO <EXTENT> elements - a top-level one
    # under FILEDESC describing the scan/transcription (e.g. "Approx.
    # 244 KB ... transcribed from 91 ... TIFF page images.", never a
    # usable page count) and the actual bibliographic one nested deeper
    # (e.g. "[4], 147, [1] p."). The wrapper around that second EXTENT
    # varies across records/phases (BIBLFULL vs BIBLSTRUCT vs others), so
    # rather than hard-coding one ancestor tag we take the LAST EXTENT
    # under the header - the technical one is reliably first in document
    # order and the bibliographic one comes after it regardless of what
    # wraps it. A plain ".//EXTENT" (first match) would return the wrong,
    # non-bibliographic one.
    extent_nodes = header.findall(".//EXTENT") if header is not None else []
    extent = safe_text(extent_nodes[-1]) if extent_nodes else ""
    pages = parse_page_count(extent)

    return {
        "record_id": book_id,
        "book_id": book_id,
        "title": title,
        "author": author,
        "birth_year": birth_year,
        "death_year": death_year,
        "date": date,
        "place": place,
        "publisher": publisher,
        "main_language": main_language,
        "number_of_pages": pages,
        "all_ids": ";".join([x for x in [book_id, estc, stc] if x]),
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
    Handles nested structure: outer.zip -> A0.zip -> A0/*.xml
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


def process_dataset(zip_paths, output_csv, output_txt_dir):
    """
    Process one or more (nested) zip files belonging to the same phase,
    accumulating all metadata rows in memory and writing the metadata CSV
    exactly once at the end, after every zip has been processed.

    zip_paths: iterable of paths to phase zip files (e.g. one tmp zip per
               inner phase zip extracted from the outer archive).

    output_txt_dir is expected to be a phase subdirectory directly under
    the shared "txt" directory (i.e. txt_dir / "phaseN"), so that path_txt
    below can be written relative to txt_dir's parent (output_dir),
    matching the "txt/<...>.txt" convention used by preprocess_ecco.py and
    preprocess_evans.py.
    """
    output_txt_dir = Path(output_txt_dir)
    output_txt_dir.mkdir(parents=True, exist_ok=True)

    # output_txt_dir = <output_dir>/txt/phaseN, so parent.parent = <output_dir>
    path_txt_root = output_txt_dir.parent.parent

    rows = []
    seen_book_ids = set()

    for zip_path in zip_paths:
        print(f"Processing: {zip_path}")

        for xml_bytes, name in iter_xml_files_from_zip(zip_path):
            try:
                metadata, text = process_record(xml_bytes, name)

                book_id = metadata.get("book_id") or Path(name).stem

                # Guard against txt filename collisions across different
                # inner zips (or malformed records without a usable ID):
                # if we've already written this book_id, disambiguate the
                # filename instead of silently overwriting the earlier file.
                if book_id in seen_book_ids:
                    safe_stem = Path(name).stem
                    print(f"[WARN] duplicate book_id '{book_id}' encountered "
                          f"again in {zip_path} ({name}); "
                          f"disambiguating output filename")
                    book_id = f"{book_id}__{safe_stem}"
                seen_book_ids.add(book_id)

                txt_path = output_txt_dir / f"{book_id}.txt"

                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(text)

                metadata["path_txt"] = str(txt_path.relative_to(path_txt_root))
                metadata["num_words"] = len(text.split())

                rows.append(metadata)
            except Exception as e:
                print(f"[ERROR] {name}: {e}")

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
        "date", "place", "publisher", "main_language",
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

    phase1_marker = "eebo_phase1/P4_XML_TCP"
    phase2_marker = "eebo_phase2/P4_XML_TCP_Ph2"

    with zipfile.ZipFile(input_zip, "r") as z:
        phase1_zips = [n for n in z.namelist() if phase1_marker in n and n.endswith(".zip")]
        phase2_zips = [n for n in z.namelist() if phase2_marker in n and n.endswith(".zip")]

        # Extract every inner zip for a phase to its own temp file first,
        # then hand the FULL list to process_dataset() so metadata rows
        # from all inner zips in the phase are accumulated together and
        # the CSV is written exactly once (previously each inner zip
        # overwrote the CSV from the last call, losing all prior rows).
        phase1_tmp_paths = []
        for i, p in enumerate(phase1_zips):
            tmp_path = output_dir / f"tmp_phase1_{i}.zip"
            tmp_path.write_bytes(z.read(p))
            phase1_tmp_paths.append(tmp_path)

        if phase1_tmp_paths:
            process_dataset(phase1_tmp_paths, output_dir / "phase1_metadata.csv", txt_dir / "phase1")
            for tmp_path in phase1_tmp_paths:
                tmp_path.unlink(missing_ok=True)

        phase2_tmp_paths = []
        for i, p in enumerate(phase2_zips):
            tmp_path = output_dir / f"tmp_phase2_{i}.zip"
            tmp_path.write_bytes(z.read(p))
            phase2_tmp_paths.append(tmp_path)

        if phase2_tmp_paths:
            process_dataset(phase2_tmp_paths, output_dir / "phase2_metadata.csv", txt_dir / "phase2")
            for tmp_path in phase2_tmp_paths:
                tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process EEBO XML dataset")
    parser.add_argument("--input_zip", required=True, help="Path to eebo_all.zip")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    args = parser.parse_args()

    main(args.input_zip, args.output_dir)