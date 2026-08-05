"""
Preprocess EEBO XML dataset into per-book text files and a metadata CSV.

Usage:
    python3 preprocess_eebo.py --input_zip /gpfs/projects/bsc100/textmachine-data/eebo_all.zip --output_dir ./gpfs/projects/bsc100/textmachine-data/preprocessed_data/output_eebo
"""

import csv
import re
import zipfile
import io
import argparse
from pathlib import Path
from lxml import etree


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

    id_nodes = header.findall(".//IDNO") if header is not None else []
    id_map = {n.get("TYPE"): safe_text(n) for n in id_nodes if n is not None}

    book_id = id_map.get("DLPS") or id_map.get("TCP") or id_map.get("eebo citation") or ""
    estc = id_map.get("ESTC", "")
    stc = id_map.get("stc", "")

    extent = find(".//EXTENT")
    pages = ""
    m = re.search(r"(\d+)\s*p", extent)
    if m:
        pages = m.group(1)

    return {
        "record_id": book_id,
        "book_id": book_id,
        "title": title,
        "author": author,
        "date": date,
        "place": place,
        "publisher": publisher,
        "language": language,
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
    """
    output_txt_dir = Path(output_txt_dir)
    output_txt_dir.mkdir(parents=True, exist_ok=True)

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

                #metadata["path_txt"] = str(txt_path)
                metadata["path_txt"] = str(txt_path.relative_to(output_txt_dir.parent))
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
        "record_id", "book_id", "title", "author", "date", "place",
        "publisher", "language", "number_of_pages", "num_words",
        "all_ids", "path_xml", "path_txt",
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
