"""
Preprocess ECCO XML dataset into per-book text files and a metadata CSV.

Usage:
    python3 preprocess_ecco.py --input_dir /gpfs/projects/bsc100/textmachine-data/ecco/p4/ecco_p4_released --output_dir ./output
"""

import csv
import re
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

    book_id = id_map.get("TCP", "")
    estc = id_map.get("ESTC", "")
    dlps = id_map.get("DLPS", "")

    extent = find(".//EXTENT")
    pages = ""
    m = re.search(r"(\d+)\s*p", extent)
    if m:
        pages = m.group(1)

    return {
        "record_id": dlps or book_id,
        "book_id": book_id,
        "title": title,
        "author": author,
        "date": date,
        "place": place,
        "publisher": publisher,
        "language": language,
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

    rows = []
    print(f"Processing: {input_dir}")

    for xml_bytes, path in iter_xml_files_from_dir(input_dir):
        try:
            metadata, text = process_record(xml_bytes, path)

            book_id = metadata.get("book_id") or Path(path).stem
            txt_path = output_txt_dir / f"{book_id}.txt"

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)

            metadata["path_txt"] = str(txt_path)
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

def main(input_dir, output_dir):
    output_dir = Path(output_dir)
    txt_dir = output_dir / "txt"
    txt_dir.mkdir(parents=True, exist_ok=True)

    process_dataset(input_dir, output_dir / "metadata_ecco.csv", txt_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ECCO XML dataset")
    parser.add_argument("--input_dir", required=True, help="Path to dataset root containing XML files")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    args = parser.parse_args()

    main(args.input_dir, args.output_dir)