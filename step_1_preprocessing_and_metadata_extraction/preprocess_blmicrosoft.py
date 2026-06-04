import argparse
import os
import tarfile
import gzip
import json
import pandas as pd

# ---- FILTER SETTINGS ----
ALLOWED_LANGS = {"English", "French", "Spanish", "Italian", "Dutch", "Russian"}

# ---- INPUT FILES ----
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


def process_jsonl_file(file_obj, file_path):
    records = []

    for line in file_obj:
        try:
            records.append(json.loads(line))
        except Exception:
            continue

    if not records:
        return None

    first = records[0]

    # ---- LANGUAGE FILTER ----
    multi_lang = first.get("multi_language")
    lang = first.get("Language_1")

    if isinstance(lang, str):
        lang = lang.strip().capitalize()

    if str(multi_lang).lower() == "true":
        return None

    if lang not in ALLOWED_LANGS:
        return None

    # ---- PARSE DATE ----
    date = first.get("date")
    year = None

    try:
        year = int(str(date)[:4])
    except Exception:
        pass

    # ---- TEXT STATS ----
    num_pages = 0
    num_words = 0

    for r in records:
        text = r.get("text")
        if text:
            num_pages += 1
            num_words += len(text.split())

    # ---- AGGREGATE ----
    mean_wc = [r["mean_wc_ocr"] for r in records if r.get("mean_wc_ocr") is not None]
    std_wc  = [r["std_wc_ocr"]  for r in records if r.get("std_wc_ocr")  is not None]

    metadata = {
        "record_id":                 first.get("record_id"),
        "date":                      first.get("date"),
        "title":                     first.get("title", ""),
        "place":                     first.get("place"),
        "mean_wc_ocr":               sum(mean_wc) / len(mean_wc) if mean_wc else None,
        "std_wc_ocr":                sum(std_wc)  / len(std_wc)  if std_wc  else None,
        "author":                    first.get("Name"),
        "all_authors":               first.get("All names"),
        "publisher":                 first.get("Publisher"),
        "main_publication_country":  first.get("Country of publication 1"),
        "all_publication_countries": first.get("All Countries of publication"),
        "main_language":             lang,
        "number_of_pages":           num_pages,
        "number_of_words":           num_words,
        "path_to_json":              file_path,
    }

    return metadata


def process_tar(tar_path):
    """Process one tar.gz archive."""
    results = []

    with tarfile.open(tar_path, "r:gz") as tar:
        members = tar.getmembers()

        print(f"{os.path.basename(tar_path)}: total members = {len(members)}")

        for i, member in enumerate(members):

            if not member.isfile():
                continue

            if not member.name.endswith(".jsonl.gz"):
                continue

            f = tar.extractfile(member)
            if f is None:
                continue

            try:
                with gzip.open(f, "rt", encoding="utf-8") as gz:
                    metadata = process_jsonl_file(gz, member.name)
                    if metadata:
                        results.append(metadata)
            except Exception as e:
                print(f"Error processing {member.name}: {e}")
                continue

            if i % 500 == 0:
                print(f"{os.path.basename(tar_path)}: processed {i}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract and preprocess BL Microsoft OCR metadata from tar.gz archives. "
            "Writes a single CSV file with one row per valid book."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing the source tar.gz archives.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path for the output metadata CSV.",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    first_write = True

    for tar_name in TARGET_TARS:
        tar_path = os.path.join(args.input_dir, tar_name)

        if not os.path.exists(tar_path):
            print(f"Missing: {tar_path}")
            continue

        print(f"\nProcessing {tar_name}")

        tar_results = process_tar(tar_path)

        if not tar_results:
            print(f"No valid books found in {tar_name}")
            continue

        df_chunk = pd.DataFrame(tar_results)
        df_chunk.to_csv(args.output, mode="a", header=first_write, index=False)
        first_write = False

        print(f"{tar_name}: wrote {len(df_chunk)} books")

    print(f"\nDone. Metadata saved to: {args.output}")


if __name__ == "__main__":
    main()