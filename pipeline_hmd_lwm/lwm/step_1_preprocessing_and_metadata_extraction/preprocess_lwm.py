import zipfile
import os
import pandas as pd

zip_path = "/gpfs/projects/bsc100/textmachine-data/downloaded_data_vm/17425252/lwm-csv.zip"
output_dir = "/gpfs/projects/bsc100/textmachine-data/preprocessed_data/output_lwm"
os.makedirs(output_dir, exist_ok=True)

chunksize = 5000
date_cutoff = "1900-01-01"  # optional filter

def process_csv_from_zip(zip_path, csv_filename, output_dir, chunksize=5000, date_cutoff=None):
    """Process a single CSV in the ZIP: split into metadata and content JSONL"""
    base_name = os.path.splitext(os.path.basename(csv_filename))[0]
    meta_out = os.path.join(output_dir, f"{base_name}_metadata.jsonl")
    content_out = os.path.join(output_dir, f"{base_name}_content.jsonl")

    for f in [meta_out, content_out]:
        if os.path.exists(f):
            os.remove(f)

    if date_cutoff is not None:
        date_cutoff = pd.to_datetime(date_cutoff)

    total_rows = 0
    with zipfile.ZipFile(zip_path) as z:
        with z.open(csv_filename) as f:
            reader = pd.read_csv(f, chunksize=chunksize)
            for i, chunk in enumerate(reader):
                chunk = chunk.rename(columns={"Unnamed: 0": "article_id"})

                # Clean article_id
                chunk["article_id"] = (
                    chunk["article_id"]
                    .astype(str)
                    .str.split("/")        # take last part of path
                    .str[-1]
                    .str.replace("_metadata.xml", "", regex=False))
                    
                                
                # Parse date once
                chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")

                # Apply cutoff if needed
                if date_cutoff is not None:
                    chunk = chunk[chunk["date"] < date_cutoff]

                if len(chunk) == 0:
                    continue
                
                chunk["year"]  = chunk["date"].dt.year
                chunk["month"] = chunk["date"].dt.month
                chunk["day"]   = chunk["date"].dt.day

                # Convert date to string (after extracting components)
                chunk["date"] = chunk["date"].dt.strftime("%Y-%m-%d")



                if len(chunk) == 0:
                    continue

                metadata_cols = [
                    "article_id", "article_headline", "item_type",
                    "ocr_quality_mean", "ocr_quality_sd", "word_count",
                    "date", "newspaper_title", "location",
                    "year", "month", "day", "NLP", "issue", "art_num"
                ]
                content_cols = ["article_id", "text"]

                meta_chunk = chunk[metadata_cols]
                content_chunk = chunk[content_cols]

                meta_chunk.to_json(meta_out, orient="records", lines=True, mode="a", force_ascii=False)
                #meta_chunk.to_json(meta_out,orient="records",lines=True,mode="a",date_format="iso")
                content_chunk.to_json(content_out, orient="records", lines=True, mode="a", force_ascii=False)

                total_rows += len(chunk)
                print(f"{csv_filename} - Chunk {i}: {len(chunk)} rows")

    print(f"{csv_filename} - Total rows written: {total_rows}")
    return meta_out, content_out

# ---- Loop over all CSV files in the ZIP ----
with zipfile.ZipFile(zip_path) as z:
    for csv_file in z.namelist():
        if csv_file.endswith(".csv"):
            print(f"\nProcessing file: {csv_file}")
            process_csv_from_zip(zip_path, csv_file, output_dir, chunksize=chunk