# Data Preprocessing

## Overview

This project contains raw datasets and their corresponding preprocessed outputs. The preprocessing pipeline is divided into three separate scripts, each responsible for a specific dataset.

All preprocessing scripts require:

```
pandas==3.0.1
```

---

## Preprocessing Logic Summary

### BL Microsoft (`preprocess_blmicrosoft.py`)
* Iterates over multiple `.tar.gz` archives containing compressed `.jsonl.gz` files (one per book).
* Extracts and parses each JSONL file line-by-line.
* Filters out:
  * Multi-language records
  * Books not in a predefined set of allowed languages
* Aggregates per-book statistics (e.g., mean and standard deviation of OCR word counts).
* Outputs a single consolidated CSV file with one row per book.

### HMD (`preprocess_hmd.py`)
- Iterates over all CSV files inside a ZIP archive.  
- Processes the data in chunks. 
- Generates a normalized `article_id` from the `plain_text_file` path.  
- Parses the `date` column and applies a cutoff filter (e.g., pre-1900 data).   
- Extracts additional temporal features: **year, month, and day**.  
- Splits each chunk into:
  - **Metadata:** structured fields (e.g., headline, OCR quality, publication details, date features)  
  - **Content:** raw article text  
- Writes outputs as JSONL files (one metadata file and one content file per input CSV).
- Uses `force_ascii=False` when writing JSONL to preserve non-ASCII characters (e.g., accented letters and non-English text).  

---

### LWM (`preprocess_lwm.py`)
- Iterates over all CSV files inside a ZIP archive.  
- Processes the data in chunks.
- Cleans and standardizes the `article_id` from source-specific filename identifiers.  
- Parses the `date` column and applies a cutoff filter (e.g., pre-1900 data).  
- Extracts **year, month, and day** from the parsed date.  
- Splits each chunk into:
  - **Metadata:** structured fields, simialr to the HMD schema  
  - **Content:** raw article text  
- Writes outputs as JSONL files (one metadata file and one content file per input CSV).
- Uses `force_ascii=False` when writing JSONL to preserve non-ASCII characters (e.g., accented letters and non-English text).   
---

## Input Data

### BL Microsoft Input

* **Path:** `/gpfs/projects/bsc100/textmachine-data/downloaded_data_vm`
* **Number of archives:** 12
* **Total size:** 9.77 GB
* **Total files inside archives:** 48,106

---

### HMD Input

* **Path:** `/gpfs/projects/bsc100/textmachine-data/downloaded_data_vm/15056046/hmd-csv.zip`
* **Size:** 5.57 GB
* **Description:** ZIP archive containing 14 files

---

### LwM Input

* **Path:** `/gpfs/projects/bsc100/textmachine-data/downloaded_data_vm/17425252/lwm-csv.zip`
* **Size:** 8.85 GB
* **Description:** ZIP archive containing 107 files

---

## Output Data

### BL Microsoft Output

* **Path:** `/gpfs/projects/bsc100/textmachine-data/preprocessed_data/output_blmicrosoft`
* **Size:** 11.63 MB
* **Structure:** 1 file (no subfolders)

---

### HMD Output

* **Path:** `/gpfs/projects/bsc100/textmachine-data/preprocessed_data/output_hmd`
* **Size:** 13.60 GB
* **Structure:** 28 files (no subfolders)

---

### LwM Output

* **Path:** `/gpfs/projects/bsc100/textmachine-data/preprocessed_data/output_lwm`
* **Size:** 15.44 GB
* **Structure:** 202 files (no subfolders)

---

## Preprocessing Scripts

The following scripts must be executed to generate the outputs:

* `preprocess_blmicrosoft.py`
* `preprocess_hmd.py`
* `preprocess_lwm.py`

Each script processes its corresponding dataset independently. Make sure the input data is available at the specified paths before running them.

---

## Usage

Example:

```bash
python3 preprocess_blmicrosoft.py
python3 preprocess_hmd.py
python3 preprocess_lwm.py
```


