# HMD & LwM Data Processing

## Overview

This directory contains the scripts used to preprocess and filter the HMD and LwM datasets.

The pipeline consists of two stages:

1. **Preprocessing and metadata extraction**
   - Converts the original CSV datasets into JSONL files.
   - Extracts metadata and article content into separate outputs.
   - Normalizes article identifiers and parses publication dates.

2. **Data filtering**
   - Filters the preprocessed datasets using regex query terms.
   - Produces subsets of articles matching the configured search expressions.

---

## Directory Structure

```
pipeline_hmd_lwm/
├── preprocess_and_metadata_extraction_hmd_lwm/
│   ├── preprocess_hmd.py
│   └── preprocess_lwm.py
│
└── data_filtering_hmd_lwm/
    ├── filter_hmd_lwm_nlpsentencizer.py
    ├── filter_hmd_lwm_simplelogic.py
    ├── launch_filter_hmd_lwm_high_mem.sh
    └── launch_filter_hmd_lwm_remaining_words.sh
```

---

## Stage 1 – Preprocessing and Metadata Extraction

### Overview

The preprocessing scripts:

- Read CSV files from ZIP archives.
- Process the data in chunks.
- Normalize article identifiers.
- Parse publication dates and apply date filters.
- Extract year, month, and day.
- Split each article into:
  - **Metadata**
  - **Content**
- Write JSONL output using `force_ascii=False` to preserve Unicode characters.

### Scripts

| Dataset | Script |
|---|---|
| HMD | `preprocess_hmd.py` |
| LwM | `preprocess_lwm.py` |

*(Existing Input Data, Output Data, and Usage sections can be kept largely unchanged here.)*

---

## Stage 2 – Data Filtering

### Overview

The filtering scripts search the preprocessed datasets for articles matching configurable regular-expression query terms.

Depending on the filtering strategy, they can:

- process the data in chunks,
- identify matching articles using regex patterns,
- optionally perform sentence segmentation before matching,
- write filtered metadata and content for downstream processing.

### Scripts

| Script | Description |
|---|---|
| `filter_hmd_lwm_nlpsentencizer.py` | Filters articles using regex matching with sentence segmentation. |
| `filter_hmd_lwm_simplelogic.py` | Filters articles using a simpler matching strategy without sentence segmentation. |

### Launch Scripts

- `launch_filter_hmd_lwm_high_mem.sh`
- `launch_filter_hmd_lwm_remaining_words.sh`

These shell scripts submit filtering jobs with different resource configurations.