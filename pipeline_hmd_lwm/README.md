# HMD & LwM Datasets — Processing Pipeline

This document describes the end-to-end pipeline for producing fully processed, query-filtered, masked-word-prediction output for the **HMD** (Heritage Made Digital) and **LwM** (Living with Machines) newspaper datasets.

HMD and LwM are processed as two **independent, parallel pipelines** that mirror each other step for step — same stage names, same script logic, different input data. Each dataset has its own directory, its own `query_word_config.sh`, and is run separately.

> **Note:** This requires having downloaded the relevant raw datasets first. See `data-processing-code/download_raw_data_code/README.md` for the HMD and LwM collection sections, and run `model_download.py` (top level of this directory) to fetch the BERT models used in step 4 before running that step for either dataset.

---

## Directory Structure

```
pipeline_hmd_lwm/
├── hmd/
│   ├── step_1_preprocessing_and_metadata_extraction/
│   ├── step_2_data_filtering/
│   ├── step_3_mask_unrolling/
│   ├── step_4_bert_masked_word_prediction/
│   └── query_word_config.sh
├── lwm/
│   ├── step_1_preprocessing_and_metadata_extraction/
│   ├── step_2_data_filtering/
│   ├── step_3_mask_unrolling/
│   ├── step_4_bert_masked_word_prediction/
│   └── query_word_config.sh
├── model_download.py
└── README.md
```

Each of `hmd/` and `lwm/` is a self-contained 4-step pipeline. Every step directory contains its own script plus an associated launcher (`launch.sh`); run each step from inside its own directory, in order, for whichever dataset you're processing.

---

## Pipeline Outputs

Each dataset pipeline produces two main outputs:

**1. Preprocessed metadata + content (Step 1)**

Paired JSONL files, one pair per newspaper/source file:

```
{prefix}_metadata.jsonl
{prefix}_content.jsonl
```

Metadata and article content are joined via `article_id`.

**2. Query-specific prediction files (Steps 2–4)**

```
{dataset}_final_{query}.jsonl
```

For example:

```
hmd_final_machine.jsonl
hmd_final_slave.jsonl
lwm_final_machine.jsonl
lwm_final_slave.jsonl
```

---

## Pipeline Overview

| Step | Directory | Purpose |
|------|-----------|---------|
| 1 | `step_1_preprocessing_and_metadata_extraction/` | Converts the raw dataset into paired `*_metadata.jsonl` / `*_content.jsonl` files, normalizes article identifiers, and parses publication dates. |
| 2 | `step_2_data_filtering/` | Filters articles by query word, with sentence-level output including context sentences and masked versions. |
| 3 | `step_3_mask_unrolling/` | Unrolls rows containing multiple instances of the query word, so each output row contains exactly one `[MASK]` token. |
| 4 | `step_4_bert_masked_word_prediction/` | Runs masked-word prediction using the BERT models, one GPU per query word (up to the number of GPUs requested). |



---

## Step 1 — Preprocessing and Metadata Extraction

Entry point for each dataset. Reads the raw CSV/ZIP source data, processes it in chunks, normalizes article identifiers, parses publication dates (extracting year/month/day), and splits each article into separate metadata and content JSONL files (written with `force_ascii=False` to preserve Unicode).

```bash
cd hmd/step_1_preprocessing_and_metadata_extraction/
sbatch launch.sh
```

```bash
cd lwm/step_1_preprocessing_and_metadata_extraction/
sbatch launch.sh
```

---

## Step 2 — Data Filtering with Query Words

Specify the query words used for sentence extraction and masked-word prediction (used by steps 2–4). Update the query word configuration for **each dataset separately** — `hmd/query_word_config.sh` and `lwm/query_word_config.sh` are independent and don't need to match:

```bash
QUERY_WORDS=(slave slaves machine machines mornings nights morning night)
```

Filters the preprocessed dataset by each configured query word, joining metadata and content via `article_id` and producing sentence-level output with context sentences and masked versions — same output shape as BL Microsoft's step 4.

```bash
cd hmd/step_2_data_filtering/
sbatch launch.sh
```

```bash
cd lwm/step_2_data_filtering/
sbatch launch.sh
```

---

## Step 3 — Mask Unrolling

Unrolls rows that contain multiple instances of the query word, so each row in the output contains exactly one `[MASK]` token. Row-level logic here is dataset-agnostic (HMD and LwM share the same unrolling script, and it also works unchanged on BL Microsoft's step-4 output), so the only thing that varies between `hmd/` and `lwm/` is which `query_word_config.sh` and which step-2 output directory each launcher points at.

```bash
cd hmd/step_3_mask_unrolling/
sbatch launch.sh
```

```bash
cd lwm/step_3_mask_unrolling/
sbatch launch.sh
```

---

## Step 4 — BERT Masked Word Prediction

Runs masked-word prediction over the dataset using the same BERT models as the BL Microsoft pipeline. As with BL Microsoft, this step is designed to use **one GPU per query word**, up to a maximum of 4 GPUs — files belonging to query words outside the current `query_word_config.sh` are never opened or reprocessed, so a run for a new word never touches previously completed words' output.

```bash
cd hmd/step_4_bert_masked_word_prediction/
sbatch launch.sh
```

```bash
cd lwm/step_4_bert_masked_word_prediction/
sbatch launch.sh
```

### GPU allocation

Adjust the GPU allocation in each `launch.sh` according to the number of query words being processed for that dataset run:

```bash
# 1 query word
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20

# 2 query words
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=40

# 3 query words
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=60

# 4 or more query words
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=80
```

If a run has fewer query words than GPUs requested, only that many GPUs are actually used — the extra ones aren't spun up just to sit idle. If a run has more query words than GPUs requested, GPUs pick up the next word dynamically as soon as they finish their current one, rather than being statically assigned up front.

---

## Model Download

Run the script below from the top level of `pipeline_hmd_lwm/`, before running step 4 for either dataset:

```bash
python3 model_download.py
```

---

