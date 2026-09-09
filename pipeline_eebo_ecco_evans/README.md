# EEBO, ECCO & Evans Datasets — Processing Pipeline

This document describes the end-to-end pipeline for producing fully processed, query-filtered, masked-word-prediction output for three TCP-transcribed historical text collections:

- **EEBO** (Early English Books Online, TCP)
- **ECCO** (Eighteenth Century Collections Online)
- **Evans** (Early American Imprints, TCP)

EEBO, ECCO, and Evans are processed as three **independent, parallel pipelines** that mirror each other step for step — same stage names, same script logic, different input data. Each dataset has its own directory, its own `query_word_config.sh`, and is run separately.

> **Note:** This requires having downloaded the relevant raw datasets first (the EEBO/ECCO XML archives, the Evans zip — see step 1 for exact source paths). Model paths for step 4 are **not yet finalized** for these three datasets — see the Step 4 section below before expecting real predictions out of that step.

---

## Directory Structure

```
pipeline_eebo_ecco_evans/
├── evans/
│   ├── step_1_preprocessing_and_metadata_extraction/
│   ├── step_2_data_filtering/
│   ├── step_3_mask_unrolling/
│   ├── step_4_bert_masked_word_prediction/
│   └── query_word_config.sh
├── eebo/
│   ├── step_1_preprocessing_and_metadata_extraction/
│   ├── step_2_data_filtering/
│   ├── step_3_mask_unrolling/
│   ├── step_4_bert_masked_word_prediction/
│   └── query_word_config.sh
├── ecco/
│   ├── step_1_preprocessing_and_metadata_extraction/
│   ├── step_2_data_filtering/
│   ├── step_3_mask_unrolling/
│   ├── step_4_bert_masked_word_prediction/
│   └── query_word_config.sh
└── README.md
```

Each of `evans/`, `eebo/`, and `ecco/` is a self-contained 4-step pipeline. Every step directory contains its own script plus an associated launcher; run each step from inside its own directory, in order, for whichever dataset you're processing.

### Environments

Step 1 (raw XML/zip preprocessing) only needs `lxml`:

```bash
python3 -m venv venv
source venv/bin/activate
pip install lxml
```

Steps 2–4 (filtering, mask unrolling, BERT prediction) use the shared `textmachine-llm-v2` conda environment (spaCy for sentence splitting, transformers/torch for step 4) — same environment as the BL Microsoft/HMD/LwM pipelines, no separate setup needed if you already have that.

---

## Pipeline Outputs

**1. Preprocessed text + metadata (Step 1)**

```
{data_root}/txt/...              -- one plain-text file per book/document
{data_root}/{name}_metadata.csv  -- one row per book/document
```

Unlike HMD/LwM (paired metadata + content JSONL per newspaper), each record here is one plain-text file located via the metadata CSV's `path_txt` column. EEBO is the one exception with two metadata CSVs instead of one — `phase1_metadata.csv` and `phase2_metadata.csv` — handled automatically by step 2, see below.

**2. Query-specific prediction files (Steps 2–4)**

```
{dataset}_final_{query}.jsonl
```

For example:

```
evans_final_machine.jsonl
eebo_final_machine.jsonl
ecco_final_machine.jsonl
```

---

## Pipeline Overview

| Step | Directory | Purpose |
|------|-----------|---------|
| 1 | `step_1_preprocessing_and_metadata_extraction/` | Converts the raw XML/zip archive into one plain-text file per book plus a metadata CSV. |
| 2 | `step_2_data_filtering/` | Filters records by query word, with sentence-level output including context sentences and masked versions. Output files are tagged `_step2`. |
| 3 | `step_3_mask_unrolling/` | Unrolls rows containing multiple instances of the query word, so each output row contains exactly one `[MASK]` token. Output files are tagged `_step3`. |
| 4 | `step_4_bert_masked_word_prediction/` | Runs masked-word prediction using the BERT models, one GPU per query word (up to the number of GPUs requested). **Model paths are currently placeholders — see below.** |

> **Naming convention note:** these three datasets use `_step2` / `_step3` tags (no underscore before the digit), which differs from BL Microsoft/HMD/LwM's `_step_4` / `_step_5` convention. `unroll_masks.py` (step 3) auto-detects and increments whichever style it finds, so this doesn't require separate scripts — just be aware the tag looks different if you're used to reading the other pipelines' filenames.

---

## Step 1 — Preprocessing and Metadata Extraction

Entry point for each dataset. Reads the raw XML/zip source data and produces one plain-text file per book/document plus a metadata CSV.

### EEBO

```bash
cd eebo/step_1_preprocessing_and_metadata_extraction/
sbatch launch_preprocess_eebo.sh
```

### ECCO

Lightweight enough to run directly from a transfer node (e.g. `transfer1`), no sbatch needed:

```bash
cd ecco/step_1_preprocessing_and_metadata_extraction/
source venv/bin/activate
python3 preprocess_ecco.py \
    --input_dir /gpfs/projects/bsc100/textmachine-data/ecco/p4/ecco_p4_released \
    --output_dir /gpfs/projects/bsc100/textmachine-data/preprocessed_data/output_ecco
```

### Evans

Also lightweight enough for a transfer node:

```bash
cd evans/step_1_preprocessing_and_metadata_extraction/
source venv/bin/activate
python3 preprocess_evans.py \
    --input_zip /gpfs/projects/bsc100/textmachine-data/evans.zip \
    --output_dir /gpfs/projects/bsc100/textmachine-data/preprocessed_data/output_evans
```

---

## Step 2 — Data Filtering with Query Words

Specify the query words used for sentence extraction and masked-word prediction (used by steps 2–4). Update the query word configuration for **each dataset separately** — `evans/query_word_config.sh`, `eebo/query_word_config.sh`, and `ecco/query_word_config.sh` are independent and don't need to match:

```bash
QUERY_WORDS=(machine machines)
```

Filters the preprocessed dataset by each configured query word: for each record in the metadata CSV, reads the corresponding plain-text file (via `path_txt`), sentence-splits it, and writes sentence-level output with context sentences and masked versions — same output shape as BL Microsoft/HMD/LwM's filtering step.

A few things `filter_tcp.py` handles automatically so you don't need dataset-specific scripts:

- **EEBO's two metadata CSVs** — discovers every `*.csv` in the dataset root rather than assuming one filename, so `phase1_metadata.csv` and `phase2_metadata.csv` are both picked up.
- **The `date`/`edition_date` column difference** — EEBO uses `date`, Evans and ECCO use `edition_date`; both are normalized to a `date` field in the output.
- **Very large documents** — some Evans records exceed 5 million characters (multi-volume bound works); spaCy's default 1M-character safety limit is raised, since only the lightweight rule-based sentencizer is used here, not the parser or NER components that limit exists to protect.
- **Language filtering** — defaults to `main_language == English`, override with `--language-filter ALL` if needed.

Runs one filtering process per query word in parallel within a single job (rather than one word per job), throttled by a concurrency limit:

```bash
cd evans/step_2_data_filtering/
sbatch launch_filter_tcp_parallel.sh   # DATASET="evans" at the top of the script
```

```bash
cd eebo/step_2_data_filtering/
sbatch launch_filter_tcp_parallel.sh   # DATASET="eebo"
```

```bash
cd ecco/step_2_data_filtering/
sbatch launch_filter_tcp_parallel.sh   # DATASET="ecco"
```

---

## Step 3 — Mask Unrolling

Unrolls rows that contain multiple instances of the query word, so each row in the output contains exactly one `[MASK]` token. This is the exact same `unroll_masks.py` script used by BL Microsoft, HMD, and LwM — it's dataset-agnostic (only ever touches `sentence`/`masked_sentence` fields), so the only thing that varies between `evans/`, `eebo/`, and `ecco/` is which `query_word_config.sh` and which step-2 output directory each launcher points at.

```bash
cd evans/step_3_mask_unrolling/
sbatch launch_unroll_tcp.sh   # DATASET="evans"
```

```bash
cd eebo/step_3_mask_unrolling/
sbatch launch_unroll_tcp.sh   # DATASET="eebo"
```

```bash
cd ecco/step_3_mask_unrolling/
sbatch launch_unroll_tcp.sh   # DATASET="ecco"
```

---

## Step 4 — BERT Masked Word Prediction

Runs masked-word prediction over the dataset, using the same multi-GPU `bert_predictions.py` engine as BL Microsoft/HMD/LwM (one GPU per query word, dynamic scheduling if word count exceeds GPU count — see GPU allocation below).

> **Model paths are not yet decided for EEBO, ECCO, or Evans.** `MODELS_BY_DATASET` in `bert_predictions.py` currently has two placeholder columns (`pred_model_1`, `pred_model_2`) with empty paths for all three datasets. Running step 4 as-is will discover the correct step-3 files, spin up workers, and then each worker will log a clear error and exit without attempting to load anything — it will **not** silently produce empty or wrong output, and it will **not** crash with a confusing `transformers` traceback. Fill in the two placeholder paths in `bert_predictions.py` once models are chosen, then rerun.

```bash
cd evans/step_4_bert_masked_word_prediction/
sbatch launch_bert_tcp.sh   # BERT_DATASET="evans"
```

```bash
cd eebo/step_4_bert_masked_word_prediction/
sbatch launch_bert_tcp.sh   # BERT_DATASET="eebo"
```

```bash
cd ecco/step_4_bert_masked_word_prediction/
sbatch launch_bert_tcp.sh   # BERT_DATASET="ecco"
```

### GPU allocation

Same rule as BL Microsoft/HMD/LwM — adjust `--gres=gpu:N` and `BERT_MAX_GPUS` in `launch_bert_tcp.sh` according to the number of query words being processed:

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
