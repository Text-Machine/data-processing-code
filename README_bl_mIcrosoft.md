# BL Microsoft Dataset — Metadata Processing Pipeline

This document describes the end-to-end pipeline for producing a fully processed metadata file for the BL Microsoft dataset. The pipeline is currently spread across several feature branches of the `data-processing-code` repository and must be executed in the order described below.

> **Note:** This requires having downloaded the relevant raw dataset as explained in `data-processing-code/download_raw_data_code/README.md` (see section called  `2. BL Microsoft Collection`).

---

## Pipeline Overview

Every script file needs to be launched with the associated launcher from its respective directory. Every directory also contains a README.md documenting the script ans associate launcher

| Step | Script | Branch | Execution time | 
|------|--------|--------|--------| 
| 1 | `step_1_preprocessing_and_metadata_extraction/preprocess_blmicrosoft.py` | `main` |  13 minutes (Job 41421081) |
| 2 | `step_2_deduplication/deduplicate_blmicrosoft.py` | `main` | 1h 10 minutes (Job 41430091) |
| 3 | `step_3_llm_genre_classifier/genre_classifier_blmicrosoft.py` | `main` |  X minutes |
| 4 | `genre_classifier.py` | `tm62_llm_genre_classifier` |  10 minutes |
| 5 | `run_predictions.py` | `tm63_bert_masked_word_prediction` |   10 minutes |

---

## Step 1 — Preprocessing and Metadata Extraction

This is the entry point of the pipeline. It ingests the raw BL Microsoft dataset and performs initial preprocessing and metadata extraction, producing a structured intermediate file that is passed to subsequent steps.

```bash
cd step_1_preprocessing_and_metadata_extraction/
sbatch launch.sh
```

## Step 2 — Deduplication

Identifies and labels duplicate records from the filtered dataset. 

```bash
cd step_2_deduplication/
#Recreate conda env if needed (and replace <your-name>)
#conda env create -f env.yml --prefix /gpfs/scratch/bsc100/<your-name>/.conda/envs/dedupe-env

sbatch launch.sh
```

## Step 3 — LLM-based genre classification

Uses a large language model (i.e. `gemma4`) to label each record as fiction or non-fiction.

```bash
cd step_3_llm_genre_classifier/
#Recreate conda env if needed (and replace <your-name>)
#conda env create -f env.yml --prefix /gpfs/scratch/bsc100/<your-name>/.conda/envs/llm-genre-classification-env

sbatch launch.sh
```



## Step 4 — Data Filtering with query words

Filter dataset by different query words, with sentence-level output
including context sentences and masked versions.


```bash
bash data_filtering/launch_filter_blmicrosoft.sh
```

**To run manually:**
```bash
python data_filtering/filter_blmicrosoft.py
```

---

## Step 5 — Mask unrolling



---

## Step 6 — BERT Masked Word Prediction

Runs BERT-based masked word predictions over the dataset, using 3 different models.

**To run via launcher:**
```bash
bash bert_masked_word_prediction/launch.sh
```


---

## Repository Structure (relevant branches)

```
data-processing-code/
├── main
│   └── preprocess_and_metadata_extraction/
│       └── preprocess_blmicrosoft.py              # Step 1
│
├── tm_30_data_filtering
│   └── data_filtering/
│       ├── filter_blmicrosoft.py                   # Step 2
│       └── launch_filter_blmicrosoft.sh
│
├── tm31_deduplication_blmicrosoft
│   └── preprocess_and_metadata_extraction/
│       └── deduplicate_blmicrosoft.py              # Step 3
│
├── tm62_llm_genre_classifier
│   └── llm_genre_classifier_blmicrosoft/
│       ├── genre_classifier.py                     # Step 4
│       └── launch_genre_classifier.sh
│
└── tm63_bert_masked_word_prediction
    └── bert_masked_word_prediction/
        ├── run_predictions.py                      # Step 5
        └── launch.sh
```

