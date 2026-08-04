# BL Microsoft Dataset — Metadata Processing Pipeline

This document describes the end-to-end pipeline for producing a fully processed metadata file for the BL Microsoft dataset. The pipeline must be executed in the order described below.

> **Note:** This requires having downloaded the relevant raw dataset as explained in `data-processing-code/download_raw_data_code/README.md` (see section called  `2. BL Microsoft Collection`).

---

## Pipeline Outputs

The pipeline produces two main outputs:

1. Final Metadata File (Steps 1-3)

Output file:

`blmicrosoft_final_metadata.csv`


2. Query-specific Prediction Files (Steps 4–6)

Output file:

`blmicrosoft_final_{query}.jsonl`

For example:

`blmicrosoft_final_machine.jsonl`
`blmicrosoft_final_slave.jsonl`


## Pipeline Overview

Every script file needs to be launched with the associated launcher from its respective directory. Every directory also contains a `README.md` documenting the script and associate launcher (TODO).

| Step | Script | Execution time | 
|------|--------|--------| 
| 1 | `step_1_preprocessing_and_metadata_extraction/preprocess_blmicrosoft.py`  |  13 minutes (Job 41421081) |
| 2 | `step_2_deduplication/deduplicate_blmicrosoft.py` | 1h 10 minutes (Job 41430091) |
| 3 | `step_3_genre_classifier/genre_classifier_blmicrosoft.py`|  12 minutes (Job 41432292)|
| 4 | `step_4_data_filtering/filter_data_blmicrosoft.py`|  17 minutes for regex version (Job 41442898) or 52 minutes for spacy version (Job 42040439)
| 5 | `step_5_mask_unrolling/unroll_masks.py`  |   1 minute (Job 41448457)|
| 6 | `step_6_masked_word_prediction/run_predictions.py` |   18 minutes (Job 42314469) |
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

Filter dataset by different query words, with sentence-level output including context sentences and masked versions.


```bash
#Recreate conda env if needed (and replace <your-name>)
#conda env create -f env.yml --prefix /gpfs/scratch/bsc100/<your-name>/.conda/envs/llm-data-filtering-env
cd step_4_data_filtering/
sbatch launch.sh
```

## Note — Updating the Query Words

If you want to modify the query words used for sentence extraction and masked-word prediction (Steps 4-6), you need to update the query word configuration used by the scripts, by modifying the following line inside `pipeline_bl_microsoft/query_word_config.sh`:

```bash
QUERY_WORDS=(slave slaves machine machines mornings nights morning night)
```



---

## Step 5 — Mask unrolling

Unrolls rows that contain multiple instances of the query word. As a result, each row in the metadata file only contains one instance of the special [MASK] token.



```bash
cd step_5_mask_unrolling/
sbatch launch.sh
```

---

## Step 6 — BERT Masked Word Prediction

Runs masked word prediction over the dataset, using 3 different BERT models. This step is only executed for the query words `slave`, `slaves`, `machine` and `machines`.

```bash
#Recreate conda env if needed (and replace <your-name>)
#conda env create -f env.yml --prefix /gpfs/scratch/bsc100/<your-name>/.conda/envs/bert-masked-word-prediction-env
cd step_6_bert_masked_word_prediction/
sbatch launch.sh
```


## Final note on execution time
The execution times listed in the Pipeline Overview section are based on the query words `slave`, `slaves`, `machine`, `machines`, `mornings`, `nights`, `morning` and `night`. Other query words may require more or less time, but this cannot be predicted upfront. Update the time allocation of your job accordingly.

```bash
#SBATCH --time=00:10:00
```

