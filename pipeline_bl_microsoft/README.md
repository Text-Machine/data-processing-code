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

Specify the query words used for sentence extraction and masked-word prediction (to be used for Steps 4-6). To do this, you need to update the query word configuration used by the scripts, by modifying the following line inside `pipeline_bl_microsoft/query_word_config.sh`:

```bash
QUERY_WORDS=(slave slaves machine machines mornings nights morning night)
```

Filter dataset by different query words, with sentence-level output including context sentences and masked versions.

```bash
#Recreate conda env if needed (and replace <your-name>)
#conda env create -f env.yml --prefix /gpfs/scratch/bsc100/<your-name>/.conda/envs/llm-data-filtering-env
cd step_4_data_filtering/
sbatch launch.sh
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

Runs masked word prediction over the dataset, using 3 different BERT models.

Two versions of this step's script are available, depending on how many query words you're running relative to how many GPUs you have available.

### `run_predictions.py`

Assigns one query word's file to one GPU. If you request N GPUs and have N query words, each GPU processes exactly one word in parallel end to end. GPU count is automatically capped down to the number of matched query-word files. If you have more query words than GPUs, GPUs pick up the next word dynamically as soon as they finish their current one — not a static assignment. Use this by default. It's the simpler of the two, and is exactly as fast as the sharded version whenever query word count >= GPU count.

```bash
#Recreate conda env if needed (and replace <your-name>)
#conda env create -f env.yml --prefix /gpfs/scratch/bsc100/<your-name>/.conda/envs/bert-masked-word-prediction-env
cd step_6_bert_masked_word_prediction/
sbatch launch.sh
```

### `run_predictions_sharded.py`

If you have fewer query words than GPUs available on a single node, this version increases parallelism by splitting each query word's file into byte-range chunks 
(shards) before processing, rather than assigning one whole file per GPU:

`shards_per_file = ceil(BERT_MAX_GPUS / n_words)`

So with 1 query word and 4 GPUs, the file gets split 4 ways; with 2 words and 4 GPUs, each word's file gets split into 2 shards (4 tasks total, one per GPU); with 4+ words and 4 GPUs, shards_per_file comes out to 1 and it behaves identically to the normal script. This gives up to an approximate Nx speedup on the actual row-processing time.

Don't expect a blmicrosoft_final_{query}.jsonl produced by the sharded script to be byte-identical to the one produced by the normal script for the same query word. This comes from ordinary floating-point nondeterminism across different GPU hardware, not from a difference in logic between the two scripts.

```bash
cd step_6_bert_masked_word_prediction/
sbatch launch_sharded.sh
```

---

## Final note on execution time
The execution times listed in the Pipeline Overview section are based on the query words `slave`, `slaves`, `machine`, `machines`, `mornings`, `nights`, `morning` and `night`. Other query words may require more or less time, but this cannot be predicted upfront. Update the time allocation of your job accordingly.

If you are running step 6, you can estimate approximately 1 minute of processing time for every 30 MB of input data generated by step 5 (e.g. in our previous executions a 826 MB file generated by step 5 took 28 minutes of processing during step 6, while a 3 GB file took 1 hour 40 minutes).

```bash
#SBATCH --time=02:00:00
```

## Final note on GPU allocation

The number of GPUs requested in the launch file for Step 6 should be adjusted according to the number of query words being processed, and depends on which script you're using.

**Normal script (`run_predictions.py`)** — one GPU per query word, up to a maximum of 4:

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

**Sharded script (`run_predictions_sharded.py`)** — request 2 or 4 GPUs and set `BERT_MAX_GPUS` in `launch_sharded.sh` to the same number:

```bash
# any number of query words, want to use all available GPUs
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=80
# ...and in launch_sharded.sh:
export BERT_MAX_GPUS="4"
```

If you're unsure which script to use: if query word count >= GPU count, use the normal script — there's no benefit to sharding. If query word count < GPU count and you want to use all the GPUs of the node you have requested rather than leave them idle, use the sharded script, and double-check `BERT_MAX_GPUS` matches `--gres=gpu:N` before submitting.