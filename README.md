# Data Processing Code

This repository contains the data acquisition and processing pipelines used to build and prepare historical and literary text corpora for downstream NLP/HPC analysis as part of the [`Text Machine project`](https://www.schmidtsciences.org/humanities-and-ai-virtual-institute/#modal-text-machine-computing-literary-innovation). Text Machine is a two-year research project funded by the Schmidt Sciences Humanities and AI Virtual Institute that investigates moments of literary innovation and surprise across historical texts, specifically in relation to moments of technological change. This repo covers everything from raw dataset download to metadata extraction, deduplication, genre classification, filtering, and masked-word prediction.

## Repository Structure

```
data-processing-code/
├── download_raw_data_code/       # Scripts for downloading raw datasets and their metadata
├── pipeline_eebo_ecco_evans/     # Preprocessing pipeline for the EEBO, ECCO, and Evans TCP collections
├── pipeline_bl_microsoft/        # End-to-end processing pipeline for the BL Microsoft dataset
├── pipeline_hmd_lwm/             # Processing pipeline for the HMD/LWM dataset
├── LICENSE
└── README.md                     # This file
```

## Subdirectories

### `download_raw_data_code/`
Aggregates multiple historical and literary corpora from different sources. See [`download_raw_data_code/README.md`](download_raw_data_code/README.md) for a concise guide describing how each dataset is obtained, along with dataset version information.

### `pipeline_eebo_ecco_evans/`
Preprocesses the EEBO, ECCO, and Evans TCP XML collections into per-book plain text files and a metadata CSV (title, author, birth/death years, publication date, place, publisher, language, page count, word count, and source/output paths). Requires the raw datasets to have been downloaded first, as described in `download_raw_data_code/README.md` (section *"TCP Collections (EEBO, ECCO, Evans)"*). See [`pipeline_eebo_ecco_evans/README.md`](pipeline_eebo_ecco_evans/README.md) for full details.

### `pipeline_bl_microsoft/`
End-to-end pipeline for producing a fully processed metadata file for the BL Microsoft dataset. Must be run in the order of its numbered steps (preprocessing/metadata extraction → deduplication → LLM genre classification → data filtering → mask unrolling → BERT masked-word prediction). Requires the raw dataset to have been downloaded first, as described in `download_raw_data_code/README.md` (section *"2. BL Microsoft Collection"*). See [`pipeline_bl_microsoft/README.md`](pipeline_bl_microsoft/README.md) for full details.

### `pipeline_hmd_lwm/`
Processing pipeline for the HMD/LWM dataset. Requires the raw dataset to have been downloaded first, as described in `download_raw_data_code/README.md` (section *"3. Zenodo-hosted Datasets"*). See [`pipeline_hmd_lwm/README.md`](pipeline_hmd_lwm/README.md) for full details.

## Repository Structure & Branches

This repository's branches reflect different stages of the pipeline. Branches prefixed with `tm` are the current development feature branches. The `train` branch contains a BERT pretraining pipeline (data chunking, masking, Hugging Face Datasets integration, and Colab training notebooks). The `clean-data` branch builds on `train` with additional fixes (e.g. date/time token ordering) and adds support for GPT-2 training.


## Getting Started

1. Start with `download_raw_data_code/` to fetch the raw datasets you need.
2. Run the relevant pipeline (e.g. `pipeline_bl_microsoft/`) following the step order described in its README.
3. Refer to each subdirectory's own README for pipeline-specific requirements and configuration.

## License

See [LICENSE](LICENSE) for details.

## Acknowledgements

This work is supported by Schmidt Sciences, grant number G-25-69397.
