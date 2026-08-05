# EEBO, ECCO and Evans Raw Dataset Preprocessing

This repository contains scripts for preprocessing historical XML text collections into:

- Per-book plain text files
- A metadata CSV file

The supported datasets are:

- **EEBO** (Early English Books Online, TCP)
- **ECCO** (Eighteenth Century Collections Online)
- **Evans** (Early American Imprints, TCP)

## Requirements

Create a Python virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the only required package:

```bash
pip install lxml
```

## Files

| File | Description |
|------|-------------|
| `preprocess_eebo.py` | Preprocesses the EEBO XML dataset |
| `preprocess_ecco.py` | Preprocesses the ECCO XML dataset |
| `preprocess_evans.py` | Preprocesses the Evans XML dataset |
| `launch_preprocess_eebo.sh` | Launcher for EEBO preprocessing script |

---

## EEBO

Preprocess the EEBO XML archive into per-book text files and a metadata CSV.

### Usage

```bash
sbatch launch_preprocess_eebo.sh
```

---

## ECCO

Preprocess the ECCO XML dataset into per-book text files and a metadata CSV. Since this is a very light and short computation it can be run from a transfer node (e.g. `transfer1`).

### Usage

```bash
source venv/bin/activate
python3 preprocess_ecco.py \
    --input_dir /gpfs/projects/bsc100/textmachine-data/ecco/p4/ecco_p4_released \
    --output_dir /gpfs/projects/bsc100/textmachine-data/preprocessed_data/output_ecco
```

---

## Evans

Preprocess the Evans (Early American Imprints, TCP) XML dataset into per-book text files and a metadata CSV. Since this is a very light and short computation it can be run from a transfer node (e.g. `transfer1`)

### Usage

```bash
source venv/bin/activate
python3 preprocess_evans.py \
    --input_zip /gpfs/projects/bsc100/textmachine-data/evans.zip \
    --output_dir /gpfs/projects/bsc100/textmachine-data/preprocessed_data/output_evans
```

---

## Output

Each preprocessing script generates:

- A directory called `txt` containing one plain text file per book.
- A CSV file containing metadata extracted from the XML records.
