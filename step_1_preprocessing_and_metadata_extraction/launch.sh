#!/bin/sh
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --time=00:40:00
#SBATCH -A bsc100
#SBATCH -q gp_debug

INPUT_DIR="/gpfs/projects/bsc100/textmachine-data/downloaded_data_vm"
OUTPUT_CSV="/gpfs/projects/bsc100/textmachine-data/preprocessed_data/consolidated_metadata/metadata_blmicrosoft.csv"

cd $SLURM_SUBMIT_DIR

python3 preprocess_blmicrosoft.py --input-dir "$INPUT_DIR" --output "$OUTPUT_CSV"