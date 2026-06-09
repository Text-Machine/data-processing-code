#!/bin/sh
#SBATCH -n 1
#SBATCH -c 20
#SBATCH --time=01:10:00
#SBATCH -A bsc100
#SBATCH -q gp_debug

INPUT_CSV="/gpfs/projects/bsc100/textmachine-data/preprocessed_data/consolidated_metadata/metadata_blmicrosoft.csv"
OUTPUT_CSV="/gpfs/projects/bsc100/textmachine-data/preprocessed_data/consolidated_metadata/metadata_blmicrosoft_step_2.csv"

ENV_PATH="/gpfs/scratch/bsc100/paolo/.conda/envs/dedupe-env"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PATH"

cd $SLURM_SUBMIT_DIR

python3 deduplicate_blmicrosoft.py --input "$INPUT_CSV" --output "$OUTPUT_CSV"
