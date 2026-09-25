#!/bin/bash
set -euo pipefail

ENV_DIR="/gpfs/scratch/bsc100/paolo/venv_mn5_qwen35_9b_v3"

echo "=============================================="
echo "Creating Qwen3.5-9B Python virtual environment"
echo "Environment: $ENV_DIR"
echo "=============================================="

# MN5 modules
module purge

module load mkl
module load intel
module load impi
module load hdf5
module load python/3.12
module load sqlite3/3.45.2
module load cuda/12.8


echo
echo "[INFO] Python:"
python3 --version
which python3

echo
echo "[INFO] CUDA:"
nvcc --version

# Avoid inheriting packages from the user's ~/.local
unset PYTHONPATH
export PYTHONNOUSERSITE=1

# Create virtual environment
if [[ -d "$ENV_DIR" ]]; then
    echo
    echo "[WARNING] $ENV_DIR already exists."
    echo "Delete it first if you want a completely clean environment:"
    echo "  rm -rf $ENV_DIR"
    exit 1
fi

python3 -m venv "$ENV_DIR"

# Activate
source "$ENV_DIR/bin/activate"

echo
echo "[INFO] Environment:"
which python
python --version
which pip

# Upgrade pip tooling
python -m pip install --upgrade pip setuptools wheel

echo
echo "[INFO] Installing packages..."

# Core packages from colleague's environment
pip install \
    "pandas==3.0.3" \
    "openai==2.44.0" \
    "pydantic==2.13.4" \
    "jinja2==3.1.6" \
    "pyyaml==6.0.3" \
    "tqdm==4.68.4" \
    "duckdb==1.5.4" \
    "pyarrow==24.0.0" \
    "accelerate"

# PyTorch
pip install \
    "torch==2.10.0" \
    "torchaudio==2.10.0" \
    "torchvision==0.25.0"

# vLLM
#pip install "vllm==0.19.0"

# vLLM 0.19.0's PyPI metadata still pins transformers<5, but Qwen3.5/Gemma4
# support actually requires transformers>=5.5.0 (see vllm-project/vllm#39216).
# Force the correct version after vLLM installs its default; pip will print
# a dependency-conflict warning here, which is expected and safe to ignore.
#pip install "transformers>=5.5.0"
pip install "transformers==5.17.0"

echo
echo "=============================================="
echo "Environment created successfully"
echo "=============================================="

echo
echo "[INFO] Python:"
python --version

echo
echo "[INFO] Key packages:"
pip list | grep -E 'torch|vllm|transformers|tokenizers|huggingface|pandas|pyarrow|duckdb'

echo
echo "Activate it with:"
echo
echo "  module purge"
echo "  module load mkl intel python/3.12 sqlite3/3.45.2 cuda/12.8"
echo "  unset PYTHONPATH"
echo "  export PYTHONNOUSERSITE=1"
echo "  source $ENV_DIR/bin/activate"