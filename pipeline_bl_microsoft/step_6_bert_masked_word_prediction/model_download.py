#!/usr/bin/env python3
# Run from glogin4 or alogin4 node, after loading
# /gpfs/scratch/bsc100/<your-name>/.conda/envs/bert-masked-word-prediction-env
#
#   python3 download_bert_models.py
#
# Safe to re-run: snapshot_download compares existing local files against
# the remote repo (size/ETag) and only fetches what's missing or changed,
# so already-downloaded models are left untouched.

from huggingface_hub import snapshot_download
import os

models = [
    ("Livingwithmachines/bert_1760_1850",  "bert_textmachine/bert_1760_1850"),
    ("Livingwithmachines/bert_1890_1900",  "bert_textmachine/bert_1890_1900"),
    ("google-bert/bert-base-uncased",       "bert_textmachine/bert-base-uncased"),
    ("Livingwithmachines/bert_1760_1900",  "bert_textmachine/bert_1760_1900"),
]

base_dir = "/gpfs/projects/bsc100/models"

for repo_id, local_path in models:
    dest = os.path.join(base_dir, local_path)
    print(f"\n⬇ Checking/downloading {repo_id} → {dest}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=dest,
        local_dir_use_symlinks=False,
    )
    print(f"✓ Done: {repo_id}")

