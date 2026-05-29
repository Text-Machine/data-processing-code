#!/bin/bash
# Run from: /home/bsc/bsc204326/text_machine_processing/bert_masked_words_prediction


from huggingface_hub import snapshot_download
import os

models = [
    ("Livingwithmachines/bert_1760_1850",  "bert_textmachine/bert_1760_1850"),
    ("Livingwithmachines/bert_1890_1900",  "bert_textmachine/bert_1890_1900"),
    ("google-bert/bert-base-uncased",       "bert_textmachine/bert-base-uncased"),
]

base_dir = "/gpfs/projects/bsc100/models"

for repo_id, local_path in models:
    dest = os.path.join(base_dir, local_path)
    print(f"\n⬇ Downloading {repo_id} → {dest}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=dest,
        local_dir_use_symlinks=False,
    )
    print(f"✓ Done: {repo_id}")

print("\nAll models downloaded.")
