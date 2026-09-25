"""
preprocess_instruction_tuning.py

Prior step (separate, not shown here): extract_paragraphs.py reads
1830_1839/1838_000932097.jsonl.gz (a page-level OCR dump of a book, e.g.
Oliver Twist), groups text into paragraphs (splits not separated by 2+ line
breaks), and keeps the first 100 as a JSON list of strings:

    ["Among other public buildings in a certain town, ...", "...", ...]

This script: for each of those paragraphs, ask a frozen instruct LLM
(Qwen3.5-9B) to write a one-sentence summary of it. The output CSV has a
"prompt" column (the generated summary) and an "expected_output" column
(the original paragraph) -- the same shape instruction_tuning_adapter_gemma.py
expects as its INPUT_CSV, so the two scripts chain directly.
"""

from pathlib import Path
import json
import os
import re

import pandas as pd
import numpy as np
from datetime import datetime
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

np.random.seed(42)
random.seed(42)

import torch

print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.device_count())

RUN_ROOT = Path(os.environ["RUN_ROOT"])
OUT_DIR = RUN_ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Output of the prior-work extraction step: a JSON list of paragraph strings
# for a single book (e.g. Oliver Twist, first 100 paragraphs).
PARAGRAPHS_FILE = os.environ.get(
    "PARAGRAPHS_FILE",
    "/data/home/qs251108/text_machine_nilop/data/paragraphs/oliver_twist_paragraphs.json",
)
BOOK_NAME = os.environ.get("BOOK_NAME", "oliver_twist")

system_prompt = """
You are a helpful assistant who is familiar with literary fiction.
"""

user_prompt = """Your task is to summarize the following text in one sentence,
 removing the style and keeping only the core message, while at
 the same time trying to be as specific as possible. The subject,
 if missing, is the narrator. Return only the summarized text
 without any additional explanation. Just give one option and nothing else"""

with open(PARAGRAPHS_FILE, encoding="utf-8") as f:
    paragraphs = json.load(f)

rows_for_df = []
for chunk_id, paragraph in enumerate(paragraphs):
    rows_for_df.append({
        "book": BOOK_NAME,
        "chunk_id": chunk_id,
        "chunk": paragraph,
        "user_input": paragraph,
        "expected_output": paragraph,
        "system_prompt": system_prompt,
    })
resulting_df = pd.DataFrame(data=rows_for_df)

# Unlike the narrator-style-writing pipeline, every paragraph here needs a
# summary (there's no separate inference split at this stage) -- the
# train/eval split for fine-tuning happens downstream, inside
# instruction_tuning_adapter_gemma.py.
MODEL_ID = "/gpfs/projects/bsc100/models/Qwen/Qwen3.5-9B/"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto", device_map="auto")

THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

responses = []
print(len(resulting_df))
for i, row in tqdm(resulting_df.iterrows()):
    messages = [{"role": "system", "content": row["system_prompt"]},
                {"role": "user", "content": user_prompt + row["user_input"]}]
    # Qwen3.5 thinks by default (wraps reasoning in <think>...</think> before
    # the answer) -- enable_thinking=False turns that off so generation stays
    # short and we get a plain one-sentence summary directly.
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    outputs = model.generate(**inputs, max_new_tokens=128)
    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    # Defensive strip in case a stray <think> block slips through anyway.
    response = THINK_BLOCK_RE.sub("", response).strip()
    responses.append(response)
resulting_df['prompt'] = responses

outprompts = OUT_DIR / "results_prompts.csv"
resulting_df.to_csv(outprompts, index=False)
print("Wrote", outprompts)