#!/usr/bin/env python3
"""
Generate one-sentence summaries for extracted paragraphs using a frozen
Qwen3.5-9B model.

Input:
    JSON list of paragraph strings.

Output:
    CSV containing, among other fields:
      book
      chunk_id
      chunk
      user_input
      expected_output
      system_prompt
      prompt

Here "prompt" is the generated summary and "expected_output" is the
original paragraph. The CSV is consumed by the LoRA training or inference
stage.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant who is familiar with literary fiction."
)

DEFAULT_USER_PROMPT = """Your task is to summarize the following text in one sentence,
removing the style and keeping only the core message, while at
the same time trying to be as specific as possible. The subject,
if missing, is the narrator. Return only the summarized text
without any additional explanation. Just give one option and nothing else

"""


def generate_summary(
    model,
    tokenizer,
    paragraph: str,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": user_prompt + paragraph,
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )

    response = tokenizer.decode(
        outputs[0][input_len:],
        skip_special_tokens=True,
    )

    return THINK_BLOCK_RE.sub("", response).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--paragraphs-file",
        required=True,
        type=Path,
        help="JSON list produced by extract_paragraphs.py.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output CSV.",
    )
    parser.add_argument(
        "--book-name",
        required=True,
        help="Dataset/book identifier stored in the output CSV.",
    )
    parser.add_argument(
        "--model-id",
        default="/gpfs/projects/bsc100/models/Qwen/Qwen3.5-9B/",
        help="Frozen Qwen model.",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
    )
    parser.add_argument(
        "--user-prompt",
        default=DEFAULT_USER_PROMPT,
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this inference stage.")

    print("Torch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("GPU count:", torch.cuda.device_count())

    with open(args.paragraphs_file, encoding="utf-8") as handle:
        paragraphs = json.load(handle)

    if not isinstance(paragraphs, list):
        raise ValueError("Paragraphs file must contain a JSON list.")

    rows = [
        {
            "book": args.book_name,
            "chunk_id": chunk_id,
            "chunk": paragraph,
            "user_input": paragraph,
            "expected_output": paragraph,
            "system_prompt": args.system_prompt,
        }
        for chunk_id, paragraph in enumerate(paragraphs)
    ]

    df = pd.DataFrame(rows)

    print("Paragraphs:", len(df))

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype="auto",
        device_map="auto",
    )
    model.eval()

    responses = []

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=f"Summarising {args.book_name}",
    ):
        response = generate_summary(
            model=model,
            tokenizer=tokenizer,
            paragraph=row["user_input"],
            system_prompt=row["system_prompt"],
            user_prompt=args.user_prompt,
            max_new_tokens=args.max_new_tokens,
        )
        responses.append(response)

    df["prompt"] = responses

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
