#!/usr/bin/env python3

"""
Run cross-book inference with an existing Qwen3.5-9B LoRA adapter.

Experiment:

    Pride and Prejudice paragraphs
            |
            v
    random 50 paragraphs
            |
            v
    generate one-sentence summaries
            |
            v
    Oliver Twist LoRA
            |
            v
    generate paragraphs

The final CSV contains:

    paragraph_index
    page_source
    summary
    reference_paragraph
    generated_output
"""


from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import re
import sys
import tarfile
from pathlib import Path

import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import PeftModel


# =====================================================================
# Configuration
# =====================================================================

MODEL_ID = "/gpfs/projects/bsc100/models/Qwen/Qwen3.5-9B/"

ADAPTER_DIR = os.environ.get(
    "ADAPTER_DIR",
    "/gpfs/scratch/bsc100/paolo/"
    "tm96_finetuning/data-processing-code/finetuning/outputs/",
)

BOOK_PATH = "1890_1899/1895_000144653.jsonl.gz"

INPUT_DIR = (
    "/gpfs/projects/bsc100/textmachine-data/downloaded_data_vm"
)

OUTPUT_CSV = os.environ.get(
    "OUTPUT_CSV",
    "/gpfs/scratch/bsc100/paolo/"
    "tm96_finetuning/data-processing-code/finetuning/"
    "outputs/pride_prejudice_lora_inference.csv",
)

N_PARAGRAPHS = 50
RANDOM_SEED = 42
MIN_CHARS = 100

MAX_NEW_TOKENS = 180

TEMPERATURE = 0.7
TOP_P = 0.9
TOP_K = 20
REPETITION_PENALTY = 1.05


# =====================================================================
# Book archive configuration
# =====================================================================

TARGET_TARS = [
    "OCR_text_c_1510_-_1699.tar.gz",
    "OCR_text_1700_-_1799.tar.gz",
    "OCR_text_1800_-_1809.tar.gz",
    "OCR_text_1810_-_1819.tar.gz",
    "OCR_text_1820_-_1829.tar.gz",
    "OCR_text_1830_-_1839.tar.gz",
    "OCR_text_1840_-_1849.tar.gz",
    "OCR_text_1850_-_1859.tar.gz",
    "OCR_text_1860_-_1869.tar.gz",
    "OCR_text_1870_-_1879.tar.gz",
    "OCR_text_1880_-_1889.tar.gz",
    "OCR_text_1890_-_1899.tar.gz",
]


# =====================================================================
# LoRA training conversation format
# =====================================================================

START_USER = "<start_of_turn>user\n"
START_MODEL = "<start_of_turn>model\n"
END_TURN = "<end_of_turn>\n"


LORA_INSTRUCTION = (
    "Write a paragraph, in the style of the source novel, "
    "based on this one-sentence summary:"
)


# =====================================================================
# Summary-generation prompt
#
# This is intentionally separate from the LoRA prompt.
# =====================================================================

SUMMARY_INSTRUCTION = (
    "Summarize the following paragraph in one sentence. "
    "The summary should describe the main events, actions, "
    "characters, and information needed to reconstruct the paragraph. "
    "Do not quote the paragraph."
)


# =====================================================================
# Archive helpers
# =====================================================================

def parse_year_range(tar_filename: str):
    match = re.search(
        r"(\d{4})_-_(\d{4})",
        tar_filename,
    )

    if match:
        return int(match.group(1)), int(match.group(2))

    return None


def resolve_tar_for_book_path(
    input_dir: Path,
    book_path: str,
) -> Path:

    decade_folder = book_path.split("/")[0]

    match = re.match(
        r"(\d{4})_(\d{4})",
        decade_folder,
    )

    if not match:
        raise ValueError(
            f"Couldn't parse year range from '{decade_folder}'"
        )

    start_year = int(match.group(1))

    exact_name = (
        f"OCR_text_{match.group(1)}_-_{match.group(2)}.tar.gz"
    )

    exact_path = input_dir / exact_name

    if exact_path.exists():
        return exact_path

    for name in TARGET_TARS:

        year_range = parse_year_range(name)

        if (
            year_range
            and year_range[0] <= start_year <= year_range[1]
            and (input_dir / name).exists()
        ):
            return input_dir / name

    for path in sorted(
        input_dir.glob("OCR_text_*.tar.gz")
    ):

        year_range = parse_year_range(path.name)

        if (
            year_range
            and year_range[0] <= start_year <= year_range[1]
        ):
            return path

    raise FileNotFoundError(
        f"No OCR archive found for year {start_year}"
    )


def read_jsonl_bytes_from_tar(
    tar_path: Path,
    book_path: str,
) -> bytes:

    target_basename = Path(book_path).name

    print(
        f"[info] Searching archive: {tar_path}",
        file=sys.stderr,
    )

    with tarfile.open(
        tar_path,
        mode="r|gz",
    ) as tar:

        for member in tar:

            if not member.isfile():
                continue

            if (
                member.name == book_path
                or member.name.endswith("/" + book_path)
                or Path(member.name).name == target_basename
            ):

                print(
                    f"[info] Found member: {member.name}",
                    file=sys.stderr,
                )

                fileobj = tar.extractfile(member)

                if fileobj is None:
                    continue

                return fileobj.read()

    raise FileNotFoundError(
        f"Book '{book_path}' not found in {tar_path}"
    )


def iter_jsonl_lines(raw_bytes: bytes):

    text = gzip.decompress(
        raw_bytes
    ).decode(
        "utf-8",
        errors="replace",
    )

    for line in text.splitlines():
        yield line


# =====================================================================
# Page parsing
# =====================================================================

def parse_pages(
    lines,
    text_field="text",
):

    pages = []

    for line_number, line in enumerate(
        lines,
        start=1,
    ):

        line = line.strip()

        if not line:
            continue

        try:
            record = json.loads(line)

        except json.JSONDecodeError as exc:

            print(
                f"[warn] malformed line {line_number}: {exc}",
                file=sys.stderr,
            )

            continue

        text = record.get(text_field)

        if text is None:

            text = next(
                (
                    value
                    for value in record.values()
                    if isinstance(value, str)
                    and value.strip()
                ),
                None,
            )

        if text:
            pages.append(text)

    return pages


# =====================================================================
# Paragraph extraction
# =====================================================================

def pages_to_paragraphs(
    pages,
):

    full_text = "\n\n".join(pages)

    full_text = (
        full_text
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )

    raw_paragraphs = re.split(
        r"\n\s*\n+",
        full_text,
    )

    paragraphs = []

    for paragraph in raw_paragraphs:

        cleaned = re.sub(
            r"\s+",
            " ",
            paragraph,
        ).strip()

        if cleaned:
            paragraphs.append(cleaned)

    return paragraphs


def load_book_paragraphs():

    input_dir = Path(INPUT_DIR)

    book_path = BOOK_PATH

    tar_path = resolve_tar_for_book_path(
        input_dir,
        book_path,
    )

    print(
        f"[info] Book archive: {tar_path}"
    )

    raw_bytes = read_jsonl_bytes_from_tar(
        tar_path,
        book_path,
    )

    print(
        f"[info] Compressed book bytes: "
        f"{len(raw_bytes):,}"
    )

    pages = parse_pages(
        iter_jsonl_lines(raw_bytes)
    )

    print(
        f"[info] Pages loaded: {len(pages):,}"
    )

    paragraphs = pages_to_paragraphs(
        pages
    )

    paragraphs = [
        paragraph
        for paragraph in paragraphs
        if len(paragraph) >= MIN_CHARS
    ]

    print(
        f"[info] Paragraphs after filtering: "
        f"{len(paragraphs):,}"
    )

    return paragraphs


# =====================================================================
# Random paragraph selection
# =====================================================================

def select_random_paragraphs(
    paragraphs,
):

    if len(paragraphs) < N_PARAGRAPHS:

        raise ValueError(
            f"Only {len(paragraphs)} paragraphs available; "
            f"cannot select {N_PARAGRAPHS}."
        )

    rng = random.Random(
        RANDOM_SEED
    )

    selected_indices = rng.sample(
        range(len(paragraphs)),
        N_PARAGRAPHS,
    )

    selected_indices.sort()

    selected = []

    for index in selected_indices:

        selected.append(
            {
                "paragraph_index": index,
                "reference_paragraph": paragraphs[index],
            }
        )

    return selected


# =====================================================================
# Load Qwen base model
# =====================================================================

def load_base_model():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="eager",
    )

    model.config.use_cache = True
    model.eval()

    return model, tokenizer


# =====================================================================
# Generic generation
# =====================================================================

def generate_text(
    model,
    tokenizer,
    prompt,
    max_new_tokens,
):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    input_device = next(
        model.parameters()
    ).device

    inputs = {
        key: value.to(input_device)
        for key, value in inputs.items()
    }

    input_length = inputs[
        "input_ids"
    ].shape[1]

    with torch.inference_mode():

        output = model.generate(
            **inputs,

            max_new_tokens=max_new_tokens,

            do_sample=True,

            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,

            repetition_penalty=REPETITION_PENALTY,

            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,

            use_cache=True,
        )

    generated_tokens = output[
        0
    ][input_length:]

    return tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()


# =====================================================================
# Generate summaries
# =====================================================================

def generate_summary(
    model,
    tokenizer,
    paragraph,
):

    prompt = (
        f"{START_USER}"
        f"{SUMMARY_INSTRUCTION}\n\n"
        f"{paragraph}"
        f"{END_TURN}"
        f"{START_MODEL}"
    )

    return generate_text(
        model,
        tokenizer,
        prompt,
        max_new_tokens=80,
    )


# =====================================================================
# Generate with LoRA
# =====================================================================

def generate_lora_paragraph(
    model,
    tokenizer,
    summary,
):

    prompt = (
        f"{START_USER}"
        f"{LORA_INSTRUCTION} {summary}"
        f"{END_TURN}"
        f"{START_MODEL}"
    )

    return generate_text(
        model,
        tokenizer,
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
    )


# =====================================================================
# Main
# =====================================================================

def main():

    print("=" * 70)
    print("Pride and Prejudice → LoRA inference")
    print("=" * 70)

    print(
        "Torch:",
        torch.__version__,
    )

    print(
        "CUDA:",
        torch.version.cuda,
    )

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    print(
        "Book:",
        BOOK_PATH,
    )

    print(
        "Random seed:",
        RANDOM_SEED,
    )

    print(
        "Paragraphs:",
        N_PARAGRAPHS,
    )

    # --------------------------------------------------------------
    # Extract book
    # --------------------------------------------------------------

    paragraphs = load_book_paragraphs()

    selected = select_random_paragraphs(
        paragraphs
    )

    print()
    print(
        f"[info] Selected {len(selected)} paragraphs"
    )

    # --------------------------------------------------------------
    # Load base Qwen
    # --------------------------------------------------------------

    print()
    print(
        "[info] Loading Qwen3.5-9B..."
    )

    base_model, tokenizer = load_base_model()

    # --------------------------------------------------------------
    # Generate summaries with base model
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("Generating summaries")
    print("=" * 70)

    for number, item in enumerate(
        selected,
        start=1,
    ):

        print(
            f"\n[{number}/{len(selected)}] "
            f"Paragraph {item['paragraph_index']}"
        )

        summary = generate_summary(
            base_model,
            tokenizer,
            item["reference_paragraph"],
        )

        item["summary"] = summary

        print("Summary:")
        print(summary)

    # --------------------------------------------------------------
    # Load LoRA
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("Loading LoRA adapter")
    print("=" * 70)

    adapter_config = Path(
        ADAPTER_DIR
    ) / "adapter_config.json"

    if not adapter_config.exists():

        raise FileNotFoundError(
            f"Could not find:\n{adapter_config}"
        )

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_DIR,
    )

    model.set_adapter("default")
    model.eval()

    print(
        "[info] LoRA adapter loaded"
    )

    # --------------------------------------------------------------
    # Generate Pride and Prejudice paragraphs
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("Generating paragraphs with LoRA")
    print("=" * 70)

    for number, item in enumerate(
        selected,
        start=1,
    ):

        print(
            f"\n[{number}/{len(selected)}] "
            f"Paragraph {item['paragraph_index']}"
        )

        generated = generate_lora_paragraph(
            model,
            tokenizer,
            item["summary"],
        )

        item["generated_output"] = generated

        print("Generated:")
        print(generated)

    # --------------------------------------------------------------
    # Save
    # --------------------------------------------------------------

    output_rows = []

    for item in selected:

        output_rows.append(
            {
                "paragraph_index": item[
                    "paragraph_index"
                ],
                "summary": item[
                    "summary"
                ],
                "reference_paragraph": item[
                    "reference_paragraph"
                ],
                "generated_output": item[
                    "generated_output"
                ],
            }
        )

    result_df = pd.DataFrame(
        output_rows
    )

    output_path = Path(
        OUTPUT_CSV
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        output_path,
        index=False,
    )

    print()
    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)

    print(
        "Output:",
        output_path,
    )

    print(
        "Rows:",
        len(result_df),
    )


if __name__ == "__main__":
    main()