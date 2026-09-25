#!/usr/bin/env python3

import os
import argparse

import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import PeftModel


# ============================================================
# Configuration
# ============================================================

MODEL_ID = "/gpfs/projects/bsc100/models/Qwen/Qwen3.5-9B/"

ADAPTER_DIR = (
    "/gpfs/scratch/bsc100/paolo/"
    "tm96_finetuning/data-processing-code/finetuning/outputs/"
)

INPUT_CSV = (
    "/gpfs/scratch/bsc100/paolo/"
    "tm96_finetuning/data-processing-code/finetuning/"
    "outputs/results_prompts.csv"
)

OUTPUT_CSV = (
    "/gpfs/scratch/bsc100/paolo/"
    "tm96_finetuning/data-processing-code/finetuning/"
    "outputs/inference_results.csv"
)

MAX_NEW_TOKENS = 180

TEMPERATURE = 0.7
TOP_P = 0.9
TOP_K = 20
REPETITION_PENALTY = 1.05


# ============================================================
# Conversation format
# Must exactly match training
# ============================================================

START_USER = "<start_of_turn>user\n"
START_MODEL = "<start_of_turn>model\n"
END_TURN = "<end_of_turn>\n"


INSTRUCTION = (
    "Write a paragraph, in the style of the source novel, "
    "based on this one-sentence summary:"
)


# ============================================================
# Load model
# ============================================================

def load_model():

    print("=" * 70)
    print("Loading tokenizer")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("EOS token:", tokenizer.eos_token)
    print("EOS token ID:", tokenizer.eos_token_id)
    print("PAD token:", tokenizer.pad_token)
    print("PAD token ID:", tokenizer.pad_token_id)

    print()
    print("=" * 70)
    print("Loading base model")
    print("=" * 70)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="eager",
    )

    base_model.config.use_cache = True

    print("Base model loaded")
    print("GPU:", torch.cuda.get_device_name(0))

    print()
    print("=" * 70)
    print("Loading LoRA adapter")
    print("=" * 70)

    adapter_config = os.path.join(
        ADAPTER_DIR,
        "adapter_config.json",
    )

    if not os.path.exists(adapter_config):
        raise FileNotFoundError(
            f"Could not find adapter_config.json:\n"
            f"{adapter_config}\n\n"
            f"Check ADAPTER_DIR."
        )

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_DIR,
    )

    model.set_adapter("default")
    model.eval()

    print("LoRA adapter loaded")
    print("Active adapters:", model.active_adapters)

    return model, tokenizer


# ============================================================
# Prompt formatting
# ============================================================

def format_prompt(summary):

    summary = str(summary).strip()

    prompt = f"{INSTRUCTION} {summary}"

    return (
        f"{START_USER}"
        f"{prompt}"
        f"{END_TURN}"
        f"{START_MODEL}"
    )


# ============================================================
# Generate one paragraph
# ============================================================

def generate_paragraph(
    model,
    tokenizer,
    summary,
):

    formatted_prompt = format_prompt(summary)

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
    )

    input_device = next(model.parameters()).device

    inputs = {
        key: value.to(input_device)
        for key, value in inputs.items()
    }

    input_length = inputs["input_ids"].shape[1]

    with torch.inference_mode():

        output = model.generate(
            **inputs,

            max_new_tokens=MAX_NEW_TOKENS,

            do_sample=True,

            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,

            repetition_penalty=REPETITION_PENALTY,

            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,

            use_cache=True,
        )

    generated_tokens = output[0][input_length:]

    generated_text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return generated_text.strip()


# ============================================================
# Main inference over CSV
# ============================================================

def main():

    print("=" * 70)
    print("Qwen3.5-9B + LoRA batch inference")
    print("=" * 70)

    print("Torch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))

    print()
    print("Input:")
    print(INPUT_CSV)

    print()
    print("Output:")
    print(OUTPUT_CSV)

    # --------------------------------------------------------
    # Load CSV
    # --------------------------------------------------------

    df = pd.read_csv(INPUT_CSV)

    required_columns = {
        "prompt",
        "expected_output",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df.dropna(
        subset=["prompt", "expected_output"]
    ).copy()

    df["prompt"] = df["prompt"].astype(str)
    df["expected_output"] = df["expected_output"].astype(str)

    print()
    print("Rows:", len(df))

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model, tokenizer = load_model()

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    generated = []

    total = len(df)

    for i, row in df.iterrows():

        summary = row["prompt"]

        print()
        print(
            f"[{len(generated) + 1}/{total}] "
            f"Generating..."
        )

        print("Summary:")
        print(summary)

        try:

            paragraph = generate_paragraph(
                model=model,
                tokenizer=tokenizer,
                summary=summary,
            )

            generated.append(paragraph)

            print()
            print("Generated:")
            print(paragraph)

        except Exception as e:

            print(
                f"ERROR on row {i}: {e}"
            )

            generated.append("")

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    df["generated_output"] = generated

    # Keep only useful columns first
    result_columns = [
        "prompt",
        "expected_output",
        "generated_output",
    ]

    remaining_columns = [
        c for c in df.columns
        if c not in result_columns
    ]

    df = df[
        result_columns + remaining_columns
    ]

    df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print()
    print("=" * 70)
    print("Inference complete")
    print("=" * 70)

    print("Rows processed:", len(df))
    print("Output:", OUTPUT_CSV)


if __name__ == "__main__":
    main()