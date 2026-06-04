"""
Genre classifier using vLLM. Classifies book titles as fiction or non-fiction.
"""

import argparse
import gc
import os
import re
import random
from datetime import datetime

import pandas as pd
import torch
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

def _guided_decoding():
    return {"structured_outputs": StructuredOutputsParams(choice=["fiction", "non-fiction"])}

random.seed(42)

# ---------------------------------------------------------------------------
# Model registry — add entries here as you discover what's on the cluster
# ---------------------------------------------------------------------------
MODELS_BASE = "/gpfs/projects/bsc100/models"

MODEL_REGISTRY = {
    "llama-8b":       f"{MODELS_BASE}/meta-llama/Llama-3.1-8B-Instruct",
    "gemma-4-31b":    f"{MODELS_BASE}/gemma4/gemma-4-31B-it"
}

GEMMA4_MODEL_KEY = "gemma-4-31b"

DEFAULT_MODEL = "llama-8b"

# ---------------------------------------------------------------------------
# Model-specific LLM kwargs — tune per model as needed
# ---------------------------------------------------------------------------
LLM_KWARGS = {
    "llama-8b":    {"max_model_len": 4096},
    "gemma-4-31b": {"max_model_len": 4096, "gpu_memory_utilization": 0.90, "tensor_parallel_size": 4},
}


def resolve_model(model_arg: str) -> str:
    """Return full path: expand shorthand from registry, or pass through if already a path."""
    if model_arg in MODEL_REGISTRY:
        path = MODEL_REGISTRY[model_arg]
        print(f"Model shorthand '{model_arg}' → {path}")
        return path
    return model_arg  # assume it's already a full path


# ---------------------------------------------------------------------------
# Few-shot examples
# ---------------------------------------------------------------------------
FICTION_TITLES = [
    "The white doe of Rylstone; or The fate of the Nortons. A poem by Wordsworth, William",
    "Oliver Twist by Dickens, Charles",
    "Emma by Austen, Jane",
    "The Luck of Barry Lyndon, a romance of the last century by Thackeray, William Makepeace",
    "The Bride of Abydos. A Turkish tale by Byron, George Gordon Byron, Baron",
]

NONFICTION_TITLES = [
    "Ceylon in 1893, etc by Ferguson, John",
    "Memoirs of the principal events in the campaigns of North Holland and Egypt: together with a brief description of the Islands of Crete, Rhodes, Syracuse, etc by Maule, Francis",
    "The Campaigns in Virginia 1861-62. Reprinted from the 'Illustrated Naval and Military Magazine.' by Maguire, T. Miller (Thomas Miller)",
    "The Tourists' Handy Guide to Scotland ... Twelfth edition ... enlarged by Scotland",
    "Among the Goths and Vandals [On Sweden.] by Blaikie, John",
]

SYSTEM_PROMPT = """
    You are a helpful assistant that classifies books as either fiction or non-fiction based on their title.
    """

# ---------------------------------------------------------------------------
# Test data (used when --test flag is passed or no CSV is provided)
# ---------------------------------------------------------------------------
TEST_DATA = [
    {"title": "The Adventures of Huckleberry Finn", "author": "Twain, Mark"},
    {"title": "Pride and Prejudice", "author": "Austen, Jane"},
    {"title": "A Tale of Two Cities", "author": "Dickens, Charles"},
    {"title": "Travels in Arabia Deserta", "author": "Doughty, Charles M."},
    {"title": "The Voyage of the Beagle", "author": "Darwin, Charles"},
    {"title": "Confessions of an English Opium-Eater", "author": "De Quincey, Thomas"},
    {"title": "Sonnets from the Portuguese", "author": "Browning, Elizabeth Barrett"},
    {"title": "The History of the Decline and Fall of the Roman Empire", "author": "Gibbon, Edward"},
]


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------
def generate_user_prompt_zeroshot(fiction_titles, nonfiction_titles, target_title, target_author):
    return f"""Given a title and its author, return only the 'fiction' or 'non-fiction' label, without any additional explanation.

    Examples of genres that should be labeled as fiction are: poetry, novels, romances, plays, comedies, tragedies, recitations, short stories, tales, songs, operas, odes, poems.
    Examples of genres that should be labeled as non-fiction are: memoirs, biographies, autobiographies, essays, history books, travelogues, textbooks, guidebooks, scientific studies, philosophical treatises, and commentaries.

    Classify the following title as either "fiction" or "non-fiction": \n{target_title} by {target_author}.

    Label:
    """


def generate_user_prompt_fewshot(fiction_titles, nonfiction_titles, target_title, target_author):
    return f"""Given a title and its author, return only the 'fiction' or 'non-fiction' label, without any additional explanation.

    Examples of genres that should be labeled as fiction are: poetry, novels, romances, plays, comedies, tragedies, recitations, short stories, tales, songs, operas, odes, poems.
    Examples of genres that should be labeled as non-fiction are: memoirs, biographies, autobiographies, essays, history books, travelogues, textbooks, guidebooks, scientific studies, philosophical treatises, and commentaries.

    Below are five examples of fiction book titles:\n{chr(10).join(fiction_titles)}

    Below are five examples of non-fiction book titles:\n{chr(10).join(nonfiction_titles)}

    Classify the following title as either "fiction" or "non-fiction": \n{target_title} by {target_author}.

    Label:
    """


# ---------------------------------------------------------------------------
# Output normalizer — fallback for models that can't use guided_choice
# ---------------------------------------------------------------------------
def normalize_label(raw: str) -> str:
    """Map any LLM output to 'fiction', 'non-fiction', or 'unknown'.

    Handles edge cases like:
      - "fiction\nfiction\nnon-fiction"  → takes first line
      - "fiction_1814"                   → strips trailing non-alpha tokens
      - "nonfiction"                     → normalised to "non-fiction"
    """
    text = raw.strip().lower()
    # Take only the first line
    text = text.splitlines()[0].strip()
    # Keep only letters and hyphens (drops years, underscores, punctuation, etc.)
    text = re.sub(r"[^a-z\-].*", "", text)
    if text == "fiction":
        return "fiction"
    if text in ("non-fiction", "nonfiction"):
        return "non-fiction"
    return "unknown"


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------
def classify_genre(df, llm, system_prompt, user_prompt_col, output_col, batch_size=10, sampling_params=None):
    if sampling_params is None:
        sampling_params = {"temperature": 0, "top_p": 0.9, "repetition_penalty": 1}

    df = df.copy()
    sampling = SamplingParams(
        temperature=sampling_params["temperature"],
        top_p=sampling_params["top_p"],
        repetition_penalty=sampling_params["repetition_penalty"],
        max_tokens=10,
        seed=42,
        **_guided_decoding(),
    )

    df[output_col] = None
    total = len(df)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = df.iloc[start:end]

        messages = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": str(text)},
            ]
            for text in batch[user_prompt_col]
        ]

        outputs = llm.chat(messages=messages, sampling_params=sampling)

        for i, output in enumerate(outputs):
            raw = output.outputs[0].text.strip()
            df.at[batch.index[i], output_col] = normalize_label(raw)

        print(f"Processed rows {start + 1}-{end}/{total}")

    return df


def load_and_classify(model_path, model_key, df, output_col, prompt_fn, args):
    """Load a model, run classification, unload it, and return the updated DataFrame."""
    print(f"\nLoading model from: {model_path}")
    kwargs = LLM_KWARGS.get(model_key, {"max_model_len": 4096, "gpu_memory_utilization": 0.90})
    llm = LLM(model=model_path, **kwargs)

    sampling_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
    }

    df = classify_genre(
        df,
        llm=llm,
        system_prompt=SYSTEM_PROMPT,
        user_prompt_col="user_prompt",
        output_col=output_col,
        batch_size=args.batch_size,
        sampling_params=sampling_params,
    )

    print(f"Unloading model: {model_path}")
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    print("GPU memory released.")

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Classify book titles as fiction or non-fiction using vLLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Model shorthand (e.g. 'llama-8b', 'mistral-small') or full path. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="Print available model shorthands and exit.",
    )
    parser.add_argument("--input_csv", type=str, default=None,
                        help="Path to input CSV. If not provided, test data is used.")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Path to save the output CSV. Defaults to timestamped file in current dir.")
    parser.add_argument("--mode", type=str, choices=["zeroshot", "fewshot"], default="zeroshot",
                        help="Prompting strategy.")
    parser.add_argument("--batch_size", type=int, default=20,
                        help="Batch size for vLLM inference.")
    parser.add_argument("--slice_start", type=int, default=0,
                        help="Start row index (for slicing large CSVs).")
    parser.add_argument("--slice_end", type=int, default=None,
                        help="End row index (for slicing large CSVs).")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()

    # --- List models and exit ---
    if args.list_models:
        print("\nAvailable model shorthands:\n")
        for name, path in MODEL_REGISTRY.items():
            print(f"  {name:<20} → {path}")
        print()
        return

    primary_model_path = resolve_model(args.model)
    gemma4_model_path = resolve_model(GEMMA4_MODEL_KEY)

    # --- GPU info ---
    print(f"CUDA version : {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU count    : {torch.cuda.device_count()}")

    # --- Load data ---
    if args.input_csv:
        df = pd.read_csv(args.input_csv).iloc[args.slice_start:args.slice_end]
        print(f"Loaded {len(df)} rows from {args.input_csv}")
    else:
        print("No input CSV provided — using built-in test data.")
        df = pd.DataFrame(TEST_DATA)

    print(df.head())

    # --- Build prompts (shared by both models) ---
    prompt_fn = generate_user_prompt_fewshot if args.mode == "fewshot" else generate_user_prompt_zeroshot
    df["user_prompt"] = df.apply(
        lambda x: prompt_fn(FICTION_TITLES, NONFICTION_TITLES, x["title"], x["author"]),
        axis=1,
    )
    print(f"\nExample prompt:\n{df.iloc[0]['user_prompt']}")

    # --- Pass 1: primary model → genre_llama-8b ---
    print(f"\n{'='*60}")
    print(f"Pass 1 — primary model: {args.model}")
    print('='*60)
    df = load_and_classify(primary_model_path, args.model, df, "genre_llama-8b", prompt_fn, args)

    # --- Pass 2: Gemma-4 → genre_gemma4 ---
    print(f"\n{'='*60}")
    print(f"Pass 2 — Gemma-4: {GEMMA4_MODEL_KEY}")
    print('='*60)
    df = load_and_classify(gemma4_model_path, GEMMA4_MODEL_KEY, df, "genre_gemma4", prompt_fn, args)

    # --- Results ---
    print("\nResults:")
    print(df[["title", "author", "genre_llama-8b", "genre_gemma4"]].to_string(index=False))

    # --- Save ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_path = args.output_csv or f"genre_classified_{args.mode}_{timestamp}.csv"
    df.drop(columns=["user_prompt"]).to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
