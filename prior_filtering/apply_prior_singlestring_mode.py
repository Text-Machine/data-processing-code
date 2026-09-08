#!/usr/bin/env python3

"""
apply_prior_v3.py —  Step 2 of the prior-filtering pipeline.

Score every sentence in a JSONL file using the same corpus-baseline
logic as single-string mode.

For every JSONL record:

    record["sentence"]
        ↓
    tokenize using the tokenizer stored with the prior
        ↓
    look up token probabilities in the prior
        ↓
    calculate mu_d and sigma_d
        ↓
    compare against the corpus baseline stored in the prior
        ↓
    calculate d_mu and d_sigma

The output preserves every original field and adds:

    mu_d
    sigma_d
    d_mu
    d_sigma

Unlike the previous bulk implementation, this script does NOT:

    - rank records against each other
    - calculate rank_dist_mu / rank_dist_sigma
    - calculate mu_outlier / sigma_outlier
    - force a TARGET_KEEP_FRACTION
    - calculate a dataset-relative discard decision

This is therefore equivalent to running score_string() independently
for every sentence in the JSONL file.
"""

import argparse
import json
import os
from pathlib import Path

import torch


# =============================================================================
# CONFIG
# =============================================================================

PRIOR_PATH = Path(
    "prior_filter/blmicrosoft/all_prior_tfdf_v3.pt"
)

INPUT_JSONL_PATH = Path(
    "/gpfs/projects/bsc100/textmachine-data/preprocessed_data/"
    "consolidated_metadata_blmicrosoft/"
    "bl_microsoft_slaves_spacy_step_6.jsonl"
)

OUTPUT_JSONL_PATH = Path(
    "prior_filter/blmicrosoft/"
    "bl_microsoft_slaves_spacy_step_7_scored_singlestring_mode.jsonl"
)

TEXT_FIELD = "sentence"

PROGRESS_EVERY_N_RECORDS = 20_000


# =============================================================================
# Environment
# =============================================================================

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


# =============================================================================
# Tokenizer
# =============================================================================

def load_tokenizer(tokenizer_name: str):
    """
    Load the exact tokenizer specified when the prior was built.
    """

    if tokenizer_name == "whitespace_fallback":
        raise RuntimeError(
            "Prior was built with the whitespace fallback tokenizer, "
            "whose vocab mapping is not reproducible here. "
            "Rebuild the prior with the real GPT2TokenizerFast "
            "before running apply_prior_v3.py."
        )

    import transformers

    tok = transformers.GPT2TokenizerFast.from_pretrained(
        tokenizer_name
    )

    tok.model_max_length = int(1e12)

    test_ids = tok(
        "the quick brown fox",
        add_special_tokens=False,
    )["input_ids"]

    if tok.vocab_size < 1000 or len(test_ids) == 0:
        raise RuntimeError(
            f"Tokenizer loaded but looks broken "
            f"(vocab_size={tok.vocab_size}, "
            f"test={len(test_ids)} ids). "
            f"Check HF_HOME / offline cache."
        )

    print(
        f"Using GPT2TokenizerFast ('{tokenizer_name}'), "
        f"vocab_size={tok.vocab_size}."
    )

    return lambda text: tok(
        text,
        add_special_tokens=False,
    )["input_ids"]


# =============================================================================
# Prior cache
# =============================================================================

_PRIOR_CACHE = {}


def load_prior(prior_path: Path):
    """
    Load the prior and associated corpus baseline statistics.

    Cached so that repeated sentence scoring does not reload
    the .pt file.
    """

    cache_key = str(prior_path)

    if cache_key not in _PRIOR_CACHE:

        print(f"Loading prior from {prior_path} ...")

        saved = torch.load(
            prior_path,
            map_location="cpu",
            weights_only=False,
        )

        tok_fn = load_tokenizer(
            saved["tokenizer_name"]
        )

        _PRIOR_CACHE[cache_key] = {
            "prior": saved["prior"],
            "tok_fn": tok_fn,

            # Corpus baseline
            "mu_avg": saved["mu_avg"],
            "mu_std": saved["mu_std"],
            "sigma_avg": saved["sigma_avg"],
            "sigma_std": saved["sigma_std"],

            # Metadata
            "n_chunks": saved.get(
                "n_chunks",
                saved.get("n_pages", "?"),
            ),
            "n_books": saved.get(
                "n_books",
                "?",
            ),
            "tokenizer_name": saved["tokenizer_name"],
            "vocab_size": saved["vocab_size"],
        }

        print(
            f"  prior built from "
            f"{saved.get('n_chunks', saved.get('n_pages', '?'))} "
            f"chunks / "
            f"{saved.get('n_books', '?')} books"
        )

        print(
            f"  tokenizer: "
            f"{saved['tokenizer_name']}"
        )

        print(
            f"  vocab_size: "
            f"{saved['vocab_size']}"
        )

        print(
            f"  corpus baseline:"
        )

        print(
            f"    mu    = "
            f"{saved['mu_avg']:.4f} "
            f"± {saved['mu_std']:.4f}"
        )

        print(
            f"    sigma = "
            f"{saved['sigma_avg']:.4f} "
            f"± {saved['sigma_std']:.4f}"
        )

    return _PRIOR_CACHE[cache_key]


# =============================================================================
# Single-string scoring
# =============================================================================

def score_string(
    text: str,
    prior_path: Path = PRIOR_PATH,
) -> dict:
    """
    Score one string against the corpus prior.

    This is the same logic as the previous single-string mode.

    Returns:

        n_tokens
        mu_d
        sigma_d
        d_mu
        d_sigma
    """

    c = load_prior(prior_path)

    prior = c["prior"]
    tok_fn = c["tok_fn"]

    mu_avg = c["mu_avg"]
    mu_std = c["mu_std"]

    sigma_avg = c["sigma_avg"]
    sigma_std = c["sigma_std"]

    # ------------------------------------------------------------
    # Tokenize
    # ------------------------------------------------------------

    ids = tok_fn(text)

    n_tokens = len(ids)

    if n_tokens == 0:
        return {
            "n_tokens": 0,
            "mu_d": None,
            "sigma_d": None,
            "d_mu": None,
            "d_sigma": None,
        }

    # ------------------------------------------------------------
    # Look up token prior probabilities
    # ------------------------------------------------------------

    x = torch.tensor(
        ids,
        dtype=torch.long,
    )

    tp = prior[x]

    # ------------------------------------------------------------
    # Calculate sentence-level statistics
    # ------------------------------------------------------------

    mu_d = tp.log().mean().item()

    sigma_d = (
        (tp * 1000).std().item()
        if n_tokens > 1
        else 0.0
    )

    # ------------------------------------------------------------
    # Compare against corpus baseline
    # ------------------------------------------------------------

    d_mu = (
        (mu_avg - mu_d) / mu_std
        if mu_std > 0
        else 0.0
    )

    d_sigma = (
        (sigma_avg - sigma_d) / sigma_std
        if sigma_std > 0
        else 0.0
    )

    return {
        "n_tokens": n_tokens,
        "mu_d": mu_d,
        "sigma_d": sigma_d,
        "d_mu": d_mu,
        "d_sigma": d_sigma,
    }


# =============================================================================
# JSONL processing
# =============================================================================

def run_jsonl(
    input_path: Path = INPUT_JSONL_PATH,
    output_path: Path = OUTPUT_JSONL_PATH,
    prior_path: Path = PRIOR_PATH,
):
    """
    Read sentences from JSONL and score each one using score_string().
    """

    print("=" * 70)
    print("Single-string prior scoring over JSONL")
    print("=" * 70)

    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print(f"Prior : {prior_path}")
    print()

    # Load prior once.
    load_prior(prior_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    n = 0
    n_empty = 0

    with open(
        input_path,
        "r",
        encoding="utf-8",
    ) as infile, open(
        output_path,
        "w",
        encoding="utf-8",
    ) as outfile:

        for line in infile:

            line = line.strip()

            if not line:
                continue

            rec = json.loads(line)

            # --------------------------------------------------------
            # Get sentence
            # --------------------------------------------------------

            text = rec.get(
                TEXT_FIELD,
                "",
            ) or ""

            # --------------------------------------------------------
            # Score exactly as single-string mode
            # --------------------------------------------------------

            result = score_string(
                text,
                prior_path=prior_path,
            )

            # --------------------------------------------------------
            # Add scores to original record
            # --------------------------------------------------------

            rec["n_tokens_prior"] = result["n_tokens"]

            rec["mu_d"] = result["mu_d"]

            rec["sigma_d"] = result["sigma_d"]

            rec["d_mu"] = result["d_mu"]

            rec["d_sigma"] = result["d_sigma"]

            outfile.write(
                json.dumps(
                    rec,
                    ensure_ascii=False,
                ) + "\n"
            )

            n += 1

            if result["n_tokens"] == 0:
                n_empty += 1

            if n % PROGRESS_EVERY_N_RECORDS == 0:
                print(
                    f"  ... {n:,} records scored"
                )

    print()
    print("=" * 70)
    print("Finished")
    print("=" * 70)

    print(
        f"Records scored : {n:,}"
    )

    print(
        f"Empty sentences: {n_empty:,}"
    )

    print(
        f"Output         : {output_path}"
    )

    print("=" * 70)


# =============================================================================
# Command-line interface
# =============================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Score every sentence in a JSONL file "
            "using the single-string corpus-baseline prior."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_JSONL_PATH,
        help=(
            f"Input JSONL "
            f"(default: {INPUT_JSONL_PATH})"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_JSONL_PATH,
        help=(
            f"Output JSONL "
            f"(default: {OUTPUT_JSONL_PATH})"
        ),
    )

    parser.add_argument(
        "--prior",
        type=Path,
        default=PRIOR_PATH,
        help=(
            f"Prior .pt file "
            f"(default: {PRIOR_PATH})"
        ),
    )

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():

    args = parse_args()

    run_jsonl(
        input_path=args.input,
        output_path=args.output,
        prior_path=args.prior,
    )


if __name__ == "__main__":
    main()