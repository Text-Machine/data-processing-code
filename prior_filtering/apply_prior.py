#!/usr/bin/env python3
"""
apply_prior_final.py — Score sentences against the BL Microsoft corpus prior.

Two modes:

  Bulk JSONL scoring (default):
      Reads a JSONL file, scores the sentence field of every record, and
      writes a new JSONL with these fields added per record:

          n_tokens_prior
              Number of tokens in the sentence (as seen by the tokenizer).

          sentence_mean_log_prior         (mu_d)
              Mean log-probability of the sentence's tokens under the prior.
              Low  → tokens are rarer than typical (noise, garbled OCR).
              High → tokens are very common.

          sentence_prior_dispersion       (sigma_d)
              Std of token prior-probabilities (scaled ×1000) within the
              sentence. Low → uniform token frequencies (repetition or
              gibberish). High → healthy mix of common and rare words.

          sentence_mean_log_prior_deviation
              Z-score of mu_d vs the corpus baseline:
                  (corpus_mean_log_prior - mu_d) / corpus_mean_log_prior_std
              Positive → sentence is below corpus average (noisier than
              typical). Negative → above average (cleaner than typical).

          sentence_prior_dispersion_deviation
              Z-score of sigma_d vs the corpus baseline:
                  (corpus_prior_dispersion - sigma_d) / corpus_prior_dispersion_std
              Positive → less spread than typical (repetitive/garbled).
              Negative → more spread than typical (healthy prose).

          abs_sentence_mean_log_prior_deviation
              |sentence_mean_log_prior_deviation|
              Useful for flagging outliers in either direction.

          abs_sentence_prior_dispersion_deviation
              |sentence_prior_dispersion_deviation|

  Single-string mode (--text "..."):
      Scores one string from the command line and prints the result.

      Example:
          python apply_prior_final.py --text "The French Revolution began in 1789."
          python apply_prior_final.py --text "buy cheap buy now buy cheap"
          python apply_prior_final.py --text "xq3 z9w m1x qz9 x0p l4k"

  Importable:
          from apply_prior_final import score_string
          result = score_string("The French Revolution began in 1789.")

Corpus baseline
---------------
The four baseline values below are loaded from the prior file, where they
were computed by build_prior_final.py over a reservoir sample of BL corpus
pages. They are the fixed reference for deviation scoring and are NEVER
recomputed from the dataset being scored.

    corpus_mean_log_prior       mean of mu_d    across reservoir pages
    corpus_mean_log_prior_std   std  of mu_d    across reservoir pages
    corpus_prior_dispersion     mean of sigma_d across reservoir pages
    corpus_prior_dispersion_std std  of sigma_d across reservoir pages
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
    "prior_filter/blmicrosoft/all_prior_tfdf_final.pt"
)

INPUT_JSONL_PATH = Path(
    "/gpfs/projects/bsc100/textmachine-data/preprocessed_data/"
    "consolidated_metadata_blmicrosoft/"
    "bl_microsoft_slaves_spacy_step_6.jsonl"
)

OUTPUT_JSONL_PATH = Path(
    "prior_filter/blmicrosoft/"
    "bl_microsoft_slaves_spacy_step_7_scored_final.jsonl"
)

TEXT_FIELD               = "sentence"
PROGRESS_EVERY_N_RECORDS = 20_000


# =============================================================================
# environment
# =============================================================================

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE",       "1")


# =============================================================================
# tokenizer
# =============================================================================

def load_tokenizer(tokenizer_name: str):
    if tokenizer_name == "whitespace_fallback":
        raise RuntimeError(
            "Prior was built with the whitespace fallback tokenizer, whose "
            "vocab mapping is not reproducible here. Rebuild the prior with "
            "the real GPT2TokenizerFast."
        )

    import transformers

    tok = transformers.GPT2TokenizerFast.from_pretrained(tokenizer_name)
    tok.model_max_length = int(1e12)

    test_ids = tok("the quick brown fox", add_special_tokens=False)["input_ids"]
    if tok.vocab_size < 1000 or len(test_ids) == 0:
        raise RuntimeError(
            f"Tokenizer looks broken (vocab_size={tok.vocab_size}, "
            f"test={len(test_ids)} ids). Check HF_HOME / offline cache."
        )

    print(f"Using GPT2TokenizerFast ('{tokenizer_name}'), "
          f"vocab_size={tok.vocab_size}.")
    return lambda text: tok(text, add_special_tokens=False)["input_ids"]


# =============================================================================
# prior cache
# =============================================================================

_PRIOR_CACHE: dict = {}


def load_prior(prior_path: Path) -> dict:
    cache_key = str(prior_path)
    if cache_key in _PRIOR_CACHE:
        return _PRIOR_CACHE[cache_key]

    print(f"Loading prior from {prior_path} ...")
    saved = torch.load(prior_path, map_location="cpu", weights_only=False)
    tok_fn = load_tokenizer(saved["tokenizer_name"])

    _PRIOR_CACHE[cache_key] = {
        "prior":                       saved["prior"],
        "tok_fn":                      tok_fn,
        "corpus_mean_log_prior":       saved["corpus_mean_log_prior"],
        "corpus_mean_log_prior_std":   saved["corpus_mean_log_prior_std"],
        "corpus_prior_dispersion":     saved["corpus_prior_dispersion"],
        "corpus_prior_dispersion_std": saved["corpus_prior_dispersion_std"],
        "n_pages":                     saved.get("n_pages", "?"),
        "n_books":                     saved.get("n_books", "?"),
        "tokenizer_name":              saved["tokenizer_name"],
        "vocab_size":                  saved["vocab_size"],
    }

    c = _PRIOR_CACHE[cache_key]
    print(f"  built from {c['n_pages']} pages / {c['n_books']} books")
    print(f"  tokenizer:  {c['tokenizer_name']}")
    print(f"  vocab_size: {c['vocab_size']}")
    print(f"  corpus_mean_log_prior:   "
          f"{c['corpus_mean_log_prior']:.4f} "
          f"± {c['corpus_mean_log_prior_std']:.4f}")
    print(f"  corpus_prior_dispersion: "
          f"{c['corpus_prior_dispersion']:.4f} "
          f"± {c['corpus_prior_dispersion_std']:.4f}")

    return _PRIOR_CACHE[cache_key]


# =============================================================================
# core scoring
# =============================================================================

def score_string(
    text: str,
    prior_path: Path = PRIOR_PATH,
) -> dict:
    """
    Score a single string against the corpus prior.

    Returns a dict with all seven output fields plus corpus_ref.
    Can be called repeatedly — the prior is loaded once and cached.

    Example:
        from apply_prior_final import score_string
        r = score_string("The French Revolution began in 1789.")
        print(r["sentence_mean_log_prior_deviation"])
    """
    c      = load_prior(prior_path)
    prior  = c["prior"]
    tok_fn = c["tok_fn"]

    corpus_mean_log_prior       = c["corpus_mean_log_prior"]
    corpus_mean_log_prior_std   = c["corpus_mean_log_prior_std"]
    corpus_prior_dispersion     = c["corpus_prior_dispersion"]
    corpus_prior_dispersion_std = c["corpus_prior_dispersion_std"]

    ids      = tok_fn(text)
    n_tokens = len(ids)

    if n_tokens == 0:
        nan = float("nan")
        return {
            "n_tokens_prior":                             0,
            "sentence_mean_log_prior":                    nan,
            "sentence_prior_dispersion":                  nan,
            "sentence_mean_log_prior_deviation":          nan,
            "sentence_prior_dispersion_deviation":        nan,
            "abs_sentence_mean_log_prior_deviation":      nan,
            "abs_sentence_prior_dispersion_deviation":    nan,
        }

    x  = torch.tensor(ids, dtype=torch.long)
    tp = prior[x]

    # --- raw sentence-level stats ---
    sentence_mean_log_prior  = tp.log().mean().item()
    sentence_prior_dispersion = (
        (tp * 1000).std().item() if n_tokens > 1 else 0.0
    )

    # --- z-scores vs corpus baseline ---
    sentence_mean_log_prior_deviation = (
        (corpus_mean_log_prior - sentence_mean_log_prior)
        / corpus_mean_log_prior_std
        if corpus_mean_log_prior_std > 0 else 0.0
    )
    sentence_prior_dispersion_deviation = (
        (corpus_prior_dispersion - sentence_prior_dispersion)
        / corpus_prior_dispersion_std
        if corpus_prior_dispersion_std > 0 else 0.0
    )

    abs_sentence_mean_log_prior_deviation   = abs(sentence_mean_log_prior_deviation)
    abs_sentence_prior_dispersion_deviation = abs(sentence_prior_dispersion_deviation)

    return {
        "n_tokens_prior":                             n_tokens,
        "sentence_mean_log_prior":                    sentence_mean_log_prior,
        "sentence_prior_dispersion":                  sentence_prior_dispersion,
        "sentence_mean_log_prior_deviation":          sentence_mean_log_prior_deviation,
        "sentence_prior_dispersion_deviation":        sentence_prior_dispersion_deviation,
        "abs_sentence_mean_log_prior_deviation":      abs_sentence_mean_log_prior_deviation,
        "abs_sentence_prior_dispersion_deviation":    abs_sentence_prior_dispersion_deviation,
        # corpus baseline included for reference / diagnostics
        "corpus_ref": {
            "corpus_mean_log_prior":       round(corpus_mean_log_prior,       4),
            "corpus_mean_log_prior_std":   round(corpus_mean_log_prior_std,   4),
            "corpus_prior_dispersion":     round(corpus_prior_dispersion,     4),
            "corpus_prior_dispersion_std": round(corpus_prior_dispersion_std, 4),
            "built_from": f"{c['n_pages']} pages / {c['n_books']} books",
        },
    }


def _print_score(text: str, result: dict):
    """Pretty-print a score_string() result to stdout."""
    ref = result["corpus_ref"]
    print()
    print("─" * 68)
    print(f"  text    : {text[:90]}{'...' if len(text) > 90 else ''}")
    print(f"  tokens  : {result['n_tokens_prior']}")
    print()
    print(f"  sentence_mean_log_prior      : "
          f"{result['sentence_mean_log_prior']:>9.4f}   "
          f"(corpus {ref['corpus_mean_log_prior']:.4f} "
          f"± {ref['corpus_mean_log_prior_std']:.4f})")
    print(f"  sentence_mean_log_prior_dev  : "
          f"{result['sentence_mean_log_prior_deviation']:>+9.4f}σ  "
          f"{'← rarer/noisier than typical' if result['sentence_mean_log_prior_deviation'] > 0 else '← typical or cleaner'}")
    print()
    print(f"  sentence_prior_dispersion    : "
          f"{result['sentence_prior_dispersion']:>9.4f}   "
          f"(corpus {ref['corpus_prior_dispersion']:.4f} "
          f"± {ref['corpus_prior_dispersion_std']:.4f})")
    print(f"  sentence_prior_disp_dev      : "
          f"{result['sentence_prior_dispersion_deviation']:>+9.4f}σ  "
          f"{'← more uniform/repetitive than typical' if result['sentence_prior_dispersion_deviation'] > 0 else '← typical spread'}")
    print()
    print(f"  prior built from: {ref['built_from']}")
    print("─" * 68)
    print()


# =============================================================================
# bulk JSONL scoring
# =============================================================================

def run_jsonl(input_path: Path, output_path: Path, prior_path: Path):
    print("=" * 70)
    print("apply_prior_final.py — bulk JSONL scoring")
    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print(f"Prior : {prior_path}")
    print("=" * 70)

    load_prior(prior_path)   # warm up cache and print summary

    output_path.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT_FIELDS = [
        "n_tokens_prior",
        "sentence_mean_log_prior",
        "sentence_prior_dispersion",
        "sentence_mean_log_prior_deviation",
        "sentence_prior_dispersion_deviation",
        "abs_sentence_mean_log_prior_deviation",
        "abs_sentence_prior_dispersion_deviation",
    ]

    n = 0
    n_empty = 0

    with (
        open(input_path,  "r", encoding="utf-8") as infile,
        open(output_path, "w", encoding="utf-8") as outfile,
    ):
        for line in infile:
            line = line.strip()
            if not line:
                continue

            rec  = json.loads(line)
            text = rec.get(TEXT_FIELD, "") or ""

            result = score_string(text, prior_path=prior_path)

            for field in OUTPUT_FIELDS:
                rec[field] = result[field]

            outfile.write(json.dumps(rec, ensure_ascii=False) + "\n")

            n += 1
            if result["n_tokens_prior"] == 0:
                n_empty += 1
            if n % PROGRESS_EVERY_N_RECORDS == 0:
                print(f"  ... {n:,} records scored")

    print()
    print("=" * 70)
    print(f"Records scored : {n:,}")
    print(f"Empty sentences: {n_empty:,}")
    print(f"Output         : {output_path}")
    print("=" * 70)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--text",   "-t", metavar="STRING", default=None,
                   help="Score a single string instead of the JSONL file.")
    p.add_argument("--input",  type=Path, default=INPUT_JSONL_PATH,
                   help=f"Input JSONL (default: {INPUT_JSONL_PATH})")
    p.add_argument("--output", type=Path, default=OUTPUT_JSONL_PATH,
                   help=f"Output JSONL (default: {OUTPUT_JSONL_PATH})")
    p.add_argument("--prior",  type=Path, default=PRIOR_PATH,
                   help=f"Prior .pt file (default: {PRIOR_PATH})")
    return p.parse_args()


def main():
    args = parse_args()
    if args.text is not None:
        result = score_string(args.text, prior_path=args.prior)
        _print_score(args.text, result)
    else:
        run_jsonl(
            input_path=args.input,
            output_path=args.output,
            prior_path=args.prior,
        )


if __name__ == "__main__":
    main()