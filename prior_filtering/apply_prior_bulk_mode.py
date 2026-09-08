"""
apply_prior_no_chunk.py  —  Step 2 of the prior-filtering pipeline.

Two modes:

  Bulk dataset scoring (python apply_prior.py):
      Reads a JSONL file, scores every record's sentence against the prior,
      and writes a new JSONL with 7 additional fields per record:
          mu_d, sigma_d, rank_dist_mu, rank_dist_sigma,
          mu_outlier, sigma_outlier, discard

  Single-string inference (python apply_prior.py --text "..."):
      Scores one string from the command line and prints a human-readable
      breakdown of how far it deviates from the corpus baseline.

      Example:
          python apply_prior.py --text "The French Revolution began in 1789."
          python apply_prior.py --text "buy cheap buy now buy cheap"
          python apply_prior.py --text "xq3 z9w m1x qz9 x0p l4k"

  It can also be imported and used interactively:
          from apply_prior import score_string
          result = score_string("The French Revolution began in 1789.")
          print(result["interpretation"])

Scoring approach
----------------
Bulk mode uses a rank-based filter matching the original notebook
(s3_prior_filter.ipynb): mu_d and sigma_d are ranked across the full
population, and records whose rank falls outside a central band of width 2r
(resolved by bisection to hit TARGET_KEEP_FRACTION) are discarded. Both
tails are penalised symmetrically, matching the paper's Fmu = Fsigma
constraint.

Single-string mode uses z-score deviation from the corpus baseline statistics
(mu_avg, sigma_avg) saved in the prior file. Rank-based scoring is not
meaningful for a single string in isolation.

"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

# =============================================================================
# CONFIG  —  used by bulk dataset scoring (main)
# =============================================================================

PRIOR_PATH = Path("prior_filter/blmicrosoft/all_prior_tfdf_v3.pt")

INPUT_JSONL_PATHS = [
    Path(
        "/gpfs/projects/bsc100/textmachine-data/preprocessed_data/"
        "consolidated_metadata_blmicrosoft/bl_microsoft_slaves_spacy_step_6.jsonl"
    ),
]

OUTPUT_JSONL_PATH = Path(
    "prior_filter/blmicrosoft/bl_microsoft_slaves_spacy_step_7_scored_v3.jsonl"
)

TARGET_KEEP_FRACTION    = 0.5      # matches the notebook's "select 0.5 data"
TEXT_FIELD              = "sentence"
PROGRESS_EVERY_N_RECORDS = 20_000


# =============================================================================
# tokenizer
# =============================================================================

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE",       "1")


def load_tokenizer(tokenizer_name: str):
    if tokenizer_name == "whitespace_fallback":
        raise RuntimeError(
            "Prior was built with the whitespace fallback tokenizer, whose "
            "vocab mapping is not reproducible here. Rebuild the prior with "
            "the real GPT2TokenizerFast before running apply_prior.py."
        )

    import transformers
    tok = transformers.GPT2TokenizerFast.from_pretrained(tokenizer_name)
    tok.model_max_length = int(1e12)

    test_ids = tok("the quick brown fox")["input_ids"]
    if tok.vocab_size < 1000 or len(test_ids) == 0:
        raise RuntimeError(
            f"Tokenizer loaded but looks broken (vocab_size={tok.vocab_size}, "
            f"test={len(test_ids)} ids). Check HF_HOME / offline cache."
        )

    print(f"Using GPT2TokenizerFast ('{tokenizer_name}'), "
          f"vocab_size={tok.vocab_size}.")
    return lambda text: tok(text, add_special_tokens=False)["input_ids"]


# =============================================================================
# scoring  —  no chunking: each text scored as a single unit
# =============================================================================

def compute_mu_sigma(
    tok_fn,
    prior: torch.Tensor,
    text: str,
    chunk_size: int | None = None,   # accepted but ignored in this variant
) -> tuple[float, float]:
    """
    Tokenize text and return (mu_d, sigma_d) over all tokens at once.

    chunk_size is accepted for API compatibility with apply_prior_chunked.py
    but is ignored — no splitting is performed regardless of text length.
    """
    ids = tok_fn(text)
    if not ids:
        return float("nan"), float("nan")

    x       = torch.tensor(ids, dtype=torch.long)
    tp      = prior[x]
    mu_d    = tp.log().mean().item()
    sigma_d = (tp * 1000).std().item() if len(ids) > 1 else 0.0
    return mu_d, sigma_d


# =============================================================================
# single-string inference
# =============================================================================

# Module-level cache so repeated score_string() calls don't reload from disk.
_PRIOR_CACHE: dict = {}


def score_string(
    text: str,
    prior_path: Path = PRIOR_PATH,
) -> dict:
    """
    Score a single string against the pre-computed prior.

    Returns a dict with:
        text            the input (truncated to 100 chars in the output)
        n_tokens        number of tokens in the text
        n_chunks        number of 512-token chunks scored
        mu_d            mean log-prior per token (averaged across chunks)
        sigma_d         std of (prior * 1000) per token (averaged across chunks)
        d_mu            z-score deviation of mu_d from corpus baseline
                        positive = below corpus avg = rarer tokens = noisier
        d_sigma         z-score deviation of sigma_d from corpus baseline
                        positive = below corpus avg = more uniform = repetitive
        corpus_ref      the baseline stats the prior was built from
        interpretation  plain-English summary

    Deviations are z-scores relative to the corpus baseline stats saved in
    the prior file. They are NOT rank-based (that requires a full population).
    As a rough guide: |d| < 1 is typical, |d| > 2 is unusual, |d| > 3 is an
    outlier by corpus standards.
    """
    cache_key = str(prior_path)
    if cache_key not in _PRIOR_CACHE:
        saved = torch.load(prior_path, map_location="cpu", weights_only=False)
        tok_fn = load_tokenizer(saved["tokenizer_name"])
        _PRIOR_CACHE[cache_key] = {
            "prior":      saved["prior"],
            "tok_fn":     tok_fn,
            "chunk_size": saved.get("chunk_size"),   # None for older priors
            "mu_avg":     saved["mu_avg"],
            "mu_std":     saved["mu_std"],
            "sigma_avg":  saved["sigma_avg"],
            "sigma_std":  saved["sigma_std"],
            "n_chunks":   saved.get("n_chunks", saved.get("n_pages", "?")),
            "n_books":    saved.get("n_books", "?"),
        }

    c          = _PRIOR_CACHE[cache_key]
    prior      = c["prior"]
    tok_fn     = c["tok_fn"]
    chunk_size = c["chunk_size"]
    mu_avg     = c["mu_avg"]
    mu_std     = c["mu_std"]
    sigma_avg  = c["sigma_avg"]
    sigma_std  = c["sigma_std"]

    ids      = tok_fn(text)
    n_tokens = len(ids)

    mu_d, sigma_d = compute_mu_sigma(tok_fn, prior, text)

    d_mu    = (mu_avg    - mu_d)    / mu_std    if mu_std    > 0 else 0.0
    d_sigma = (sigma_avg - sigma_d) / sigma_std if sigma_std > 0 else 0.0

    return {
        "text":        text[:100] + "..." if len(text) > 100 else text,
        "n_tokens":    n_tokens,
        "mu_d":        round(mu_d,    4),
        "sigma_d":     round(sigma_d, 4),
        "d_mu":        round(d_mu,    4),
        "d_sigma":     round(d_sigma, 4),
        "corpus_ref": {
            "mu_avg":     round(mu_avg,    4),
            "mu_std":     round(mu_std,    4),
            "sigma_avg":  round(sigma_avg, 4),
            "sigma_std":  round(sigma_std, 4),
            "built_from": f"{c['n_chunks']} chunks / {c['n_books']} books",
        },
        "interpretation": (
            f"mu_d={mu_d:.3f}  "
            f"({abs(d_mu):.2f}σ {'BELOW' if d_mu > 0 else 'above'} "
            f"corpus avg {mu_avg:.3f})  "
            f"| sigma_d={sigma_d:.3f}  "
            f"({abs(d_sigma):.2f}σ {'BELOW' if d_sigma > 0 else 'above'} "
            f"corpus avg {sigma_avg:.3f})"
        ),
    }


def _print_score(result: dict):
    """Pretty-print a score_string() result to stdout."""
    print()
    print("─" * 64)
    print(f"  text     : {result['text']}")
    print(f"  tokens   : {result['n_tokens']}  (scored as one unit, no chunking)")
    print()
    print(f"  mu_d     : {result['mu_d']:>9.4f}   "
          f"(corpus avg {result['corpus_ref']['mu_avg']:.4f} "
          f"± {result['corpus_ref']['mu_std']:.4f})")
    print(f"  d_mu     : {result['d_mu']:>+9.4f}σ  "
          f"{'← rarer/noisier than typical' if result['d_mu'] > 0 else '← typical or cleaner'}")
    print()
    print(f"  sigma_d  : {result['sigma_d']:>9.4f}   "
          f"(corpus avg {result['corpus_ref']['sigma_avg']:.4f} "
          f"± {result['corpus_ref']['sigma_std']:.4f})")
    print(f"  d_sigma  : {result['d_sigma']:>+9.4f}σ  "
          f"{'← more uniform/repetitive than typical' if result['d_sigma'] > 0 else '← typical spread'}")
    print()
    print(f"  prior built from: {result['corpus_ref']['built_from']}")
    print("─" * 64)
    print()


# =============================================================================
# rank-based outlier detection  (bulk mode)
# =============================================================================

def rank_fraction(values: torch.Tensor) -> torch.Tensor:
    n     = len(values)
    ranks = values.argsort(descending=False).argsort(descending=False)
    return ranks.float() / max(n - 1, 1)


def find_band_radius(
    dist_mu: torch.Tensor,
    dist_sigma: torch.Tensor,
    target: float,
    tol: float = 1e-6,
) -> float:
    lo, hi = 0.0, 0.5

    def keep_rate(r):
        return ((dist_mu <= r) & (dist_sigma <= r)).float().mean().item()

    if keep_rate(hi) < target:
        return hi

    while hi - lo > tol:
        mid = (lo + hi) / 2
        (hi if keep_rate(mid) >= target else lo).__class__   # no-op
        if keep_rate(mid) >= target:
            hi = mid
        else:
            lo = mid

    return hi


# =============================================================================
# bulk dataset scoring
# =============================================================================

def iter_lines(paths):
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line


def run_bulk(prior_path: Path = PRIOR_PATH):
    print(f"Loading prior from {prior_path} ...")
    saved          = torch.load(prior_path, map_location="cpu", weights_only=False)
    prior          = saved["prior"]
    tokenizer_name = saved["tokenizer_name"]
    chunk_size     = saved.get("chunk_size")
    print(
        f"  built from {saved.get('n_chunks', saved.get('n_pages', '?'))} chunks / "
        f"{saved.get('n_books', '?')} books | "
        f"chunk_size={chunk_size} | vocab_size={saved['vocab_size']}"
    )
    print(
        f"  corpus baseline: "
        f"mu_avg={saved['mu_avg']:.4f} ± {saved['mu_std']:.4f} | "
        f"sigma_avg={saved['sigma_avg']:.4f} ± {saved['sigma_std']:.4f}"
    )

    tok_fn = load_tokenizer(tokenizer_name)

    # Pass 1: score every record
    print("\nPass 1/2: scoring records ...")
    mus, sigmas = [], []
    n = 0
    for line in iter_lines(INPUT_JSONL_PATHS):
        rec  = json.loads(line)
        text = rec.get(TEXT_FIELD, "") or ""
        mu_d, sigma_d = compute_mu_sigma(tok_fn, prior, text, chunk_size)
        mus.append(mu_d)
        sigmas.append(sigma_d)
        n += 1
        if n % PROGRESS_EVERY_N_RECORDS == 0:
            print(f"  ... {n:,} records scored")

    print(f"Scored {n:,} record(s).")
    if n < 2:
        raise SystemExit("Need at least 2 records for rank-based scoring.")

    mus_t    = torch.tensor(mus)
    sigmas_t = torch.tensor(sigmas)

    mu_rank_frac    = rank_fraction(mus_t)
    sigma_rank_frac = rank_fraction(sigmas_t)
    rank_dist_mu    = (mu_rank_frac    - 0.5).abs()
    rank_dist_sigma = (sigma_rank_frac - 0.5).abs()

    r      = find_band_radius(rank_dist_mu, rank_dist_sigma, TARGET_KEEP_FRACTION)
    mu_out = rank_dist_mu    > r
    sg_out = rank_dist_sigma > r
    discard = mu_out | sg_out

    n_kept = (~discard).sum().item()
    print(f"Band radius r={r:.4f} → keeping {n_kept:,}/{n:,} ({n_kept/n:.1%})")

    # Pass 2: write output
    print("\nPass 2/2: writing scored records ...")
    OUTPUT_JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_JSONL_PATH, "w", encoding="utf-8") as out:
        for i, line in enumerate(iter_lines(INPUT_JSONL_PATHS)):
            rec                    = json.loads(line)
            rec["mu_d"]            = mus_t[i].item()
            rec["sigma_d"]         = sigmas_t[i].item()
            rec["rank_dist_mu"]    = rank_dist_mu[i].item()
            rec["rank_dist_sigma"] = rank_dist_sigma[i].item()
            rec["mu_outlier"]      = bool(mu_out[i].item())
            rec["sigma_outlier"]   = bool(sg_out[i].item())
            rec["discard"]         = bool(discard[i].item())
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (i + 1) % PROGRESS_EVERY_N_RECORDS == 0:
                print(f"  ... {i + 1:,} records written")

    print(f"\nWrote {n:,} records → {OUTPUT_JSONL_PATH}")


# =============================================================================
# entry point
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--text", "-t",
        metavar="STRING",
        default=None,
        help=(
            "Score a single string and print the result. "
            "If omitted, runs bulk dataset scoring over INPUT_JSONL_PATHS."
        ),
    )
    p.add_argument(
        "--prior",
        metavar="PATH",
        type=Path,
        default=PRIOR_PATH,
        help=f"Path to the prior .pt file (default: {PRIOR_PATH}).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.text is not None:
        # Single-string mode
        result = score_string(args.text, prior_path=args.prior)
        _print_score(result)
    else:
        # Bulk mode
        run_bulk(prior_path=args.prior)


if __name__ == "__main__":
    main()