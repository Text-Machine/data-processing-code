"""

STEP 2 of the workflow: load the prior built by build_prior.py, score
each sentence-level record's `sentence` field against it, and add:

    mu_d, sigma_d, d_mu, d_sigma, mu_outlier, sigma_outlier, discard

to every input dict, preserving all of its original fields

The actual reference notebook does NOT discard only the "low tail" of mu_d
and sigma_d. It:
  1. Ranks every document's mu_d and sigma_d SEPARATELY (rank 0..N-1, i.e.
     percentile position in the population being filtered).
  2. Searches for the smallest radius r around the population MEDIAN rank
     (0.5) such that keeping only documents whose mu-rank AND sigma-rank
     both fall inside [0.5-r, 0.5+r] retains just over the target keep
     fraction (0.5 in the notebook).
  3. Discards everything outside that joint central band -- i.e. documents
     that are unusually extreme in EITHER direction on EITHER metric, not
     just documents with low mu_d/sigma_d.

Field mapping used here:
    d_mu / d_sigma      = |rank_fraction - 0.5|, i.e. how far this record's
                          mu_d / sigma_d sits from the population median,
                          in rank-space (0 = dead center, up to 0.5 = most
                          extreme)
    mu_outlier          = d_mu > r      (r resolved once per population)
    sigma_outlier       = d_sigma > r
    discard             = mu_outlier OR sigma_outlier

"""

import json
import os
from pathlib import Path

import torch

# =============================================================================
# CONFIG
# =============================================================================

PRIOR_PATH = Path("prior_filter/blmicrosoft/all_prior_tfdf.pt")

INPUT_JSONL_PATHS = [
    Path(
        "/gpfs/projects/bsc100/textmachine-data/preprocessed_data/"
        "consolidated_metadata/bl_microsoft_slaves_spacy_step_6.jsonl"
    ),
]

OUTPUT_JSONL_PATH = Path(
    "prior_filter/blmicrosoft/bl_microsoft_slaves_spacy_step_7_scored.jsonl"
)

TARGET_KEEP_FRACTION = 0.5  # matches the notebook's "select 0.5 data" setting
TEXT_FIELD = "sentence"     # which field to score

PROGRESS_EVERY_N_RECORDS = 20000


# =============================================================================
# tokenizer (must match whatever s2_build_prior.py used to build the prior)
# =============================================================================

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def load_tokenizer(tokenizer_name: str):
    if tokenizer_name == "whitespace_fallback":
        # If the prior was built with the whitespace fallback (no HF cache
        # available at build time), scoring here MUST use the exact same
        # vocab mapping, which isn't persisted by that fallback. In
        # practice: rebuild the prior with the real GPT2 tokenizer first.
        raise RuntimeError(
            "Prior was built with the whitespace fallback tokenizer, whose "
            "vocab mapping isn't reproducible here. Rebuild the prior with "
            "the real GPT2TokenizerFast (ensure the offline HF cache is "
            "populated) before running s3_apply_prior.py."
        )

    import transformers

    tok = transformers.GPT2TokenizerFast.from_pretrained(tokenizer_name)
    tok.model_max_length = int(1e12)  # silence the harmless truncation warning

    # same silent-failure guard as s2_build_prior.py: verify the offline
    # cache actually loaded a working tokenizer before trusting it
    test_ids = tok("the quick brown fox")["input_ids"]
    if tok.vocab_size < 1000 or len(test_ids) == 0:
        raise RuntimeError(
            f"Tokenizer loaded but looks broken (vocab_size={tok.vocab_size}, "
            f"test tokenization returned {len(test_ids)} ids). This must match "
            f"the tokenizer used to build the prior -- check the offline HF "
            f"cache / HF_HOME on this node."
        )

    print(f"Using GPT2TokenizerFast ('{tokenizer_name}') to match the saved prior, "
          f"vocab_size={tok.vocab_size}.")
    return lambda text: tok(text)["input_ids"]


# =============================================================================
# scoring
# =============================================================================

def compute_mu_sigma(tok_fn, prior: torch.Tensor, text: str):
    ids = tok_fn(text)
    if not ids:
        return float("nan"), float("nan")
    x = torch.tensor(ids, dtype=torch.long)
    token_priors = prior[x]
    mu_d = token_priors.log().mean().item()
    sigma_d = (token_priors * 1000).std().item()
    return mu_d, sigma_d


def rank_fraction(values: torch.Tensor) -> torch.Tensor:
    """Dense rank (0..N-1) converted to a 0..1 fraction, ties broken by order."""
    n = len(values)
    ranks = values.argsort(descending=False).argsort(descending=False)
    return ranks.float() / max(n - 1, 1)


def find_band_radius(dist_from_median_mu: torch.Tensor,
                      dist_from_median_sigma: torch.Tensor,
                      target_keep_fraction: float,
                      tol: float = 1e-6) -> float:
    """
    Bisection search for the smallest r such that keeping records with
    BOTH dist_from_median_mu <= r AND dist_from_median_sigma <= r retains
    at least target_keep_fraction of the population.

    Equivalent in spirit to the notebook's greedy decimal-digit search over
    r, but implemented via bisection since keep-rate(r) is monotonically
    non-decreasing in r.
    """
    lo, hi = 0.0, 0.5

    def keep_rate(r):
        mask = (dist_from_median_mu <= r) & (dist_from_median_sigma <= r)
        return mask.float().mean().item()

    if keep_rate(hi) < target_keep_fraction:
        # even the widest possible band can't hit the target (shouldn't
        # normally happen since r=0.5 keeps everyone) -- fall back to "keep all"
        return hi

    while hi - lo > tol:
        mid = (lo + hi) / 2
        if keep_rate(mid) >= target_keep_fraction:
            hi = mid
        else:
            lo = mid
    return hi


def iter_lines(paths):
    """Yields raw (non-empty, stripped) JSONL lines across all input files, in order."""
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line


def main():
    print(f"Loading prior from {PRIOR_PATH} ...")
    saved = torch.load(PRIOR_PATH)
    prior = saved["prior"]
    tokenizer_name = saved["tokenizer_name"]
    print(f"  prior built from {saved.get('n_pages', '?')} pages across "
          f"{saved.get('n_books', '?')} books, vocab_size={saved['vocab_size']}")

    tok_fn = load_tokenizer(tokenizer_name)

    # ---------------------------------------------------------------
    # Pass 1: stream once, compute mu_d/sigma_d per record, keep only
    # two small float lists in memory (not the parsed dicts themselves)
    # ---------------------------------------------------------------
    print("Pass 1/2: scoring records ...")
    mus, sigmas = [], []
    n = 0
    for line in iter_lines(INPUT_JSONL_PATHS):
        rec = json.loads(line)
        text = rec.get(TEXT_FIELD, "") or ""
        mu_d, sigma_d = compute_mu_sigma(tok_fn, prior, text)
        mus.append(mu_d)
        sigmas.append(sigma_d)
        n += 1
        if n % PROGRESS_EVERY_N_RECORDS == 0:
            print(f"  ... scored {n} records")

    print(f"Scored {n} sentence-level record(s) total.")
    if n < 2:
        raise SystemExit("Need at least 2 records to compute rank-based outlier stats.")

    mus = torch.tensor(mus)
    sigmas = torch.tensor(sigmas)

    # rank-fraction (percentile position, 0..1) within this population
    mu_rank_frac = rank_fraction(mus)
    sigma_rank_frac = rank_fraction(sigmas)

    # distance from the population median, in rank-space
    d_mu = (mu_rank_frac - 0.5).abs()
    d_sigma = (sigma_rank_frac - 0.5).abs()

    r = find_band_radius(d_mu, d_sigma, TARGET_KEEP_FRACTION)
    print(f"Resolved central-band radius r={r:.4f} "
          f"(targeting keep-fraction={TARGET_KEEP_FRACTION})")

    mu_outlier = d_mu > r
    sigma_outlier = d_sigma > r
    discard = mu_outlier | sigma_outlier

    n_kept = (~discard).sum().item()
    print(f"Keeping {n_kept}/{n} records ({n_kept/n:.1%})")

    # ---------------------------------------------------------------
    # Pass 2: re-stream the SAME files in the SAME order, inject the 7
    # fields per record using the index, write straight to disk
    # ---------------------------------------------------------------
    print("Pass 2/2: writing scored records ...")
    OUTPUT_JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSONL_PATH, "w", encoding="utf-8") as out:
        for i, line in enumerate(iter_lines(INPUT_JSONL_PATHS)):
            rec = json.loads(line)
            rec["mu_d"] = mus[i].item()
            rec["sigma_d"] = sigmas[i].item()
            rec["d_mu"] = d_mu[i].item()
            rec["d_sigma"] = d_sigma[i].item()
            rec["mu_outlier"] = bool(mu_outlier[i].item())
            rec["sigma_outlier"] = bool(sigma_outlier[i].item())
            rec["discard"] = bool(discard[i].item())
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (i + 1) % PROGRESS_EVERY_N_RECORDS == 0:
                print(f"  ... wrote {i + 1} records")

    print(f"Wrote {n} scored records to {OUTPUT_JSONL_PATH}")


if __name__ == "__main__":
    main()
