#!/usr/bin/env python3
"""
build_prior_final.py — Build the TF x DF token prior over the BL Microsoft corpus.

This version uses each processed page as a single document. There is NO
512-token chunking.

Required corpus filters (all three enforced in filter_metadata):
    - best_version == 1
    - main_language == "English"
    - date in [1800, 1899]

Document definition
-------------------
Each valid page is treated as one document for TF/DF and QA statistics.

    page tokens: [t0, t1, ..., t999]
    → one document containing all tokens

Long pages are NOT truncated or split. Short pages that pass MIN_PAGE_CHARS
are also kept as complete documents.

The TF/DF prior therefore represents:
    TF = total token frequency across all pages
    DF = number of pages containing each token (once per page, regardless
         of how many times the token repeats within that page)

QA reservoir and corpus baseline statistics
-------------------------------------------
While streaming through the corpus to build the prior, a random sample of
pages is collected into a QA reservoir (default: 5,000 pages) using
reservoir sampling. Every page has an equal probability of being selected
regardless of the order it is encountered or the total corpus size.

Once the prior is fully built, each reservoir page is scored against it to
compute two statistics:

    mu_d    (sentence_mean_log_prior)
        = mean(log(prior[t])) over all tokens t in the page
        Low  → tokens are rarer than typical (noise, garbled OCR)
        High → tokens are very common (possible repetitive content)

    sigma_d (sentence_prior_dispersion)
        = std(prior[t] * 1000) over all tokens t in the page
        Low  → token frequencies are uniform (repetitive or garbled)
        High → healthy mix of common and rare words (normal prose)

The mean and std of mu_d and sigma_d across the reservoir become the
corpus baseline:

    corpus_mean_log_prior          mean of mu_d    across reservoir pages
    corpus_mean_log_prior_std      std  of mu_d    across reservoir pages
    corpus_prior_dispersion        mean of sigma_d across reservoir pages
    corpus_prior_dispersion_std    std  of sigma_d across reservoir pages

These four numbers are saved into the prior file. apply_prior.py uses them
as a fixed reference baseline when scoring sentences: it never recomputes
them from the dataset being scored.

Why the reservoir and not the full corpus?
    To score a page you need the prior (you look up prior[t] for each
    token). But the prior does not exist until all pages have been seen.
    A second full pass over 8.8M pages would roughly double the runtime.
    5,000 reservoir pages are more than enough for stable mean/std
    estimates (the values converge well before that size).

Output
------
A single .pt file containing:
    prior                       float32 tensor [vocab_size]
    vocab_size                  int
    tokenizer_name              str
    n_pages                     int
    n_books                     int
    corpus_mean_log_prior       float   mean of mu_d    across reservoir
    corpus_mean_log_prior_std   float   std  of mu_d    across reservoir
    corpus_prior_dispersion     float   mean of sigma_d across reservoir
    corpus_prior_dispersion_std float   std  of sigma_d across reservoir
"""

import gzip
import io
import json
import os
import random
import re
import tarfile
from pathlib import Path

import pandas as pd
import torch


# =============================================================================
# CONFIG
# =============================================================================

METADATA_CSV = Path(
    "/gpfs/projects/bsc100/textmachine-data/preprocessed_data/"
    "consolidated_metadata_blmicrosoft/metadata_blmicrosoft_step_3.csv"
)

DATA_ROOT = Path(
    "/gpfs/projects/bsc100/textmachine-data/downloaded_data_vm"
)

DECADE_TO_TAR = {
    "1800_1809": "OCR_text_1800_-_1809.tar.gz",
    "1810_1819": "OCR_text_1810_-_1819.tar.gz",
    "1820_1829": "OCR_text_1820_-_1829.tar.gz",
    "1830_1839": "OCR_text_1830_-_1839.tar.gz",
    "1840_1849": "OCR_text_1840_-_1849.tar.gz",
    "1850_1859": "OCR_text_1850_-_1859.tar.gz",
    "1860_1869": "OCR_text_1860_-_1869.tar.gz",
    "1870_1879": "OCR_text_1870_-_1879.tar.gz",
    "1880_1889": "OCR_text_1880_-_1889.tar.gz",
    "1890_1899": "OCR_text_1890_-_1899.tar.gz",
}

# --- subset selection -------------------------------------------------------
N_BOOKS      = None   # cap total books (random sample). None = use all
RANDOM_STATE = 42
BOOK_IDS     = None   # explicit list of book_ids to include, or None

# --- page filtering ---------------------------------------------------------
MAX_PAGES_PER_BOOK = None
SKIP_EMPTY_PAGES   = True
MIN_PAGE_CHARS     = 50

# --- tokenization -----------------------------------------------------------
TOKENIZER_NAME   = "gpt2-large"
TOKEN_BATCH_SIZE = 64

# --- output -----------------------------------------------------------------
OUTPUT_PRIOR_PATH = Path(
    "/gpfs/projects/bsc100/textmachine-data/preprocessed_data/prior_filter_blmicrosoft/prior_blmicrosoft.pt"
)

# --- smoothing / QA reservoir -----------------------------------------------
SMOOTHING_EPS          = 1e-6   # Laplace: keeps prior > 0 for all tokens
QA_SAMPLE_SIZE         = 5000   # reservoir pages for corpus baseline stats
PROGRESS_EVERY_N_BOOKS = 500


# =============================================================================
# metadata filtering
# =============================================================================

def filter_metadata(df: pd.DataFrame) -> pd.DataFrame:
    n_start = len(df)

    df = df[df["best_version"] == 1]
    print(f"  after best_version==1:        {len(df):>7,} / {n_start:,}")

    df = df[df["main_language"] == "English"]
    print(f"  after main_language==English: {len(df):>7,}")

    df = df.copy()
    df["_year"] = pd.to_numeric(df["date"], errors="coerce")
    n_bad = df["_year"].isna().sum()
    if n_bad:
        print(f"  [warn] {n_bad} row(s) with unparseable date dropped")
    df = df[df["_year"].between(1800, 1899)].drop(columns=["_year"])
    print(f"  after date in [1800, 1899]:   {len(df):>7,}")

    if BOOK_IDS is not None:
        df = df[df["book_id"].isin(BOOK_IDS)]
        print(f"  after BOOK_IDS filter:        {len(df):>7,}")

    if N_BOOKS is not None and len(df) > N_BOOKS:
        df = df.sample(n=N_BOOKS, random_state=RANDOM_STATE)
        print(f"  after N_BOOKS={N_BOOKS} cap:       {len(df):>7,}")

    return df.reset_index(drop=True)


# =============================================================================
# tokenizer
# =============================================================================

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def load_tokenizer():
    try:
        import transformers

        tok = transformers.GPT2TokenizerFast.from_pretrained(TOKENIZER_NAME)
        tok.model_max_length = int(1e12)

        test_ids = tok("the quick brown fox")["input_ids"]
        if tok.vocab_size < 1000 or len(test_ids) == 0:
            raise RuntimeError(
                f"Tokenizer looks broken (vocab_size={tok.vocab_size}, "
                f"test={len(test_ids)} ids). Check HF_HOME / offline cache."
            )

        print(
            f"Using GPT2TokenizerFast ('{TOKENIZER_NAME}'), "
            f"vocab_size={tok.vocab_size}, batch_size={TOKEN_BATCH_SIZE}."
        )

        def batch_tok_fn(texts):
            enc = tok(
                texts,
                add_special_tokens=False,
                padding=False,
                truncation=False,
            )
            return enc["input_ids"]

        return batch_tok_fn, tok.vocab_size, TOKENIZER_NAME

    except Exception as exc:
        print(f"[fallback] Could not load {TOKENIZER_NAME} "
              f"({type(exc).__name__}: {exc})")
        print("[fallback] Using simple whitespace/punctuation tokenizer.")

        vocab: dict[str, int] = {}

        def _id(w: str) -> int:
            if w not in vocab:
                vocab[w] = len(vocab)
            return vocab[w]

        def batch_tok_fn(texts):
            return [
                [_id(t) for t in re.findall(r"\w+|[^\w\s]", text.lower())]
                for text in texts
            ]

        return batch_tok_fn, None, "whitespace_fallback"


# =============================================================================
# pallet update helpers
# =============================================================================

def update_pallet_dense(pallet: torch.Tensor, ids: list[int], vocab_size: int):
    x = torch.tensor(ids, dtype=torch.long)
    pallet[0] += torch.bincount(x, minlength=vocab_size)  # TF
    pallet[1, x.unique()] += 1                             # DF: once per page


def update_pallet_sparse(tf_counts: dict, df_counts: dict, ids: list[int]):
    for tid in ids:
        tf_counts[tid] += 1
    for tid in set(ids):
        df_counts[tid] += 1


# =============================================================================
# main streaming loop
# =============================================================================

def build_prior(metadata_csv: Path, batch_tok_fn, vocab_size):
    df = pd.read_csv(metadata_csv, dtype={"record_id": str, "book_id": str})
    print(f"\nMetadata loaded: {len(df):,} total rows")
    df = filter_metadata(df)
    df = df.assign(decade_folder=df["path_to_json"].str.split("/").str[0])
    print(f"Selected {len(df):,} book(s) across "
          f"{df['decade_folder'].nunique()} decade folder(s).\n")

    use_dense = vocab_size is not None
    if use_dense:
        pallet = torch.zeros((2, vocab_size), dtype=torch.long)
        tf_counts = df_counts = None
    else:
        pallet = None
        tf_counts: dict[int, int] = {}
        df_counts: dict[int, int] = {}
        vocab_size = 0

    # Reservoir: list of (page_token_ids, metadata_dict)
    # Populated via reservoir sampling as pages are streamed.
    # Scored against the prior AFTER the full prior is built.
    qa_reservoir: list[tuple[list[int], dict]] = []
    rng = random.Random(RANDOM_STATE)

    n_pages = 0
    n_books = 0
    books_seen: set[str] = set()
    batch_texts: list[str] = []
    batch_meta:  list[dict] = []

    def flush_batch():
        nonlocal n_pages, vocab_size

        if not batch_texts:
            return

        all_ids: list[list[int]] = batch_tok_fn(batch_texts)

        for page_ids, meta in zip(all_ids, batch_meta):
            if not page_ids:
                continue

            if use_dense:
                update_pallet_dense(pallet, page_ids, vocab_size)
            else:
                update_pallet_sparse(tf_counts, df_counts, page_ids)
                vocab_size = max(vocab_size, max(page_ids) + 1)

            # Reservoir sampling (Knuth/Vitter):
            # each page seen so far has equal probability of being in
            # the reservoir at any point during streaming.
            n_pages += 1
            if len(qa_reservoir) < QA_SAMPLE_SIZE:
                qa_reservoir.append((page_ids, meta))
            else:
                j = rng.randint(0, n_pages - 1)
                if j < QA_SAMPLE_SIZE:
                    qa_reservoir[j] = (page_ids, meta)

        batch_texts.clear()
        batch_meta.clear()

    for decade_folder, group in df.groupby("decade_folder"):
        tar_name = DECADE_TO_TAR.get(decade_folder)
        if tar_name is None:
            print(f"  [warn] no tar mapping for '{decade_folder}', "
                  f"skipping {len(group)} row(s).")
            continue

        tar_path = DATA_ROOT / tar_name
        row_lookup = {row["path_to_json"]: row for _, row in group.iterrows()}
        remaining  = set(row_lookup.keys())
        print(f"  streaming {tar_path.name} ({len(remaining)} book(s)) ...")

        with tarfile.open(tar_path, mode="r|gz") as tar:
            for member in tar:
                if not remaining:
                    break
                if member.name not in remaining:
                    continue

                f = tar.extractfile(member)
                if f is None:
                    remaining.discard(member.name)
                    continue

                raw = f.read()
                row = row_lookup[member.name]
                n_added = 0

                with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
                    for line in gz:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            page = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        if SKIP_EMPTY_PAGES and page.get("empty_pg"):
                            continue
                        text = (page.get("text") or "").strip()
                        if len(text) < MIN_PAGE_CHARS:
                            continue

                        batch_texts.append(text)
                        batch_meta.append({
                            "record_id": row["record_id"],
                            "book_id":   row["book_id"],
                            "pg":        page.get("pg"),
                        })
                        n_added += 1

                        if len(batch_texts) >= TOKEN_BATCH_SIZE:
                            flush_batch()

                        if MAX_PAGES_PER_BOOK and n_added >= MAX_PAGES_PER_BOOK:
                            break

                flush_batch()
                remaining.discard(member.name)
                books_seen.add(row["book_id"])
                n_books += 1

                if n_books % PROGRESS_EVERY_N_BOOKS == 0:
                    print(f"    ... {n_books:,} books | {n_pages:,} pages so far")

        flush_batch()

        if remaining:
            print(f"  [warn] {len(remaining)} member(s) not found in "
                  f"{tar_path.name}: {sorted(remaining)[:5]}")
        print(f"  {decade_folder} done: {n_books:,} books, "
              f"{n_pages:,} pages total so far.")

    flush_batch()
    print(f"\nStreaming complete: {n_pages:,} pages across {len(books_seen):,} books.")

    if not use_dense:
        vocab_size = (max(tf_counts.keys()) + 1) if tf_counts else 1
        pallet = torch.zeros((2, vocab_size), dtype=torch.long)
        for tid, c in tf_counts.items():
            pallet[0, tid] = c
        for tid, c in df_counts.items():
            pallet[1, tid] = c

    return pallet, vocab_size, n_pages, books_seen, qa_reservoir


# =============================================================================
# prior computation and corpus baseline statistics
# =============================================================================

def compute_prior_and_stats(pallet, vocab_size, qa_reservoir):
    tf_t     = pallet[0].float()
    df_t     = pallet[1].float()
    weighted = tf_t * df_t
    prior    = (weighted + SMOOTHING_EPS) / (
        weighted.sum() + SMOOTHING_EPS * vocab_size
    )

    # ------------------------------------------------------------------
    # Score every page in the QA reservoir against the now-complete prior.
    #
    # For each page we compute:
    #   mu_d    = mean(log(prior[t])) over all tokens  → sentence_mean_log_prior
    #   sigma_d = std(prior[t] * 1000)                 → sentence_prior_dispersion
    #
    # Then we take the mean and std of those values across all reservoir
    # pages to get the four corpus baseline numbers saved to the prior file.
    # ------------------------------------------------------------------
    print(f"\nComputing corpus baseline stats from "
          f"{len(qa_reservoir)} reservoir pages ...")

    mus:    list[float] = []
    sigmas: list[float] = []

    for page_ids, _ in qa_reservoir:
        if not page_ids:
            continue
        x  = torch.tensor(page_ids, dtype=torch.long)
        tp = prior[x]
        mus.append(tp.log().mean().item())
        sigmas.append((tp * 1000).std().item() if len(page_ids) > 1 else 0.0)

    if not mus:
        raise SystemExit(
            "No valid pages in QA reservoir — cannot compute corpus baseline stats."
        )

    mus_t    = torch.tensor(mus)
    sigmas_t = torch.tensor(sigmas)

    corpus_mean_log_prior       = mus_t.mean().item()
    corpus_mean_log_prior_std   = mus_t.std().item()
    corpus_prior_dispersion     = sigmas_t.mean().item()
    corpus_prior_dispersion_std = sigmas_t.std().item()

    print(f"  corpus_mean_log_prior:       "
          f"{corpus_mean_log_prior:.4f} ± {corpus_mean_log_prior_std:.4f}")
    print(f"  corpus_prior_dispersion:     "
          f"{corpus_prior_dispersion:.4f} ± {corpus_prior_dispersion_std:.4f}")

    return (
        prior,
        corpus_mean_log_prior,
        corpus_mean_log_prior_std,
        corpus_prior_dispersion,
        corpus_prior_dispersion_std,
    )


# =============================================================================
# main
# =============================================================================

def main():
    print("=" * 72)
    print("build_prior_final.py — document unit: complete page, no chunking")
    print(f"Job ID:   {os.environ.get('SLURM_JOB_ID', 'local')}")
    print(f"Hostname: {os.uname().nodename}")
    print("=" * 72)

    batch_tok_fn, vocab_size, tokenizer_name = load_tokenizer()

    pallet, vocab_size, n_pages, books_seen, qa_reservoir = build_prior(
        METADATA_CSV, batch_tok_fn, vocab_size,
    )

    if n_pages < 2:
        raise SystemExit("Fewer than 2 pages loaded — check CONFIG filters and paths.")

    (
        prior,
        corpus_mean_log_prior,
        corpus_mean_log_prior_std,
        corpus_prior_dispersion,
        corpus_prior_dispersion_std,
    ) = compute_prior_and_stats(pallet, vocab_size, qa_reservoir)

    OUTPUT_PRIOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "prior":                       prior,
            "vocab_size":                  vocab_size,
            "tokenizer_name":              tokenizer_name,
            "n_pages":                     n_pages,
            "n_books":                     len(books_seen),
            "corpus_mean_log_prior":       corpus_mean_log_prior,
            "corpus_mean_log_prior_std":   corpus_mean_log_prior_std,
            "corpus_prior_dispersion":     corpus_prior_dispersion,
            "corpus_prior_dispersion_std": corpus_prior_dispersion_std,
        },
        OUTPUT_PRIOR_PATH,
    )

    print(f"\nSaved prior → {OUTPUT_PRIOR_PATH}")
    print(f"  pages:           {n_pages:,}")
    print(f"  books:           {len(books_seen):,}")
    print(f"  nonzero entries: {(prior > 0).sum().item():,} / {vocab_size:,}")


if __name__ == "__main__":
    main()