#!/usr/bin/env python3
"""
build_prior_v3.py — Step 1 of the prior-filtering pipeline.

Builds one TF x DF token-count pallet for ONE BL Microsoft decade folder.
The SLURM array launcher runs this script once per decade.

Required corpus filters:
    - best_version == 1
    - main_language == "English"
    - date in [1800, 1899]

For each decade task, saves:
    prior_filter/blmicrosoft/partials/pallet_{decade}.pt

The partial file contains the raw TF/DF pallet plus the page reservoir
needed by merge_prior_v3.py to compute the corpus-level mu/sigma baseline.
"""

import argparse
import gzip
import io
import json
import os
import random
import re
import tarfile
from collections import defaultdict
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

DATA_ROOT = Path("/gpfs/projects/bsc100/textmachine-data/downloaded_data_vm")

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

VALID_DECADES = tuple(DECADE_TO_TAR.keys())

# --- subset selection --------------------------------------------------------
N_BOOKS = None          # cap number of books per decade; None = all
RANDOM_STATE = 42
BOOK_IDS = None         # explicit list of book_ids to include, or None

# --- page-level filtering ----------------------------------------------------
MAX_PAGES_PER_BOOK = None   # None = no cap
SKIP_EMPTY_PAGES = True
MIN_PAGE_CHARS = 50

# --- output ------------------------------------------------------------------
PARTIAL_OUTPUT_DIR = Path("prior_filter/blmicrosoft/partials")

TOKENIZER_NAME = "gpt2-large"

# Tokenize pages in batches rather than one Python/Rust call per page.
TOKENIZE_BATCH_SIZE = 64

# Additive smoothing is applied only by merge_prior_v3.py, after all
# decade-level TF/DF pallets have been summed.
SMOOTHING_EPS = 1e-6

# --- memory / progress controls ---------------------------------------------
QA_SAMPLE_SIZE = 5000
PROGRESS_EVERY_N_BOOKS = 500


# =============================================================================
# argument parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--decade",
        required=True,
        choices=VALID_DECADES,
        help="Decade folder to process, e.g. 1840_1849.",
    )
    return parser.parse_args()


# =============================================================================
# metadata filtering
# =============================================================================

def filter_metadata(df: pd.DataFrame, decade_folder: str) -> pd.DataFrame:
    """
    Apply the corpus filters and select exactly one decade folder.
    """
    n_start = len(df)

    df = df[df["best_version"] == 1]
    print(f"  after best_version==1:        {len(df):>7,} / {n_start:,}")

    df = df[df["main_language"] == "English"]
    print(f"  after main_language==English: {len(df):>7,}")

    df = df.copy()
    df["_year"] = pd.to_numeric(df["date"], errors="coerce")
    n_bad_date = df["_year"].isna().sum()
    if n_bad_date:
        print(f"  [warn] {n_bad_date} row(s) with unparseable date dropped")

    df = df[df["_year"].between(1800, 1899)]
    df["decade_folder"] = df["path_to_json"].str.split("/").str[0]
    df = df[df["decade_folder"] == decade_folder]
    df = df.drop(columns=["_year"])

    print(f"  after date in [1800, 1899]:   {len(df):>7,}")
    print(f"  after decade={decade_folder}:      {len(df):>7,}")

    if BOOK_IDS is not None:
        df = df[df["book_id"].isin(BOOK_IDS)]
        print(f"  after BOOK_IDS filter:        {len(df):>7,}")

    if N_BOOKS is not None and len(df) > N_BOOKS:
        df = df.sample(n=N_BOOKS, random_state=RANDOM_STATE)
        print(f"  after N_BOOKS={N_BOOKS} cap:  {len(df):>7,}")

    return df.reset_index(drop=True)


# =============================================================================
# tokenizer
# =============================================================================

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def load_tokenizer():
    """
    Returns (tokenize_batch_fn, vocab_size, tokenizer_name).

    tokenize_batch_fn(texts) -> list[list[int]]

    With GPT2TokenizerFast, one tokenizer call handles a whole batch.
    The fallback keeps the same batched interface.
    """
    try:
        import transformers

        tok = transformers.GPT2TokenizerFast.from_pretrained(TOKENIZER_NAME)
        tok.model_max_length = int(1e12)

        # Silent-failure guard.
        test_ids = tok("the quick brown fox")["input_ids"]
        if tok.vocab_size < 1000 or len(test_ids) == 0:
            raise RuntimeError(
                f"Tokenizer loaded but looks broken (vocab_size={tok.vocab_size}, "
                f"test tokenization returned {len(test_ids)} ids). "
                f"Re-run download_assets.sh and verify HF_HOME on this node."
            )

        print(
            f"Using GPT2TokenizerFast ('{TOKENIZER_NAME}'), "
            f"vocab_size={tok.vocab_size}, batch_size={TOKENIZE_BATCH_SIZE}."
        )

        def tokenize_batch(texts):
            encoded = tok(
                texts,
                padding=False,
                truncation=False,
                add_special_tokens=True,
            )
            return encoded["input_ids"]

        return tokenize_batch, tok.vocab_size, TOKENIZER_NAME

    except Exception as e:
        print(f"[fallback] Could not load {TOKENIZER_NAME} ({type(e).__name__}: {e})")
        print("[fallback] Using simple whitespace/punctuation tokenizer.")

        vocab = {}

        def tokenize_one(text):
            ids = []
            for t in re.findall(r"\w+|[^\w\s]", text.lower()):
                if t not in vocab:
                    vocab[t] = len(vocab)
                ids.append(vocab[t])
            return ids

        def tokenize_batch(texts):
            return [tokenize_one(text) for text in texts]

        return tokenize_batch, None, "whitespace_fallback"


# =============================================================================
# batch processing helpers
# =============================================================================

def process_tokenized_page(
    ids,
    text,
    row,
    page,
    pallet,
    use_dense,
    vocab_size,
    tf_counts,
    df_counts,
    qa_reservoir,
    rng,
    n_pages,
):
    """Update TF/DF counts and the QA reservoir for one already-tokenized page."""
    if not ids:
        return n_pages

    if use_dense:
        x = torch.tensor(ids, dtype=torch.long)
        pallet[0] += torch.bincount(x, minlength=vocab_size)
        pallet[1, x.unique()] += 1
    else:
        for tid in ids:
            tf_counts[tid] += 1
        for tid in set(ids):
            df_counts[tid] += 1

    n_pages += 1

    meta = {
        "record_id": row["record_id"],
        "book_id": row["book_id"],
        "pg": page.get("pg"),
    }

    if len(qa_reservoir) < QA_SAMPLE_SIZE:
        qa_reservoir.append((text, meta))
    else:
        j = rng.randint(0, n_pages - 1)
        if j < QA_SAMPLE_SIZE:
            qa_reservoir[j] = (text, meta)

    return n_pages


def flush_token_batch(
    batch_texts,
    batch_pages,
    tokenize_batch,
    pallet,
    use_dense,
    vocab_size,
    tf_counts,
    df_counts,
    qa_reservoir,
    rng,
    n_pages,
):
    """Tokenize one batch and unpack the results into the running pallet."""
    if not batch_texts:
        return n_pages

    batch_ids = tokenize_batch(batch_texts)

    if len(batch_ids) != len(batch_texts):
        raise RuntimeError(
            f"Tokenizer returned {len(batch_ids)} results for "
            f"{len(batch_texts)} input texts."
        )

    for ids, (text, row, page) in zip(batch_ids, batch_pages):
        n_pages = process_tokenized_page(
            ids=ids,
            text=text,
            row=row,
            page=page,
            pallet=pallet,
            use_dense=use_dense,
            vocab_size=vocab_size,
            tf_counts=tf_counts,
            df_counts=df_counts,
            qa_reservoir=qa_reservoir,
            rng=rng,
            n_pages=n_pages,
        )

    batch_texts.clear()
    batch_pages.clear()

    return n_pages


# =============================================================================
# streaming prior build
# =============================================================================

def build_prior_streaming(
    metadata_csv: Path,
    decade_folder: str,
    tokenize_batch,
    vocab_size,
):
    df = pd.read_csv(metadata_csv, dtype={"record_id": str, "book_id": str})
    print(f"\nMetadata loaded: {len(df):,} total rows")
    df = filter_metadata(df, decade_folder)

    print(
        f"\nSelected {len(df):,} book(s) in decade folder '{decade_folder}'."
    )

    use_dense = vocab_size is not None

    if use_dense:
        pallet = torch.zeros((2, vocab_size), dtype=torch.long)
        tf_counts = None
        df_counts = None
    else:
        pallet = None
        tf_counts = defaultdict(int)
        df_counts = defaultdict(int)

    qa_reservoir = []
    # Distinguish the decade tasks while keeping deterministic sampling.
    decade_seed = RANDOM_STATE + VALID_DECADES.index(decade_folder)
    rng = random.Random(decade_seed)

    n_pages = 0
    n_books_done = 0
    books_seen: set[str] = set()

    tar_name = DECADE_TO_TAR[decade_folder]
    tar_path = DATA_ROOT / tar_name

    row_lookup = {
        row["path_to_json"]: row
        for _, row in df.iterrows()
    }
    remaining = set(row_lookup.keys())

    print(f"  streaming {tar_path.name} for {len(remaining)} book(s) ...")

    batch_texts = []
    batch_pages = []

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
                    batch_pages.append((text, row, page))
                    n_added += 1

                    if len(batch_texts) >= TOKENIZE_BATCH_SIZE:
                        n_pages = flush_token_batch(
                            batch_texts=batch_texts,
                            batch_pages=batch_pages,
                            tokenize_batch=tokenize_batch,
                            pallet=pallet,
                            use_dense=use_dense,
                            vocab_size=vocab_size,
                            tf_counts=tf_counts,
                            df_counts=df_counts,
                            qa_reservoir=qa_reservoir,
                            rng=rng,
                            n_pages=n_pages,
                        )

                    if (
                        MAX_PAGES_PER_BOOK is not None
                        and n_added >= MAX_PAGES_PER_BOOK
                    ):
                        break

            # A page/book member is complete even if its final few pages
            # remain buffered; the buffer may be flushed after this member.
            n_pages = flush_token_batch(
                batch_texts=batch_texts,
                batch_pages=batch_pages,
                tokenize_batch=tokenize_batch,
                pallet=pallet,
                use_dense=use_dense,
                vocab_size=vocab_size,
                tf_counts=tf_counts,
                df_counts=df_counts,
                qa_reservoir=qa_reservoir,
                rng=rng,
                n_pages=n_pages,
            )

            remaining.discard(member.name)
            books_seen.add(row["book_id"])
            n_books_done += 1

            if n_books_done % PROGRESS_EVERY_N_BOOKS == 0:
                print(
                    f"    ... {n_books_done} books processed, "
                    f"{n_pages} pages so far"
                )

    # Flush anything left over in the very unlikely case that the loop exits
    # after an early tar termination.
    n_pages = flush_token_batch(
        batch_texts=batch_texts,
        batch_pages=batch_pages,
        tokenize_batch=tokenize_batch,
        pallet=pallet,
        use_dense=use_dense,
        vocab_size=vocab_size,
        tf_counts=tf_counts,
        df_counts=df_counts,
        qa_reservoir=qa_reservoir,
        rng=rng,
        n_pages=n_pages,
    )

    if remaining:
        print(
            f"  [warn] {len(remaining)} member(s) not found in {tar_path.name}: "
            f"{sorted(remaining)[:5]}"
        )

    print(
        f"\nDone streaming {decade_folder}: "
        f"{n_pages:,} page(s) across {len(books_seen):,} book(s)."
    )

    if not use_dense:
        vocab_size = (max(tf_counts.keys()) + 1) if tf_counts else 1
        pallet = torch.zeros((2, vocab_size), dtype=torch.long)
        for tid, count in tf_counts.items():
            pallet[0, tid] = count
        for tid, count in df_counts.items():
            pallet[1, tid] = count

    return pallet, vocab_size, n_pages, books_seen, qa_reservoir


# =============================================================================
# main
# =============================================================================

def main():
    args = parse_args()

    print("=" * 72)
    print(f"Building prior partial for decade: {args.decade}")
    print(f"Job ID: {os.environ.get('SLURM_JOB_ID', 'local')}")
    print(f"Hostname: {os.uname().nodename}")
    print("=" * 72)

    tokenize_batch, vocab_size, tokenizer_name = load_tokenizer()

    pallet, vocab_size, n_pages, books_seen, qa_reservoir = build_prior_streaming(
        METADATA_CSV,
        args.decade,
        tokenize_batch,
        vocab_size,
    )

    if n_pages < 1:
        raise SystemExit(
            f"No valid pages were loaded for decade {args.decade}. "
            f"Check metadata, tar mapping, filters and paths."
        )

    PARTIAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PARTIAL_OUTPUT_DIR / f"pallet_{args.decade}.pt"

    torch.save(
        {
            "pallet": pallet,
            "vocab_size": vocab_size,
            "tokenizer_name": tokenizer_name,
            "decade": args.decade,
            "n_pages": n_pages,
            "n_books": len(books_seen),
            "qa_reservoir": qa_reservoir,
        },
        output_path,
    )

    print(f"\nSaved partial pallet → {output_path}")
    print(f"  decade: {args.decade}")
    print(f"  pages: {n_pages:,}")
    print(f"  books: {len(books_seen):,}")
    print(f"  QA reservoir: {len(qa_reservoir):,}")


if __name__ == "__main__":
    main()
