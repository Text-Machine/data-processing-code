"""
s2_build_prior.py
"""

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
    "consolidated_metadata/metadata_blmicrosoft_step_3.csv"
)

DATA_ROOT = Path("/gpfs/projects/bsc100/textmachine-data/downloaded_data_vm")

DECADE_TO_TAR = {
    "1510_1699": "OCR_text_c_1510_-_1699.tar.gz",
    "1700_1799": "OCR_text_1700_-_1799.tar.gz",
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

# --- subset selection (set to None to use ALL books in the metadata csv) ---
N_BOOKS = 5          # cap number of books (random sample) -- None = all
RANDOM_STATE = 42
BOOK_IDS = None
DECADE_FOLDERS = None

# --- page-level filtering ---------------------------------------------------
MAX_PAGES_PER_BOOK = None  # None = no cap (use all pages once you trust the pipeline)
SKIP_EMPTY_PAGES = True
MIN_PAGE_CHARS = 50

# --- output -------------------------------------------------------------
OUTPUT_PRIOR_PATH = Path("prior_filter/blmicrosoft/all_prior_tfdf.pt")
TOKENIZER_NAME = "gpt2-large"

# Additive (Laplace) smoothing applied to the TF x DF prior before
# normalizing. Without this, ANY token that never appears anywhere in the
# page-level corpus gets prior=0, so log(prior)=-inf for any downstream
# sentence that happens to contain it -- silently poisoning that sentence's
# mu_d to -inf. A tiny epsilon keeps every token's prior strictly positive
# without meaningfully perturbing the probabilities of tokens actually seen.
SMOOTHING_EPS = 1e-6

# --- memory / progress controls ---------------------------------------------
QA_SAMPLE_SIZE = 500       # bounded reservoir sample used for QA diagnostics only
PROGRESS_EVERY_N_BOOKS = 500


# =============================================================================
# metadata subset
# =============================================================================

def filter_metadata(df: pd.DataFrame) -> pd.DataFrame:
    if BOOK_IDS is not None:
        df = df[df["book_id"].isin(BOOK_IDS)]
    if DECADE_FOLDERS is not None:
        decade_folder = df["path_to_json"].str.split("/").str[0]
        df = df[decade_folder.isin(DECADE_FOLDERS)]
    if N_BOOKS is not None and len(df) > N_BOOKS:
        df = df.sample(n=N_BOOKS, random_state=RANDOM_STATE)
    return df.reset_index(drop=True)


# =============================================================================
# tokenizer (real GPT2TokenizerFast, offline whitespace fallback)
# =============================================================================

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def load_tokenizer():
    """
    Returns (tok_fn, vocab_size, tokenizer_name) where tok_fn(text) -> list[int].
    vocab_size is None for the whitespace fallback (unknown until the corpus
    has been fully scanned, since its vocab grows as new words are seen).
    """
    try:
        import transformers

        tok = transformers.GPT2TokenizerFast.from_pretrained(TOKENIZER_NAME)
        tok.model_max_length = int(1e12)  # <-- silences the truncation warning

        # IMPORTANT: in offline mode (TRANSFORMERS_OFFLINE=1), if the model
        # isn't actually present in the local HF cache, from_pretrained does
        # NOT reliably raise -- it can silently hand back a broken tokenizer
        # with vocab_size=0 / that tokenizes everything to nothing. Catching
        # only exceptions would miss this entirely and silently build a
        # garbage prior. Verify it actually works before trusting it.
        test_ids = tok("the quick brown fox")["input_ids"]
        if tok.vocab_size < 1000 or len(test_ids) == 0:
            raise RuntimeError(
                f"Tokenizer loaded but looks broken (vocab_size={tok.vocab_size}, "
                f"test tokenization returned {len(test_ids)} ids). This usually means "
                f"the offline HF cache at HF_HOME doesn't actually contain "
                f"'{TOKENIZER_NAME}' -- re-run download_assets.sh and check HF_HOME "
                f"points at the same path on this node."
            )

        print(f"Using real GPT2TokenizerFast ('{TOKENIZER_NAME}') from local cache, "
              f"vocab_size={tok.vocab_size}.")
        return (lambda text: tok(text)["input_ids"]), tok.vocab_size, TOKENIZER_NAME
    except Exception as e:
        print(f"[fallback] Could not load {TOKENIZER_NAME} tokenizer ({type(e).__name__}: {e})")
        print("[fallback] Using simple whitespace/punctuation tokenizer instead.")

        vocab = {}

        def simple_tokenize(text):
            return re.findall(r"\w+|[^\w\s]", text.lower())

        def tok_fn(text):
            ids = []
            for t in simple_tokenize(text):
                if t not in vocab:
                    vocab[t] = len(vocab)
                ids.append(vocab[t])
            return ids

        return tok_fn, None, "whitespace_fallback"


# =============================================================================
# streaming build: one pass over the selected tar.gz archives, folding each
# page into the running TF/DF accumulator immediately, keeping only a bounded
# reservoir sample of pages around for QA
# =============================================================================

def build_prior_streaming(metadata_csv: Path, tok_fn, vocab_size):
    df = pd.read_csv(metadata_csv, dtype={"record_id": str, "book_id": str})
    df = filter_metadata(df)
    df = df.assign(decade_folder=df["path_to_json"].str.split("/").str[0])
    print(f"Selected {len(df)} book(s) across {df['decade_folder'].nunique()} decade folder(s).")

    use_dense = vocab_size is not None
    if use_dense:
        pallet = torch.zeros((2, vocab_size))  # [0]=TF, [1]=DF
    else:
        tf_counts = defaultdict(int)
        df_counts = defaultdict(int)

    qa_reservoir = []  # bounded list of (text, meta), reservoir-sampled
    rng = random.Random(RANDOM_STATE)

    n_pages = 0
    n_books_done = 0
    books_seen = set()

    for decade_folder, group in df.groupby("decade_folder"):
        tar_name = DECADE_TO_TAR.get(decade_folder)
        if tar_name is None:
            print(f"[warn] no tar mapping for folder '{decade_folder}', skipping {len(group)} row(s)")
            continue

        tar_path = DATA_ROOT / tar_name
        row_lookup = {row["path_to_json"]: row for _, row in group.iterrows()}
        remaining = set(row_lookup.keys())
        print(f"  streaming {tar_path.name} for {len(remaining)} book(s) ...")

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

                raw = f.read()  # one book's compressed bytes -- small, transient
                row = row_lookup[member.name]
                n_added = 0

                with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
                    for line in gz:  # one page at a time, never all pages at once
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

                        ids = tok_fn(text)
                        if ids:
                            if use_dense:
                                x = torch.tensor(ids, dtype=torch.long)
                                pallet[0] += torch.bincount(x, minlength=vocab_size)
                                pallet[1, x] += 1
                            else:
                                for tid in ids:
                                    tf_counts[tid] += 1
                                for tid in set(ids):
                                    df_counts[tid] += 1

                        n_pages += 1

                        # reservoir sampling: keeps a uniform random sample of
                        # QA_SAMPLE_SIZE pages across the WHOLE corpus, bounded
                        # memory regardless of how many pages we've seen
                        meta = {"record_id": row["record_id"], "book_id": row["book_id"],
                                 "title": row["title"], "pg": page.get("pg")}
                        if len(qa_reservoir) < QA_SAMPLE_SIZE:
                            qa_reservoir.append((text, meta))
                        else:
                            j = rng.randint(0, n_pages - 1)
                            if j < QA_SAMPLE_SIZE:
                                qa_reservoir[j] = (text, meta)

                        n_added += 1
                        if MAX_PAGES_PER_BOOK is not None and n_added >= MAX_PAGES_PER_BOOK:
                            break

                # book's raw bytes / decompressed pages are now eligible for GC
                remaining.discard(member.name)
                books_seen.add(row["book_id"])
                n_books_done += 1
                if n_books_done % PROGRESS_EVERY_N_BOOKS == 0:
                    print(f"    ... {n_books_done} books processed, {n_pages} pages so far")

        if remaining:
            preview = sorted(remaining)[:5]
            print(f"  [warn] {len(remaining)} member(s) not found in {tar_path.name}: {preview}")

    print(f"Done streaming: {n_pages} page(s) across {len(books_seen)} book(s).")

    if not use_dense:
        # whitespace fallback: vocab size only known now; materialize a dense
        # tensor from the dict accumulators (bounded by actual distinct tokens
        # seen, not corpus size)
        vocab_size = (max(tf_counts.keys()) + 1) if tf_counts else 1
        pallet = torch.zeros((2, vocab_size))
        for tid, c in tf_counts.items():
            pallet[0, tid] = c
        for tid, c in df_counts.items():
            pallet[1, tid] = c

    return pallet, vocab_size, n_pages, books_seen, qa_reservoir


# =============================================================================
# main
# =============================================================================

def main():
    tok_fn, vocab_size, tokenizer_name = load_tokenizer()

    pallet, vocab_size, n_pages, books_seen, qa_reservoir = build_prior_streaming(
        METADATA_CSV, tok_fn, vocab_size
    )
    if n_pages < 2:
        raise SystemExit(
            "Fewer than 2 page-documents were loaded -- widen CONFIG (N_BOOKS / "
            "BOOK_IDS / DECADE_FOLDERS) before building the prior."
        )

    tf = pallet[0]
    df_counts = pallet[1]
    weighted = tf * df_counts
    # Laplace smoothing: guarantees prior > 0 everywhere, so log(prior) is
    # always finite downstream, even for tokens unseen in this corpus.
    prior = (weighted + SMOOTHING_EPS) / (weighted.sum() + SMOOTHING_EPS * vocab_size)

    # -------------------------------------------------------------------
    # QA diagnostics on the bounded reservoir sample only (not the whole
    # corpus) -- purely a sanity check, not saved anywhere
    # -------------------------------------------------------------------
    print(f"\n{n_pages} page-documents total, vocab_size={vocab_size}, "
          f"QA sample={len(qa_reservoir)} page(s)")

    mus, sigmas = [], []
    for i, (text, meta) in enumerate(qa_reservoir):
        ids = tok_fn(text)
        if not ids:
            continue
        x = torch.tensor(ids, dtype=torch.long)
        token_priors = prior[x]
        mu_d = token_priors.log().mean().item()
        sigma_d = (token_priors * 1000).std().item()
        mus.append(mu_d)
        sigmas.append(sigma_d)
        if i < 20:
            preview = text[:50].replace("\n", " ")
            print(f"  page {i:4d}: {len(ids):4d} tok | book {meta['book_id']} pg {meta['pg']} | {preview!r}...")
    if len(qa_reservoir) > 20:
        print(f"  ... ({len(qa_reservoir) - 20} more in QA sample)")

    if mus:
        mus_t, sigmas_t = torch.tensor(mus), torch.tensor(sigmas)
        print(f"\n[QA] page-level mu_d:    mean={mus_t.mean():.4f}  std={mus_t.std():.4f}")
        print(f"[QA] page-level sigma_d: mean={sigmas_t.mean():.4f}  std={sigmas_t.std():.4f}")

    # -------------------------------------------------------------------
    # persist
    # -------------------------------------------------------------------
    OUTPUT_PRIOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "prior": prior,
            "vocab_size": vocab_size,
            "tokenizer_name": tokenizer_name,
            "n_pages": n_pages,
            "n_books": len(books_seen),
        },
        OUTPUT_PRIOR_PATH,
    )
    print(f"\nSaved prior to {OUTPUT_PRIOR_PATH} (built from {n_pages} pages across {len(books_seen)} books)")


if __name__ == "__main__":
    main()