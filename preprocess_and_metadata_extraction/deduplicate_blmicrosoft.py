"""
deduplicate_blmicrosoft.py
==========================
Deduplicates the BL Microsoft metadata CSV on a per-author basis using a single
shared `dedupe` model trained once and reused for every author.

Overview
--------
The script runs in two distinct phases:

**Phase 1 — Training** (interactive, ~10-15 minutes, run once)::

    python deduplicate_blmicrosoft.py --input metadata.csv --sample-train

A random cross-author sample is drawn, and `dedupe` asks you to label ~10-20
record pairs as duplicates or non-duplicates (y/n/u, then 'f' to finish).
The trained model is saved to ``./dedupe_settings/shared.settings`` and the
labelled pairs to ``./dedupe_settings/shared_training.json``.  You never need
to repeat this step.

**Phase 2 — Inference** (fully automated, no user interaction)::

    python deduplicate_blmicrosoft.py --input metadata.csv

The saved model is loaded and applied to every author's subset of rows.
The input CSV is copied to ``<input_stem>_deduplicated.csv`` (same directory)
with two new columns appended:

``book_id`` (int)
    A globally unique integer for each distinct work across the entire dataset.
    The counter increments once per unique cluster (never per row), so duplicate
    editions share the same ``book_id`` and the counter never resets between
    authors.

``best_version`` (int, 0 or 1)
    Within each ``book_id`` cluster, exactly one row is flagged 1 — the row
    with the highest ``mean_wc_ocr`` value.  All other rows in the cluster
    receive 0.  Ties are broken by keeping the first occurrence (lowest
    original row index).

Post-processing
---------------
After the ML model clusters records, a deterministic substring-merge pass
catches title-prefix variants that the model tends to miss, e.g.:

- "Oliver Twist"  <->  "The Adventures of Oliver Twist"
- "Barnaby Rudge" <->  "Barnaby Rudge: A Tale of the Riots of Eighty"

See ``should_merge_by_substring()`` for the merging rules.

Dependencies
------------
::

    pip install dedupe pandas tqdm

Configuration
-------------
The key tunables at the top of this file are:

- ``MATCH_THRESHOLD``             -- dedupe confidence threshold (lower -> more permissive).
- ``SUBSTRING_MERGE_MAX_RATIO``   -- length-ratio cap for Rule 1 substring merges.
- ``SAMPLE_SIZE``                 -- number of rows drawn for the training sample.
- ``OCR_COL``                     -- column used to pick the best version per cluster.
"""

import argparse
import logging
import re
from pathlib import Path

import dedupe
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

AUTHOR_COL = "author"
OCR_COL    = "mean_wc_ocr"   # column used to select the best version per cluster

MATCH_THRESHOLD           = 0.35
SAMPLE_SIZE               = 2_000
SUBSTRING_MERGE_MAX_RATIO = 2.5

DEDUPE_FIELDS = [
    dedupe.variables.String("title",  has_missing=True),
    dedupe.variables.String("author", has_missing=True),
]

SHARED_SETTINGS = Path("./dedupe_settings/shared.settings")
SHARED_TRAINING = Path("./dedupe_settings/shared_training.json")

logging.basicConfig(level=logging.WARNING)


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def clean_value(val) -> str | None:
    """Return *val* as a stripped string, or ``None`` if blank/NaN."""
    if pd.isna(val):
        return None
    return str(val).strip() or None


# ---------------------------------------------------------------------------
# Title normalisation
# ---------------------------------------------------------------------------

_LEADING_ARTICLES = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)
_BRACKET_SUFFIX   = re.compile(r"\s*\[.*?\]")
_PAREN_SUFFIX     = re.compile(r"\s*\(.*?\)")
_MULTI_SPACE      = re.compile(r"\s+")
_PUNCTUATION      = re.compile(r"[^\w\s]")


def normalize_title(val: str | None) -> str | None:
    """Return a normalised version of *val* suitable for fuzzy comparison."""
    if val is None:
        return None
    val = val.lower()
    val = _BRACKET_SUFFIX.sub("", val)
    val = _PAREN_SUFFIX.sub("", val)
    val = _LEADING_ARTICLES.sub("", val)
    val = _PUNCTUATION.sub("", val)
    val = _MULTI_SPACE.sub(" ", val).strip()
    return val or None


def to_record_dict(df: pd.DataFrame) -> dict[str, dict]:
    """Convert a DataFrame to the ``{record_id: fields}`` format dedupe expects."""
    return {
        str(idx): {
            "title":  normalize_title(clean_value(row.get("title"))),
            "author": clean_value(row.get("author")),
        }
        for idx, row in df.iterrows()
    }


# ---------------------------------------------------------------------------
# Substring-merge post-processing
# ---------------------------------------------------------------------------

def should_merge_by_substring(
    rep_a: str,
    rep_b: str,
    max_length_ratio: float = SUBSTRING_MERGE_MAX_RATIO,
) -> bool:
    """Return True if two normalised titles likely refer to the same work.

    Rule 1 -- substring with ratio guard:
        One title is a substring of the other AND their lengths are within
        *max_length_ratio* of each other.

    Rule 2 -- prefix match (no ratio limit):
        The longer title starts with the shorter one.
    """
    if not rep_a or not rep_b:
        return False
    shorter, longer = sorted([rep_a, rep_b], key=len)
    if shorter in longer and len(longer) / len(shorter) <= max_length_ratio:
        return True
    if longer.startswith(shorter):
        return True
    return False


def _representative_title(df: pd.DataFrame, local_id: int) -> str | None:
    """Return the shortest normalised title in the cluster identified by *local_id*."""
    titles = [
        normalize_title(clean_value(t))
        for t in df.loc[df["_local_id"] == local_id, "title"]
    ]
    titles = [t for t in titles if t]
    return min(titles, key=len) if titles else None


def apply_substring_merges(df: pd.DataFrame) -> pd.DataFrame:
    """Merge clusters whose representative titles satisfy the substring rules.

    Works on the internal ``_local_id`` column and renumbers it contiguously
    from 0 after merging.
    """
    cluster_ids = sorted(df["_local_id"].unique())
    book_rep    = {cid: _representative_title(df, cid) for cid in cluster_ids}

    merges: dict[int, int] = {}
    for i, id_a in enumerate(cluster_ids):
        for id_b in cluster_ids[i + 1:]:
            if should_merge_by_substring(book_rep.get(id_a), book_rep.get(id_b)):
                canonical, other = min(id_a, id_b), max(id_a, id_b)
                merges[other] = canonical

    def resolve(bid: int) -> int:
        while bid in merges:
            bid = merges[bid]
        return bid

    df = df.copy()
    df["_local_id"] = df["_local_id"].apply(resolve)

    id_map          = {old: new for new, old in enumerate(sorted(df["_local_id"].unique()))}
    df["_local_id"] = df["_local_id"].map(id_map)
    return df


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_shared_model(df: pd.DataFrame) -> None:
    """Interactively train the shared dedupe model and persist it to disk."""
    print(f"\nDrawing a sample of up to {SAMPLE_SIZE} rows for training ...")
    sample_df = df.sample(min(SAMPLE_SIZE, len(df)), random_state=42)
    records   = to_record_dict(sample_df)

    deduper = dedupe.Dedupe(DEDUPE_FIELDS)

    if SHARED_TRAINING.exists():
        print(f"Found existing training data at {SHARED_TRAINING} -- loading it.")
        with open(SHARED_TRAINING) as f:
            deduper.prepare_training(records, f)
    else:
        deduper.prepare_training(records)

    print("\n-- Active learning ---------------------------------------------------")
    print("  For each pair: (y) duplicate  (n) not duplicate  (u) unsure  (f) finished")
    print("  Aim for ~10-20 labelled pairs, then press 'f'.")
    print()
    print("  Note: titles are shown in normalised form (lowercased, brackets")
    print("  stripped, leading articles removed) -- this is intentional.")
    print()
    dedupe.console_label(deduper)

    deduper.train()

    SHARED_SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    with open(SHARED_TRAINING, "w") as f:
        deduper.write_training(f)
    with open(SHARED_SETTINGS, "wb") as f:
        deduper.write_settings(f)

    print(f"\nModel saved to {SHARED_SETTINGS}")
    print("Run the script again without --sample-train to process all authors.")


# ---------------------------------------------------------------------------
# Per-author deduplication
# ---------------------------------------------------------------------------

def load_shared_model() -> dedupe.StaticDedupe:
    """Load the pre-trained shared model from disk (read-only)."""
    if not SHARED_SETTINGS.exists():
        raise FileNotFoundError(
            f"Shared model not found at {SHARED_SETTINGS}.\n"
            "Run with --sample-train first to create it."
        )
    with open(SHARED_SETTINGS, "rb") as f:
        return dedupe.StaticDedupe(f)


def deduplicate_author(
    author_name: str,
    df: pd.DataFrame,
    deduper: dedupe.StaticDedupe,
    global_counter: int,
) -> tuple[pd.DataFrame, int]:
    """Apply the shared model to one author's rows and assign global book ids.

    The global counter is incremented once per distinct cluster (not per row),
    so duplicate editions share the same ``book_id`` and the counter is never
    reset between authors.

    Parameters
    ----------
    author_name:
        Used only for log messages.
    df:
        Subset of the metadata DataFrame for this author (original index kept).
    deduper:
        Pre-loaded ``StaticDedupe`` model shared across all authors.
    global_counter:
        The next free global book id to assign.

    Returns
    -------
    (df_with_book_id, new_counter)
        *df_with_book_id* is a copy of *df* with a ``book_id`` column added.
        *new_counter* is the updated counter to pass to the next author call.
    """
    from dedupe.core import BlockingError

    df = df.copy()

    # Trivial case: a single row is its own unique book.
    if len(df) == 1:
        df["book_id"] = global_counter
        return df, global_counter + 1

    records = to_record_dict(df)

    try:
        clustered = deduper.partition(records, threshold=MATCH_THRESHOLD)
    except BlockingError:
        # No candidate pairs were generated; treat every row as a unique book.
        df["book_id"] = range(global_counter, global_counter + len(df))
        return df, global_counter + len(df)

    record_to_local: dict[str, int] = {
        rid: cluster_id
        for cluster_id, (record_ids, _scores) in enumerate(clustered)
        for rid in record_ids
    }

    df["_local_id"] = [record_to_local.get(str(i), -1) for i in df.index]

    # Renumber local ids contiguously from 0.
    id_map          = {old: new for new, old in enumerate(sorted(df["_local_id"].unique()))}
    df["_local_id"] = df["_local_id"].map(id_map)

    # Post-processing: merge clusters whose titles are prefix variants.
    before = df["_local_id"].nunique()
    df     = apply_substring_merges(df)
    after  = df["_local_id"].nunique()
    if before != after:
        tqdm.write(f"    substring merge: {before} -> {after} clusters")

    # Map each local cluster id to a globally unique book_id.
    n_clusters      = df["_local_id"].nunique()
    local_to_global = {
        local_id: global_counter + i
        for i, local_id in enumerate(sorted(df["_local_id"].unique()))
    }
    df["book_id"] = df["_local_id"].map(local_to_global)
    df = df.drop(columns=["_local_id"])

    return df, global_counter + n_clusters


# ---------------------------------------------------------------------------
# Best-version selection
# ---------------------------------------------------------------------------

def assign_best_version(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``best_version`` column (1 = best, 0 = not best) per ``book_id``.

    The best version within each cluster is the row with the highest value in
    ``mean_wc_ocr``.  Ties are broken by original row order (lowest index wins,
    because ``idxmax()`` returns the first occurrence of the maximum).

    Rows with a missing or non-numeric ``mean_wc_ocr`` value are ranked below
    any row with a valid numeric value.  If *all* rows in a cluster lack a
    valid value, ``best_version`` stays 0 for every row in that cluster.

    Rows with ``book_id == -1`` (unprocessed / missing author) receive 0.
    """
    df = df.copy()
    df["best_version"] = 0

    if OCR_COL not in df.columns:
        print(
            f"Warning: column '{OCR_COL}' not found in the dataset. "
            "'best_version' will be 0 for all rows."
        )
        return df

    # Coerce to numeric; non-parseable values become NaN (ranked last by idxmax).
    ocr = pd.to_numeric(df[OCR_COL], errors="coerce")

    for book_id, group in df.groupby("book_id"):
        if book_id == -1:
            continue
        group_ocr = ocr.loc[group.index]
        if group_ocr.isna().all():
            # No valid OCR value in this cluster -- leave best_version = 0.
            continue
        best_idx = group_ocr.idxmax()   # first max occurrence wins ties
        df.at[best_idx, "best_version"] = 1

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Deduplicate a book-metadata CSV per author using a shared dedupe model. "
            "Writes a copy of the input file (with '_deduplicated' suffix) containing "
            "two new columns: 'book_id' and 'best_version'."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input",        required=True, help="Path to the input metadata CSV.")
    parser.add_argument(
        "--sample-train",
        action="store_true",
        help="Train the shared model interactively on a random sample, then exit.",
    )
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = input_path.with_name(input_path.stem + "_deduplicated.csv")

    print(f"Loading {args.input} ...")
    df = pd.read_csv(args.input, dtype=str, low_memory=False)
    print(f"  {len(df):,} rows loaded.")

    df["_author_clean"] = df[AUTHOR_COL].fillna("").str.strip()

    if args.sample_train:
        train_shared_model(df)
        return

    print("Loading shared dedupe model ...")
    deduper = load_shared_model()

    unique_authors = sorted(a for a in df["_author_clean"].unique() if a.strip())
    print(f"  {len(unique_authors):,} unique authors to process.\n")

    result_frames: list[pd.DataFrame] = []
    global_counter = 0

    for author in tqdm(unique_authors, desc="Authors"):
        author_df = df[df["_author_clean"] == author].drop(columns=["_author_clean"])
        result_df, global_counter = deduplicate_author(
            author, author_df, deduper, global_counter
        )
        tqdm.write(
            f"  {author!r}: {len(result_df)} rows -> "
            f"{result_df['book_id'].nunique()} unique books"
        )
        result_frames.append(result_df)

    # Rows with a blank/missing author are left unprocessed -- sentinel book_id = -1.
    unprocessed = df[df["_author_clean"].str.strip() == ""].drop(columns=["_author_clean"])
    if not unprocessed.empty:
        unprocessed = unprocessed.copy()
        unprocessed["book_id"] = -1
        result_frames.append(unprocessed)

    # Reassemble in original row order.
    combined = pd.concat(result_frames).sort_index()
    combined["book_id"] = pd.to_numeric(combined["book_id"], errors="coerce").astype("Int64")

    # Derive best_version from mean_wc_ocr within each book_id cluster.
    print("\nSelecting best version per book cluster ...")
    combined = assign_best_version(combined)
    combined["best_version"] = combined["best_version"].astype("Int64")

    combined.to_csv(output_path, index=False)

    n_best = int(combined["best_version"].sum())
    print(f"\nDone.")
    print(f"  Output written to : {output_path}")
    print(f"  Total rows        : {len(combined):,}")
    print(f"  Unique book_ids   : {global_counter:,}")
    print(f"  best_version = 1  : {n_best:,}")


if __name__ == "__main__":
    main()