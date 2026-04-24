"""
deduplicate_blmicrosoft.py
===================
Deduplicates the BL Microsoft metadata CSV on a per-author basis using a single shared
`dedupe` model trained once and reused for every author.

Overview
--------
The script runs in two distinct phases:

**Phase 1 — Training** (interactive, ~10-15 minutes, run once)::

    python deduplicate_blmicrosoft.py --input metadata.csv --sample-train

A within-author sample is drawn (so labelled pairs are representative of
inference), and `dedupe` asks you to label ~10-20 record pairs as duplicates
or non-duplicates (y/n/u, then 'f' to finish).
The trained model is saved to ``./dedupe_settings/shared.settings`` and the
labelled pairs to ``./dedupe_settings/shared_training.json``.  You never
need to repeat this step.

**Phase 2 — Inference** (fully automated, no user interaction)::

    python deduplicate_blmicrosoft.py --input metadata.csv

The saved model is loaded and applied to every author's subset of rows.
Results are written to ``./output/<author_slug>/deduplicated.csv``, with a
``book_id`` column that groups duplicate records under the same integer id,
and a ``book_id_score`` column with the mean dedupe confidence for each cluster.

Main Changes from previous version
------------------------------
- Removed substring-merge post-processing.  
- Training sample is drawn from the ``TOP_AUTHORS_FOR_TRAINING`` most
  prolific authors (by record count), capped at ``SAMPLE_SIZE`` rows.
- ``MATCH_THRESHOLD`` raised from 0.35 → 0.5 
- Added ``book_id_score`` column (mean cluster confidence) to help audit
  borderline matches.

Dependencies
------------
::

    pip install dedupe pandas tqdm

Configuration
-------------
The key tunables at the top of this file are:

- ``MATCH_THRESHOLD`` — dedupe confidence threshold (higher → more conservative).
- ``SAMPLE_SIZE`` — max rows drawn for the training sample (default: 150).
- ``TOP_AUTHORS_FOR_TRAINING`` — how many of the most prolific authors to
  draw the sample from (default: 20).
- ``OPTIONAL_FIELDS`` — extra columns to use as dedupe signals if present.
  Do NOT add ``date`` or ``edition`` here — editions are duplicates for us.
"""

import argparse
import logging
import re
import unicodedata
from pathlib import Path

import dedupe
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

AUTHOR_COL = "author"

# Confidence threshold passed to dedupe.partition().
# Raised from 0.35 — no post-processing safety net means we want fewer false
# positives even at the cost of missing some true duplicates.
MATCH_THRESHOLD = 0.5

# Number of rows drawn from the top authors for interactive training.
SAMPLE_SIZE = 150

# How many of the most prolific authors (by record count) to draw the
# training sample from.  Prolific authors have the richest within-author
# title variation and give the labeller the most informative pairs.
TOP_AUTHORS_FOR_TRAINING = 20

# Extra columns to use as dedupe signals if present in the input CSV.
# Each entry is (column_name, dedupe_variable_class, kwargs).
#
# NOTE: ``date`` and ``edition`` are deliberately NOT listed here.
# Different editions of the same work are duplicates for our purposes, so
# we must not give the model any signal that would cause it to split them.
OPTIONAL_FIELDS: list[tuple[str, type, dict]] = [
    ("place", dedupe.variables.String, {"has_missing": True}),
]

SHARED_SETTINGS = Path("./dedupe_settings/shared.settings")
SHARED_TRAINING = Path("./dedupe_settings/shared_training.json")

logging.basicConfig(level=logging.WARNING)


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def slugify(text: str) -> str:
    """Convert *text* to a filesystem-safe ASCII slug (max 80 chars)."""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"\s+", "_", text)[:80]


def clean_value(val) -> str | None:
    """Return *val* as a stripped string, or ``None`` if blank/NaN."""
    if pd.isna(val):
        return None
    return str(val).strip() or None


# ---------------------------------------------------------------------------
# Title normalisation
# ---------------------------------------------------------------------------

_LEADING_ARTICLES = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)
_BRACKET_SUFFIX   = re.compile(r"\s*\[.*?\]")   # e.g. "[With plates.]"
_PAREN_SUFFIX     = re.compile(r"\s*\(.*?\)")    # e.g. "(With illustrations)"
_MULTI_SPACE      = re.compile(r"\s+")
_PUNCTUATION      = re.compile(r"[^\w\s]")


def normalize_title(val: str | None) -> str | None:
    """Return a normalised version of *val* suitable for fuzzy comparison.

    Transformations applied (in order):
    - lowercase
    - strip bracketed / parenthesised suffixes
    - remove leading articles (the, a, an)
    - remove non-word punctuation
    - collapse whitespace

    Returns ``None`` if the result is an empty string.

    Note: normalisation is applied only to dedupe's internal comparison.
    The CSV on disk always retains the original title text.
    """
    if val is None:
        return None
    val = val.lower()
    val = _BRACKET_SUFFIX.sub("", val)
    val = _PAREN_SUFFIX.sub("", val)
    val = _LEADING_ARTICLES.sub("", val)
    val = _PUNCTUATION.sub("", val)
    val = _MULTI_SPACE.sub(" ", val).strip()
    return val or None


def build_fields(available_columns: set[str]) -> list:
    """Build the list of dedupe fields based on what columns are in the CSV.

    Always includes ``title`` and ``author``.  Adds optional fields from
    ``OPTIONAL_FIELDS`` only when their column is present in the dataset.

    ``date`` and ``edition`` are intentionally excluded: different editions
    of the same work are considered duplicates, so the model must not use
    year or edition number as a splitting signal.
    """
    fields: list = [
        dedupe.variables.String("title",  has_missing=True),
        dedupe.variables.String("author", has_missing=True),
    ]
    for col, cls, kwargs in OPTIONAL_FIELDS:
        if col in available_columns:
            fields.append(cls(col, **kwargs))
            print(f"  + using optional field: '{col}' ({cls.__name__})")
    return fields


def to_record_dict(df: pd.DataFrame, use_cols: list[str]) -> dict[str, dict]:
    """Convert a DataFrame to the ``{record_id: fields}`` format dedupe expects.

    Titles are normalised; all other fields are cleaned but not normalised.

    Parameters
    ----------
    df:
        DataFrame slice for one author (or the training sample).
    use_cols:
        List of column names to include (must all be present in *df*).
    """
    records = {}
    for idx, row in df.iterrows():
        record: dict[str, str | None] = {}
        for col in use_cols:
            val = clean_value(row.get(col))
            record[col] = normalize_title(val) if col == "title" else val
        records[str(idx)] = record
    return records


# ---------------------------------------------------------------------------
# Training  — within-author sample
# ---------------------------------------------------------------------------

def _top_author_sample(
    df: pd.DataFrame,
    n: int,
    top_k: int = TOP_AUTHORS_FOR_TRAINING,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """Draw up to *n* rows from the *top_k* most prolific authors.

    Prolific authors have the richest within-author title variation (many
    editions, collected works, omnibus volumes, etc.), so their records
    produce the most informative pairs for the active-learning labeller.
    Only authors with ≥ 2 records are considered.

    Parameters
    ----------
    df:
        Full metadata DataFrame (must have ``_author_clean`` column).
    n:
        Maximum number of rows to return.
    top_k:
        How many of the most prolific authors to draw from.
    rng_seed:
        Random seed for reproducibility.
    """
    # Exclude blank/missing authors before counting.
    named    = df[df["_author_clean"].str.strip() != ""]
    counts   = named["_author_clean"].value_counts()
    top_auth = counts[counts >= 2].head(top_k).index
    eligible = named[named["_author_clean"].isin(top_auth)]
    return eligible.sample(min(n, len(eligible)), random_state=rng_seed)


def train_shared_model(df: pd.DataFrame, available_columns: set[str]) -> None:
    """Interactively train the shared dedupe model and persist it to disk.

    Parameters
    ----------
    df:
        The full metadata DataFrame.
    available_columns:
        Set of column names present in *df* (used to select dedupe fields).
    """
    print(
        f"\nDrawing up to {SAMPLE_SIZE} rows from the "
        f"top {TOP_AUTHORS_FOR_TRAINING} most prolific authors ..."
    )
    sample_df    = _top_author_sample(df, SAMPLE_SIZE)
    sampled_auth = sample_df["_author_clean"].value_counts()
    print(f"  Authors in sample ({len(sampled_auth)}):")
    for auth, cnt in sampled_auth.items():
        print(f"    {auth!r}: {cnt} rows")
    fields    = build_fields(available_columns)
    use_cols  = [f.field for f in fields]
    records   = to_record_dict(sample_df, use_cols)

    deduper = dedupe.Dedupe(fields)

    if SHARED_TRAINING.exists():
        print(f"Found existing training data at {SHARED_TRAINING} — loading it.")
        with open(SHARED_TRAINING) as f:
            deduper.prepare_training(records, f)
    else:
        deduper.prepare_training(records)

    print("\n-- Active learning ---------------------------------------------------")
    print("  For each pair: (y) duplicate  (n) not duplicate  (u) unsure  (f) finished")
    print("  Aim for ~10-20 labelled pairs, then press 'f'.")
    print()
    print("  Note: titles are shown in normalised form (lowercased, brackets")
    print("  stripped, leading articles removed) — this is intentional.")
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
    """Load the pre-trained shared model from disk (read-only).

    Raises
    ------
    FileNotFoundError
        If the settings file does not exist (run ``--sample-train`` first).
    """
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
    use_cols: list[str],
    output_dir: Path,
) -> pd.DataFrame:
    """Apply the shared model to one author's rows and write the result to disk.

    Each row receives a ``book_id`` integer identifying its duplicate cluster,
    and a ``book_id_score`` float with the mean dedupe confidence for that
    cluster (useful for auditing borderline matches).

    Parameters
    ----------
    author_name:
        Display name used to construct the output subdirectory slug.
    df:
        Subset of the metadata DataFrame for this author only.
    deduper:
        Pre-loaded ``StaticDedupe`` model (shared across all authors).
    use_cols:
        Ordered list of field names the model was trained on.
    output_dir:
        Root output directory; results go to ``output_dir/<slug>/deduplicated.csv``.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with ``book_id`` and ``book_id_score`` columns added.
    """
    from dedupe.core import BlockingError

    slug       = slugify(author_name)
    author_out = output_dir / slug
    author_out.mkdir(parents=True, exist_ok=True)
    out_path   = author_out / "deduplicated.csv"

    df = df.copy().reset_index(drop=True)

    # Trivial case: a single row is its own unique book.
    if len(df) == 1:
        df["book_id"]       = 0
        df["book_id_score"] = 1.0
        df.to_csv(out_path, index=False)
        return df

    records = to_record_dict(df, use_cols)

    try:
        clustered = list(deduper.partition(records, threshold=MATCH_THRESHOLD))
    except BlockingError:
        # No candidate pairs were generated; treat every row as unique.
        df["book_id"]       = range(len(df))
        df["book_id_score"] = 1.0
        df.to_csv(out_path, index=False)
        return df

    # Build lookup: record_id → (cluster_id, mean_score)
    record_to_cluster: dict[str, int]   = {}
    record_to_score:   dict[str, float] = {}
    for cluster_id, (record_ids, scores) in enumerate(clustered):
        mean_score = float(sum(scores) / len(scores)) if len(scores) > 0 else 1.0
        for rid in record_ids:
            record_to_cluster[rid] = cluster_id
            record_to_score[rid]   = mean_score

    df["book_id"]       = [record_to_cluster.get(str(i), -1) for i in df.index]
    df["book_id_score"] = [record_to_score.get(str(i),   1.0) for i in df.index]

    # Renumber book_id contiguously from 0.
    id_map        = {old: new for new, old in enumerate(sorted(df["book_id"].unique()))}
    df["book_id"] = df["book_id"].map(id_map)

    df.to_csv(out_path, index=False)
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deduplicate a book-metadata CSV per author using a shared dedupe model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input",        required=True,      help="Path to the input metadata CSV.")
    parser.add_argument("--output",       default="./output", help="Root output directory (default: ./output).")
    parser.add_argument(
        "--sample-train",
        action="store_true",
        help="Train the shared model interactively on a within-author sample, then exit.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)

    print(f"Loading {args.input} ...")
    df = pd.read_csv(args.input, dtype=str, low_memory=False)
    print(f"  {len(df):,} rows loaded.")

    df["_author_clean"] = df[AUTHOR_COL].fillna("").str.strip()

    available_columns = set(df.columns) - {"_author_clean"}

    if args.sample_train:
        train_shared_model(df, available_columns)
        return

    print("Loading shared dedupe model ...")
    deduper  = load_shared_model()
    # Derive use_cols from the same logic used at training time.
    # Reconstructing from the model object is fragile across dedupe versions.
    fields   = build_fields(available_columns)
    use_cols = [f.field for f in fields]

    unique_authors = sorted(a for a in df["_author_clean"].unique() if a.strip())
    print(f"  {len(unique_authors):,} unique authors to process.\n")

    for author in tqdm(unique_authors, desc="Authors"):
        author_df = df[df["_author_clean"] == author].drop(columns=["_author_clean"])
        result    = deduplicate_author(author, author_df, deduper, use_cols, output_dir)
        tqdm.write(
            f"  {author!r}: {len(result)} rows → {result['book_id'].nunique()} unique books"
        )

    print("\nDone. Outputs written to:", output_dir)


if __name__ == "__main__":
    main()