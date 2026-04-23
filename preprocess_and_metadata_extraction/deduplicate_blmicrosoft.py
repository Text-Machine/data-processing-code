"""
deduplicate_blmicrosoft.py
===================
Deduplicates the BL Microsoft metadata CSV on a per-author basis using a single shared
`dedupe` model trained once and reused for every author.

Overview
--------
The script runs in two distinct phases:

**Phase 1 — Training** (interactive, ~10-15 minutes, run once)::

    python dedupe_by_author.py --input metadata.csv --sample-train

A random cross-author sample is drawn, and `dedupe` asks you to label ~10-20
record pairs as duplicates or non-duplicates (y/n/u, then 'f' to finish).
The trained model is saved to ``./dedupe_settings/shared_v3.settings`` and the
labelled pairs to ``./dedupe_settings/shared_training_v3.json``.  You never
need to repeat this step.

**Phase 2 — Inference** (fully automated, no user interaction)::

    python dedupe_by_author.py --input metadata.csv

The saved model is loaded and applied to every author's subset of rows.
Results are written to ``./output/<author_slug>/deduplicated.csv``, with a
``book_id`` column that groups duplicate records under the same integer id.

Post-processing
---------------
After the ML model clusters records, a deterministic substring-merge pass
catches title-prefix variants that the model tends to miss, e.g.:

- "Oliver Twist"  ↔  "The Adventures of Oliver Twist"
- "Barnaby Rudge" ↔  "Barnaby Rudge: A Tale of the Riots of Eighty"

See ``should_merge_by_substring()`` for the merging rules.

Dependencies
------------
::

    pip install dedupe pandas tqdm

Configuration
-------------
The key tunables at the top of this file are:

- ``MATCH_THRESHOLD`` — dedupe confidence threshold (lower → more permissive).
- ``SUBSTRING_MERGE_MAX_RATIO`` — length-ratio cap for Rule 1 substring merges.
- ``SAMPLE_SIZE`` — number of rows drawn for the training sample.
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

# Confidence threshold passed to dedupe.partition().  Lowered from 0.5 because
# the model was producing false negatives on this dataset.
MATCH_THRESHOLD = 0.35

# Number of rows sampled from the full dataset for interactive training.
SAMPLE_SIZE = 2_000

# Maximum ratio of (longer title / shorter title) lengths for a substring merge
# to be accepted under Rule 1.  Example at 2.5:
#   "oliver twist" (13) vs "adventures of oliver twist" (26) → ratio 2.0  ✓
#   "works of charles dickens" (24) vs "selections from the works …" (46+) → ratio >2.5  ✗
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

def slugify(text: str) -> str:
    """Convert *text* to a filesystem-safe ASCII slug (max 80 chars).

    Normalises Unicode, strips non-alphanumeric characters, lowercases, and
    replaces whitespace runs with underscores.
    """
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
# Applied only to the record dicts fed to dedupe; the CSV written to disk
# always retains the original title text so no information is lost.

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


def to_record_dict(df: pd.DataFrame) -> dict[str, dict]:
    """Convert a DataFrame to the ``{record_id: fields}`` format dedupe expects.

    Titles are normalised before comparison so case variants and edition
    suffixes do not prevent matching.  Authors are left as-is because they
    are consistently formatted in this dataset.

    Parameters
    ----------
    df:
        DataFrame containing at least ``title`` and ``author`` columns.

    Returns
    -------
    dict
        Mapping of ``str(row_index)`` → ``{"title": ..., "author": ...}``.
    """
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
    """Decide whether two normalised titles likely refer to the same work.

    Two complementary rules are applied:

    **Rule 1 — substring with ratio guard**
        One title is a substring of the other AND their lengths are within
        *max_length_ratio* of each other.  The ratio cap prevents short,
        generic strings (e.g. "works of charles dickens") from absorbing
        unrelated longer titles.

    **Rule 2 — prefix match (no ratio limit)**
        The longer title *starts with* the shorter one.  Catches edition
        subtitles appended after the canonical title, e.g.:

        - "oliver twist"  ↔  "oliver twist with eight illustrations by …"
        - "dombey and son" ↔  "dombey and son with illustrations by …"

        A prefix match is directional so the false-positive risk is low —
        "works of charles dickens" does *not* start with
        "selections from the works of charles dickens".

    Parameters
    ----------
    rep_a, rep_b:
        Normalised representative titles for two clusters.
    max_length_ratio:
        Upper bound on ``len(longer) / len(shorter)`` for Rule 1.

    Returns
    -------
    bool
        ``True`` if the titles should be merged into one cluster.
    """
    if not rep_a or not rep_b:
        return False

    shorter, longer = sorted([rep_a, rep_b], key=len)

    if shorter in longer and len(longer) / len(shorter) <= max_length_ratio:
        return True

    if longer.startswith(shorter):
        return True

    return False


def _representative_title(df: pd.DataFrame, book_id: int) -> str | None:
    """Return the shortest normalised title in *book_id*'s cluster.

    The shortest title is treated as the most canonical form (e.g. prefer
    "Oliver Twist" over "The Adventures of Oliver Twist").
    """
    titles = [
        normalize_title(clean_value(t))
        for t in df.loc[df["book_id"] == book_id, "title"]
    ]
    titles = [t for t in titles if t]
    return min(titles, key=len) if titles else None


def apply_substring_merges(df: pd.DataFrame) -> pd.DataFrame:
    """Merge clusters whose representative titles satisfy the substring rules.

    This post-processing pass catches title-prefix variants that the dedupe
    model misses, then renumbers ``book_id`` contiguously from 0.

    Parameters
    ----------
    df:
        DataFrame with a ``book_id`` column produced by the dedupe step.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with updated ``book_id`` values.
    """
    cluster_ids = sorted(df["book_id"].unique())
    book_rep    = {cid: _representative_title(df, cid) for cid in cluster_ids}

    # Build a map from each cluster id to the canonical id it should merge into.
    merges: dict[int, int] = {}
    for i, id_a in enumerate(cluster_ids):
        for id_b in cluster_ids[i + 1:]:
            if should_merge_by_substring(book_rep.get(id_a), book_rep.get(id_b)):
                canonical, other = min(id_a, id_b), max(id_a, id_b)
                merges[other] = canonical

    def resolve(bid: int) -> int:
        """Follow merge chains to find the ultimate canonical id."""
        while bid in merges:
            bid = merges[bid]
        return bid

    df = df.copy()
    df["book_id"] = df["book_id"].apply(resolve)

    id_map        = {old: new for new, old in enumerate(sorted(df["book_id"].unique()))}
    df["book_id"] = df["book_id"].map(id_map)

    return df


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_shared_model(df: pd.DataFrame) -> None:
    """Interactively train the shared dedupe model and persist it to disk.

    Draws a random sample from *df*, launches the dedupe active-learning
    console, and saves the resulting model and training labels so they can
    be reused without re-labelling.

    Parameters
    ----------
    df:
        The full metadata DataFrame (sampling is done internally).
    """
    print(f"\nDrawing a sample of up to {SAMPLE_SIZE} rows for training ...")
    sample_df = df.sample(min(SAMPLE_SIZE, len(df)), random_state=42)
    records   = to_record_dict(sample_df)

    deduper = dedupe.Dedupe(DEDUPE_FIELDS)

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

    Returns
    -------
    dedupe.StaticDedupe

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
    output_dir: Path,
) -> pd.DataFrame:
    """Apply the shared model to one author's rows and write the result to disk.

    Each row receives a ``book_id`` integer that identifies its duplicate
    cluster.  Records with the same ``book_id`` are considered editions or
    copies of the same work.

    Parameters
    ----------
    author_name:
        Display name used to construct the output subdirectory slug.
    df:
        Subset of the metadata DataFrame for this author only.
    deduper:
        Pre-loaded ``StaticDedupe`` model (shared across all authors).
    output_dir:
        Root output directory; results go to ``output_dir/<slug>/deduplicated.csv``.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with a ``book_id`` column added.
    """
    from dedupe.core import BlockingError

    slug       = slugify(author_name)
    author_out = output_dir / slug
    author_out.mkdir(parents=True, exist_ok=True)
    out_path   = author_out / "deduplicated.csv"

    df = df.copy().reset_index(drop=True)

    # Trivial case: a single row is its own unique book.
    if len(df) == 1:
        df["book_id"] = 0
        df.to_csv(out_path, index=False)
        return df

    records = to_record_dict(df)

    try:
        clustered = deduper.partition(records, threshold=MATCH_THRESHOLD)
    except BlockingError:
        # No candidate pairs were generated; treat every row as unique.
        df["book_id"] = range(len(df))
        df.to_csv(out_path, index=False)
        return df

    record_to_cluster: dict[str, int] = {
        rid: cluster_id
        for cluster_id, (record_ids, _scores) in enumerate(clustered)
        for rid in record_ids
    }

    df["book_id"] = [record_to_cluster.get(str(i), -1) for i in df.index]

    # Renumber contiguously from 0.
    id_map        = {old: new for new, old in enumerate(sorted(df["book_id"].unique()))}
    df["book_id"] = df["book_id"].map(id_map)

    # Post-processing: merge clusters whose titles are prefix variants.
    before = df["book_id"].nunique()
    df     = apply_substring_merges(df)
    after  = df["book_id"].nunique()
    if before != after:
        tqdm.write(f"    substring merge: {before} → {after} clusters")

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
        help="Train the shared model interactively on a random sample, then exit.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)

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

    for author in tqdm(unique_authors, desc="Authors"):
        author_df = df[df["_author_clean"] == author].drop(columns=["_author_clean"])
        result    = deduplicate_author(author, author_df, deduper, output_dir)
        tqdm.write(
            f"  {author!r}: {len(result)} rows → {result['book_id'].nunique()} unique books"
        )

    print("\nDone. Outputs written to:", output_dir)


if __name__ == "__main__":
    main()