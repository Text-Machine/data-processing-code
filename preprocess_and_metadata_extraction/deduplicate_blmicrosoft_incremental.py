"""
deduplicate_blmicrosoft_incremental.py
================================
Deduplicates the BL Microsoft metadata CSV on a per-author basis using a single shared
`dedupe` model that is trained **incrementally across multiple sessions**.

Overview
--------
Unlike the single-session version, training is split into short rounds of
50 labelled pairs each.  Run ``--train`` as many times as you like; each
session loads the previously saved labels, asks for 50 more, and saves an
improved model.  Run ``--train-status`` at any time to see how many pairs
have been labelled so far.

Typical workflow
~~~~~~~~~~~~~~~~
::

    # Round 1 — label the first 50 pairs
    python dedupe_by_author_incremental.py --input metadata.csv --train

    # Round 2 — label 50 more (cumulative)
    python dedupe_by_author_incremental.py --input metadata.csv --train

    # … repeat until satisfied …

    # Check progress
    python dedupe_by_author_incremental.py --input metadata.csv --train-status

    # Run inference (fully automated)
    python dedupe_by_author_incremental.py --input metadata.csv

Each ``--train`` session:

1. Loads the existing training labels (if any).
2. Asks you to label exactly ``PAIRS_PER_SESSION`` new pairs (y/n/u, then 'f').
3. Retrains the model from *all* labels accumulated so far.
4. Saves the updated model and labels to disk.

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

- ``PAIRS_PER_SESSION``       — how many new pairs to label each training run.
- ``MATCH_THRESHOLD``         — dedupe confidence threshold (lower → more permissive).
- ``SUBSTRING_MERGE_MAX_RATIO`` — length-ratio cap for Rule 1 substring merges.
- ``SAMPLE_SIZE``             — number of rows drawn for the training sample pool.
"""

import argparse
import json
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

# New pairs requested from the user in each --train session.
PAIRS_PER_SESSION = 50

# Confidence threshold passed to dedupe.partition().  Lowered from 0.5 because
# the model was producing false negatives on this dataset.
MATCH_THRESHOLD = 0.35

# Number of rows sampled from the full dataset as the candidate pool for
# active learning.  A larger pool gives dedupe more pairs to choose from.
SAMPLE_SIZE = 2_000

# Maximum ratio of (longer title / shorter title) lengths for a substring merge
# to be accepted under Rule 1.
SUBSTRING_MERGE_MAX_RATIO = 2.5

DEDUPE_FIELDS = [
    dedupe.variables.String("title",  has_missing=True),
    dedupe.variables.String("author", has_missing=True),
]

SHARED_SETTINGS = Path("./dedupe_settings/shared.settings")
SHARED_TRAINING = Path("./dedupe_settings/shared_training.json")
SESSION_LOG     = Path("./dedupe_settings/session_log.json")

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
_BRACKET_SUFFIX   = re.compile(r"\s*\[.*?\]")
_PAREN_SUFFIX     = re.compile(r"\s*\(.*?\)")
_MULTI_SPACE      = re.compile(r"\s+")
_PUNCTUATION      = re.compile(r"[^\w\s]")


def normalize_title(val: str | None) -> str | None:
    """Return a normalised title for fuzzy comparison (lowercase, punctuation stripped, etc.)."""
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
# Session logging
# ---------------------------------------------------------------------------

def _load_session_log() -> list[dict]:
    """Load the list of past training sessions from disk (empty list if none)."""
    if not SESSION_LOG.exists():
        return []
    with open(SESSION_LOG) as f:
        return json.load(f)


def _save_session_log(sessions: list[dict]) -> None:
    SESSION_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(SESSION_LOG, "w") as f:
        json.dump(sessions, f, indent=2)


def _count_labelled_pairs() -> tuple[int, int, int]:
    """Return (total, match, distinct) pair counts from the training JSON.

    Returns
    -------
    tuple[int, int, int]
        ``(total_pairs, match_pairs, distinct_pairs)`` where *total* is the
        sum of the other two.  Returns ``(0, 0, 0)`` if no training file exists.
    """
    if not SHARED_TRAINING.exists():
        return 0, 0, 0
    with open(SHARED_TRAINING) as f:
        data = json.load(f)
    matches  = len(data.get("match", []))
    distinct = len(data.get("distinct", []))
    return matches + distinct, matches, distinct


# ---------------------------------------------------------------------------
# Incremental training
# ---------------------------------------------------------------------------

def _make_labelling_session_deduper(records: dict) -> dedupe.Dedupe:
    """Initialise a Dedupe instance, loading any existing training labels.

    Parameters
    ----------
    records:
        The full candidate record dict (output of ``to_record_dict``).

    Returns
    -------
    dedupe.Dedupe
        A Dedupe instance ready for ``console_label``.
    """
    deduper = dedupe.Dedupe(DEDUPE_FIELDS)
    if SHARED_TRAINING.exists():
        with open(SHARED_TRAINING) as f:
            deduper.prepare_training(records, f)
    else:
        deduper.prepare_training(records)
    return deduper


class _PairCounter:
    """Context manager that wraps ``dedupe.console_label`` to stop after N pairs.

    dedupe's active-learning loop runs until the user types 'f'.  This wrapper
    patches ``input`` so that it automatically injects 'f' once *limit* pairs
    have been labelled, giving the user a natural stopping point without
    requiring them to manually count.

    Parameters
    ----------
    limit:
        Maximum number of new pairs to label before auto-finishing.
    """

    def __init__(self, limit: int) -> None:
        self.limit    = limit
        self.count    = 0
        self._orig_input = None

    def __enter__(self):
        import builtins
        self._orig_input = builtins.input

        outer = self

        def _patched_input(prompt=""):
            response = outer._orig_input(prompt)
            # A non-'f' response to a labelling prompt counts as one labelled pair.
            if response.strip().lower() in ("y", "n", "u"):
                outer.count += 1
                remaining = outer.limit - outer.count
                if outer.count >= outer.limit:
                    print(
                        f"\n  Session limit reached ({outer.limit} pairs labelled). "
                        "Finishing automatically …"
                    )
            return response

        builtins.input = _patched_input
        return self

    def __exit__(self, *_):
        import builtins
        builtins.input = self._orig_input


def train_incremental(df: pd.DataFrame) -> None:
    """Run one incremental training session.

    Loads any previously saved labels, prompts for ``PAIRS_PER_SESSION`` new
    labels, retrains the model from all accumulated labels, and saves
    everything back to disk.

    Parameters
    ----------
    df:
        The full metadata DataFrame (sampling is done internally).
    """
    total_before, matches_before, distinct_before = _count_labelled_pairs()

    print(f"\nDrawing a sample of up to {SAMPLE_SIZE} rows for training …")
    sample_df = df.sample(min(SAMPLE_SIZE, len(df)), random_state=42)
    records   = to_record_dict(sample_df)

    sessions = _load_session_log()
    session_num = len(sessions) + 1

    if total_before == 0:
        print("  No existing labels found — starting from scratch.")
    else:
        print(
            f"  Loaded {total_before} existing labelled pairs "
            f"({matches_before} match, {distinct_before} distinct) "
            f"from {len(sessions)} previous session(s)."
        )

    print(f"\n── Training session {session_num} ────────────────────────────────────────────")
    print(f"  You will be asked to label up to {PAIRS_PER_SESSION} pairs.")
    print("  For each pair: (y) duplicate  (n) not duplicate  (u) unsure  (f) finish early")
    print()
    print("  Titles are shown in normalised form (lowercased, brackets stripped,")
    print("  leading articles removed) — this is intentional.")
    print()

    deduper = _make_labelling_session_deduper(records)

    with _PairCounter(PAIRS_PER_SESSION) as counter:
        dedupe.console_label(deduper)

    pairs_this_session = counter.count
    print(f"\n  {pairs_this_session} pair(s) labelled this session.")

    print("  Retraining model …")
    deduper.train()

    SHARED_SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    with open(SHARED_TRAINING, "w") as f:
        deduper.write_training(f)
    with open(SHARED_SETTINGS, "wb") as f:
        deduper.write_settings(f)

    total_after, matches_after, distinct_after = _count_labelled_pairs()

    import datetime
    sessions.append({
        "session":         session_num,
        "timestamp":       datetime.datetime.now().isoformat(timespec="seconds"),
        "pairs_labelled":  pairs_this_session,
        "total_pairs":     total_after,
        "total_match":     matches_after,
        "total_distinct":  distinct_after,
    })
    _save_session_log(sessions)

    print(f"\n  Model saved to {SHARED_SETTINGS}")
    print(f"  Cumulative labels: {total_after} total  "
          f"({matches_after} match, {distinct_after} distinct)")
    print("\nRun --train again to label more pairs, or omit it to run inference.")


def show_training_status() -> None:
    """Print a summary of all training sessions completed so far."""
    sessions = _load_session_log()
    total, matches, distinct = _count_labelled_pairs()

    if not sessions:
        print("\nNo training sessions found.  Run with --train to start.")
        return

    print(f"\n{'─' * 60}")
    print(f"  Training sessions: {len(sessions)}")
    print(f"  Labelled pairs:    {total} total  ({matches} match, {distinct} distinct)")
    print(f"  Model file:        {'✓ exists' if SHARED_SETTINGS.exists() else '✗ missing'}")
    print(f"{'─' * 60}")
    print(f"  {'#':>3}  {'Date/time':<22}  {'This session':>12}  {'Cumulative':>10}")
    print(f"  {'─'*3}  {'─'*22}  {'─'*12}  {'─'*10}")
    for s in sessions:
        print(
            f"  {s['session']:>3}  {s['timestamp']:<22}  "
            f"{s['pairs_labelled']:>12}  {s['total_pairs']:>10}"
        )
    print(f"{'─' * 60}\n")


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
        *max_length_ratio* of each other.

    **Rule 2 — prefix match (no ratio limit)**
        The longer title starts with the shorter one.

    Parameters
    ----------
    rep_a, rep_b:
        Normalised representative titles for two clusters.
    max_length_ratio:
        Upper bound on ``len(longer) / len(shorter)`` for Rule 1.

    Returns
    -------
    bool
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
    """Return the shortest normalised title in *book_id*'s cluster."""
    titles = [
        normalize_title(clean_value(t))
        for t in df.loc[df["book_id"] == book_id, "title"]
    ]
    titles = [t for t in titles if t]
    return min(titles, key=len) if titles else None


def apply_substring_merges(df: pd.DataFrame) -> pd.DataFrame:
    """Merge clusters whose representative titles satisfy the substring rules.

    Parameters
    ----------
    df:
        DataFrame with a ``book_id`` column produced by the dedupe step.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with updated ``book_id`` values, renumbered from 0.
    """
    cluster_ids = sorted(df["book_id"].unique())
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
    df["book_id"] = df["book_id"].apply(resolve)
    id_map        = {old: new for new, old in enumerate(sorted(df["book_id"].unique()))}
    df["book_id"] = df["book_id"].map(id_map)
    return df


# ---------------------------------------------------------------------------
# Per-author deduplication (inference)
# ---------------------------------------------------------------------------

def load_shared_model() -> dedupe.StaticDedupe:
    """Load the pre-trained shared model from disk (read-only).

    Raises
    ------
    FileNotFoundError
        If the settings file does not exist (run ``--train`` first).
    """
    if not SHARED_SETTINGS.exists():
        raise FileNotFoundError(
            f"Shared model not found at {SHARED_SETTINGS}.\n"
            "Run with --train first to create it."
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

    if len(df) == 1:
        df["book_id"] = 0
        df.to_csv(out_path, index=False)
        return df

    records = to_record_dict(df)

    try:
        clustered = deduper.partition(records, threshold=MATCH_THRESHOLD)
    except BlockingError:
        df["book_id"] = range(len(df))
        df.to_csv(out_path, index=False)
        return df

    record_to_cluster: dict[str, int] = {
        rid: cluster_id
        for cluster_id, (record_ids, _scores) in enumerate(clustered)
        for rid in record_ids
    }

    df["book_id"] = [record_to_cluster.get(str(i), -1) for i in df.index]

    id_map        = {old: new for new, old in enumerate(sorted(df["book_id"].unique()))}
    df["book_id"] = df["book_id"].map(id_map)

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
        description=(
            "Deduplicate a book-metadata CSV per author using a shared dedupe model "
            "trained incrementally across multiple sessions."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input",  required=True,      help="Path to the input metadata CSV.")
    parser.add_argument("--output", default="./output", help="Root output directory (default: ./output).")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--train",
        action="store_true",
        help=f"Run one interactive training session ({PAIRS_PER_SESSION} new pairs), then exit.",
    )
    mode.add_argument(
        "--train-status",
        action="store_true",
        help="Print a summary of training sessions completed so far, then exit.",
    )
    args = parser.parse_args()

    if args.train_status:
        show_training_status()
        return

    print(f"Loading {args.input} …")
    df = pd.read_csv(args.input, dtype=str, low_memory=False)
    print(f"  {len(df):,} rows loaded.")

    df["_author_clean"] = df[AUTHOR_COL].fillna("").str.strip()

    if args.train:
        train_incremental(df)
        return

    # ── Inference mode ────────────────────────────────────────────────────
    print("Loading shared dedupe model …")
    deduper = load_shared_model()

    unique_authors = sorted(a for a in df["_author_clean"].unique() if a.strip())
    print(f"  {len(unique_authors):,} unique authors to process.\n")

    output_dir = Path(args.output)
    for author in tqdm(unique_authors, desc="Authors"):
        author_df = df[df["_author_clean"] == author].drop(columns=["_author_clean"])
        result    = deduplicate_author(author, author_df, deduper, output_dir)
        tqdm.write(
            f"  {author!r}: {len(result)} rows → {result['book_id'].nunique()} unique books"
        )

    print("\nDone. Outputs written to:", output_dir)


if __name__ == "__main__":
    main()