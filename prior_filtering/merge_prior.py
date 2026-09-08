#!/usr/bin/env python3
"""
merge_prior_v3.py — Step 2 of the prior-filtering pipeline.

Loads the decade-level partial pallets created by build_prior_v3.py, sums
their TF and DF counts, computes the smoothed normalized prior, merges the
QA reservoirs, computes corpus-level mu/sigma statistics, and writes the
final prior consumed by apply_prior.py.
"""

import argparse
import math
import random
from pathlib import Path

import torch


# =============================================================================
# CONFIG
# =============================================================================

PARTIAL_DIR = Path("prior_filter/blmicrosoft/partials")
OUTPUT_PRIOR_PATH = Path("prior_filter/blmicrosoft/all_prior_tfdf_v3.pt")

TOKENIZER_NAME = "gpt2-large"
SMOOTHING_EPS = 1e-6

QA_SAMPLE_SIZE = 5000
RANDOM_STATE = 42

DECADES = [
    "1800_1809",
    "1810_1819",
    "1820_1829",
    "1830_1839",
    "1840_1849",
    "1850_1859",
    "1860_1869",
    "1870_1879",
    "1880_1889",
    "1890_1899",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Merge available partials even if some decades are missing.",
    )
    return parser.parse_args()


def load_partials(allow_missing=False):
    partials = []
    missing = []

    for decade in DECADES:
        path = PARTIAL_DIR / f"pallet_{decade}.pt"
        if not path.exists():
            missing.append(decade)
            continue

        payload = torch.load(path, map_location="cpu", weights_only=False)
        partials.append((decade, path, payload))

    if missing and not allow_missing:
        raise SystemExit(
            "Missing partial pallet(s): "
            + ", ".join(missing)
            + "\nWait for all array tasks to finish, or use --allow-missing."
        )

    if not partials:
        raise SystemExit("No partial pallets found.")

    if missing:
        print("[warn] merging without: " + ", ".join(missing))

    return partials


def merge_qa_reservoirs(partials):
    """
    Merge per-decade reservoirs into an approximately uniform corpus-level
    reservoir.

    Each decade reservoir is a uniform sample of that decade's pages. We
    allocate the final QA sample proportionally to each decade's page count,
    capped by the size of that decade's saved reservoir.
    """
    total_pages = sum(payload["n_pages"] for _, _, payload in partials)

    if total_pages == 0:
        return []

    rng = random.Random(RANDOM_STATE)

    entries = []
    capacities = {}
    allocations = {}

    for decade, _, payload in partials:
        reservoir = payload.get("qa_reservoir", [])
        n_pages = int(payload["n_pages"])
        capacities[decade] = len(reservoir)

        if not reservoir or n_pages == 0:
            allocations[decade] = 0
            continue

        raw = QA_SAMPLE_SIZE * (n_pages / total_pages)
        allocations[decade] = min(len(reservoir), int(math.floor(raw)))
        entries.append((decade, reservoir, raw))

    # Distribute any remaining slots by largest fractional remainder, while
    # respecting each decade reservoir's capacity.
    current = sum(allocations.values())
    remaining_slots = min(QA_SAMPLE_SIZE, sum(capacities.values())) - current

    ranked = []
    for decade, _, raw in entries:
        fractional = raw - math.floor(raw)
        spare = capacities[decade] - allocations[decade]
        if spare > 0:
            ranked.append((fractional, n_pages_for(partials, decade), decade))

    ranked.sort(reverse=True)

    while remaining_slots > 0 and ranked:
        progressed = False
        for _, _, decade in ranked:
            if allocations[decade] < capacities[decade]:
                allocations[decade] += 1
                remaining_slots -= 1
                progressed = True
                if remaining_slots == 0:
                    break
        if not progressed:
            break

    merged = []
    for decade, reservoir, _ in entries:
        k = allocations[decade]
        if k <= 0:
            continue
        if k >= len(reservoir):
            chosen = list(reservoir)
        else:
            chosen = rng.sample(reservoir, k)
        merged.extend(chosen)

    rng.shuffle(merged)
    return merged[:QA_SAMPLE_SIZE]


def n_pages_for(partials, decade):
    for d, _, payload in partials:
        if d == decade:
            return int(payload["n_pages"])
    return 0

def load_tokenizer():
    import transformers

    tok = transformers.GPT2TokenizerFast.from_pretrained(TOKENIZER_NAME)
    tok.model_max_length = int(1e12)
    return tok


def compute_reference_stats_with_tokenizer(prior, qa_reservoir, tok):
    if not qa_reservoir:
        raise SystemExit(
            "No valid pages in merged QA reservoir — cannot compute corpus stats."
        )

    mus = []
    sigmas = []

    # Batch the QA pass as well; it is small, but this keeps the tokenizer API
    # consistent with the main build.
    batch_size = 64

    for start in range(0, len(qa_reservoir), batch_size):
        chunk = qa_reservoir[start:start + batch_size]
        texts = [text for text, _ in chunk]

        encoded = tok(
            texts,
            padding=False,
            truncation=False,
            add_special_tokens=True,
        )

        for ids in encoded["input_ids"]:
            if not ids:
                continue

            x = torch.tensor(ids, dtype=torch.long)
            token_priors = prior[x]
            mu_d = token_priors.log().mean().item()
            sigma_d = (
                (token_priors * 1000).std().item()
                if len(ids) > 1
                else 0.0
            )
            mus.append(mu_d)
            sigmas.append(sigma_d)

    if not mus:
        raise SystemExit("No valid pages in merged QA reservoir.")

    mus_t = torch.tensor(mus)
    sigmas_t = torch.tensor(sigmas)

    return (
        mus_t.mean().item(),
        mus_t.std().item(),
        sigmas_t.mean().item(),
        sigmas_t.std().item(),
    )


def main():
    args = parse_args()
    partials = load_partials(allow_missing=args.allow_missing)

    print("=" * 72)
    print(f"Merging {len(partials)} decade partial(s)")
    print("=" * 72)

    # Validate tokenizer/vocabulary consistency before adding anything.
    vocab_size = None
    tokenizer_name = None

    for decade, path, payload in partials:
        this_vocab = int(payload["vocab_size"])
        this_tokenizer = payload["tokenizer_name"]

        if vocab_size is None:
            vocab_size = this_vocab
            tokenizer_name = this_tokenizer
        else:
            if this_vocab != vocab_size:
                raise SystemExit(
                    f"Vocab-size mismatch: {decade} has {this_vocab}, "
                    f"expected {vocab_size}."
                )
            if this_tokenizer != tokenizer_name:
                raise SystemExit(
                    f"Tokenizer mismatch: {decade} has {this_tokenizer}, "
                    f"expected {tokenizer_name}."
                )

        if payload["pallet"].shape != (2, vocab_size):
            raise SystemExit(
                f"Unexpected pallet shape in {path}: "
                f"{payload['pallet'].shape}"
            )

    merged_pallet = torch.zeros((2, vocab_size), dtype=torch.long)

    total_pages = 0
    total_books = 0

    for decade, path, payload in partials:
        merged_pallet += payload["pallet"].to(dtype=torch.long)
        total_pages += int(payload["n_pages"])
        total_books += int(payload["n_books"])
        print(
            f"  {decade}: {payload['n_pages']:,} pages, "
            f"{payload['n_books']:,} books"
        )

    print(
        f"\nMerged totals: {total_pages:,} pages across "
        f"{total_books:,} books."
    )

    tf = merged_pallet[0]
    df_tensor = merged_pallet[1]
    weighted = tf * df_tensor

    # Laplace smoothing: guarantees prior > 0 everywhere.
    prior = (weighted + SMOOTHING_EPS) / (
        weighted.sum() + SMOOTHING_EPS * vocab_size
    )

    qa_reservoir = merge_qa_reservoirs(partials)
    print(f"Merged QA reservoir: {len(qa_reservoir):,} pages")

    tok = load_tokenizer()
    mu_avg, mu_std, sigma_avg, sigma_std = (
        compute_reference_stats_with_tokenizer(
            prior,
            qa_reservoir,
            tok,
        )
    )

    print(f"  mu_d:    mean={mu_avg:.4f}  std={mu_std:.4f}")
    print(f"  sigma_d: mean={sigma_avg:.4f}  std={sigma_std:.4f}")

    OUTPUT_PRIOR_PATH.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "prior": prior,
            "vocab_size": vocab_size,
            "tokenizer_name": tokenizer_name,
            "n_pages": total_pages,
            "n_books": total_books,
            "n_decades": len(partials),
            "decades": [decade for decade, _, _ in partials],
            "mu_avg": mu_avg,
            "mu_std": mu_std,
            "sigma_avg": sigma_avg,
            "sigma_std": sigma_std,
        },
        OUTPUT_PRIOR_PATH,
    )

    print(f"\nSaved final prior → {OUTPUT_PRIOR_PATH}")
    print(
        f"  built from {total_pages:,} pages across "
        f"{total_books:,} books"
    )
    print(
        f"  nonzero prior entries: "
        f"{(prior > 0).sum().item():,} / {vocab_size:,}"
    )


if __name__ == "__main__":
    main()
