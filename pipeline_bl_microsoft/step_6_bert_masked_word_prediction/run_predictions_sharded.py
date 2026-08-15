"""
BERT Masked Word Prediction Script -- ROW-SHARDED variant
=====================================================================
BL Microsoft only. Same engine as run_predictions_faster_multigpu.py,
but addresses one gap in that script: file_queue there hands out whole
FILES as atomic units, so if you request 4 GPUs and only 1 file remains
(e.g. the tail of a run, or a run with fewer words than GPUs), 3 GPUs
sit idle for the entire duration of that last file -- the dynamic
queue only balances load ACROSS files, not within one.

This version fixes that by splitting each file into byte-range shards
BEFORE building the task queue, sized so that:

    total_shard_tasks >= n_gpus_requested

via:

    shards_per_file = ceil(n_gpus_requested * BERT_SHARD_MULTIPLIER / n_files)

Examples (4 GPUs requested):
    1 file  -> shards_per_file = 4  -> that one file split 4 ways, full
               parallelism from the start.
    2 files -> shards_per_file = 2  -> 4 shard-tasks total, one per GPU.
    7 files -> shards_per_file = 1  -> no splitting, already >= 4 tasks;
               behaves identically to the non-sharded script.

Shards are exact byte ranges (computed via a fast single pass recording
line-start byte offsets), so no line is ever split across two shards
regardless of encoding. Each shard is processed independently and
writes to its own file under {OUTPUT_DIR}/_shards/; after all workers
finish, main() concatenates a word's shards back into the single final
output file in shard order, so the final artifact is identical in shape
to what the non-sharded script produces.

Trade-offs vs. the non-sharded script:
  - Resume/skip logic is coarser: a ".done" marker file next to the
    final merged output is the only skip signal (no per-row expected-
    count comparison). If a run is interrupted mid-shard, the word is
    simply reprocessed from scratch on the next run -- but since shards
    are smaller than whole files, less work is lost per interruption,
    which is arguably a nice side effect rather than a downside.
  - Each file is read once (for the offset pass) plus once again per
    shard (each worker seeks + reads only its own byte range) -- more
    total I/O than reading the file once, but on GPFS this is cheap
    relative to GPU inference time, and only kicks in when n_files <
    n_gpus_requested in the first place (i.e. exactly the case where
    idle GPU time was the bigger cost).
  - dropped_by_filter sidecars are NOT merged across shards (left as
    separate per-shard files under _shards/ for inspection) -- they're
    diagnostic only, not needed downstream.
"""

import json
import logging
import logging.handlers
import math
import multiprocessing as mp
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

# -------------------------------------------------------------------
# Batch size / dtype
# -------------------------------------------------------------------

BATCH_SIZE = int(os.environ.get("BERT_BATCH_SIZE", "256"))

_DTYPE_MAP = {
    "float16":  torch.float16,
    "fp16":     torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16":     torch.bfloat16,
    "float32":  torch.float32,
    "fp32":     torch.float32,
}
MODEL_DTYPE = _DTYPE_MAP[os.environ.get("BERT_DTYPE", "float16").lower()]

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------

MODELS_BASE = os.environ.get(
    "BERT_MODELS_BASE",
    "/gpfs/projects/bsc100/models/bert_textmachine",
)

MODELS = {
    "pred_bert_1760_1850":    f"{MODELS_BASE}/bert_1760_1850",
    "pred_bert_1890_1900":    f"{MODELS_BASE}/bert_1890_1900",
    "pred_bert_contemporary": f"{MODELS_BASE}/bert-base-uncased",
    "pred_bert_1760_1900":    f"{MODELS_BASE}/bert_1760_1900",
}

INPUT_DIR = os.environ.get(
    "BERT_INPUT_DIR",
    "/gpfs/projects/bsc100/textmachine-data/preprocessed_data/consolidated_metadata",
)
OUTPUT_DIR = os.environ.get(
    "BERT_OUTPUT_DIR",
    "/gpfs/projects/bsc100/textmachine-data/filtered_data_predictions_batch64",
)

SUFFIX = os.environ.get("BERT_SUFFIX", "spacy")

# Sharding controls
SHARD_MULTIPLIER = float(os.environ.get("BERT_SHARD_MULTIPLIER", "1"))
CLEANUP_SHARDS = os.environ.get("BERT_CLEANUP_SHARDS", "0") == "1"

# -------------------------------------------------------------------
# Query-word resolution: env override > query_word_config.sh > hardcoded
# -------------------------------------------------------------------

QUERY_CONFIG = Path(os.environ.get("BERT_QUERY_CONFIG", "query_word_config.sh"))


def load_query_words_from_shell(config_path: Path) -> list[str]:
    if not config_path.exists():
        return []
    array_cmd = [
        "bash", "-c",
        f'source "{config_path}" && printf "%s\\n" "${{QUERY_WORDS[@]}}"',
    ]
    result = subprocess.run(array_cmd, capture_output=True, text=True)
    words = [w.strip() for w in result.stdout.splitlines() if w.strip()]
    if words:
        return words
    single_cmd = [
        "bash", "-c",
        f'source "{config_path}" && printf "%s" "$QUERY_WORD"',
    ]
    result = subprocess.run(single_cmd, capture_output=True, text=True)
    single = result.stdout.strip()
    return [single] if single else []


if os.environ.get("BERT_QUERY_WORDS"):
    QUERY_WORDS = [
        w.strip().lower()
        for w in os.environ["BERT_QUERY_WORDS"].split(",")
        if w.strip()
    ]
else:
    QUERY_WORDS = [w.lower() for w in load_query_words_from_shell(QUERY_CONFIG)]
    if not QUERY_WORDS:
        QUERY_WORDS = ["machine", "machines", "slave", "slaves"]

FILTER_WORDS = set(QUERY_WORDS)

TOP_K = 10
BERT_MAX_TOKENS = 512
MAX_GPUS = int(os.environ.get("BERT_MAX_GPUS", "4"))

LOG_FILE          = f"bert_predictions_sharded_{RUN_TIMESTAMP}.log"
UNPROCESSABLE_LOG = f"bert_predictions_sharded_{RUN_TIMESTAMP}_unprocessable.jsonl"


# -------------------------------------------------------------------
# Shared logging via a queue
# -------------------------------------------------------------------

_UNPROCESSABLE = "unproc"


def make_queue_logger(name: str, log_queue: mp.Queue) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.handlers.QueueHandler(log_queue)
        logger.addHandler(handler)
    return logger


def send_unprocessable(log_queue: mp.Queue, reason: str, row: dict, extra: dict | None = None):
    payload = {
        "reason":          reason,
        "record_id":       row.get("record_id", ""),
        "query":           row.get("query", ""),
        "sentence":        row.get("sentence", ""),
        "masked_sentence": row.get("masked_sentence", ""),
        "pg":              row.get("pg", ""),
        **(extra or {}),
    }
    log_queue.put((_UNPROCESSABLE, payload))


def start_log_listener(log_queue: mp.Queue, log_file: str, unprocessable_file: str) -> mp.Process:
    def _listen(q, lf, uf):
        fmt  = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        root = logging.getLogger("listener")
        root.setLevel(logging.INFO)
        fh = logging.FileHandler(lf)
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(fh)
        root.addHandler(sh)
        unproc_fh = open(uf, "w", encoding="utf-8")
        try:
            while True:
                item = q.get()
                if item is None:
                    break
                if isinstance(item, logging.LogRecord):
                    root.handle(item)
                elif isinstance(item, tuple) and item[0] == _UNPROCESSABLE:
                    unproc_fh.write(json.dumps(item[1], ensure_ascii=False) + "\n")
                    unproc_fh.flush()
        except Exception:
            import traceback
            traceback.print_exc()
        finally:
            unproc_fh.close()

    p = mp.Process(target=_listen, args=(log_queue, log_file, unprocessable_file), daemon=False)
    p.start()
    return p


def make_main_logger(log_file: str) -> logging.Logger:
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


@dataclass
class SessionStats:
    total_files: int = 0
    total_rows: int = 0
    session_start: float = field(default_factory=time.monotonic)

    def elapsed_str(self) -> str:
        secs = time.monotonic() - self.session_start
        h, rem = divmod(int(secs), 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"


def gpu_label(device: torch.device) -> str:
    if device.type == "cuda":
        idx = device.index or 0
        return f"GPU:{idx} ({torch.cuda.get_device_name(idx)})"
    return "CPU"


def row_matches_filter(row: dict) -> bool:
    masked_sentence   = row.get("masked_sentence") or ""
    original_sentence = row.get("sentence") or ""
    for orig, masked in zip(original_sentence.split(), masked_sentence.split()):
        if "[MASK]" in masked:
            if orig.strip(".,;:!?\"'()-").lower() in FILTER_WORDS:
                return True
    return False


# -------------------------------------------------------------------
# Byte-range sharding
# -------------------------------------------------------------------

def compute_line_offsets(filepath: Path) -> list[int]:
    """
    Byte offset of the start of each line in filepath, plus a final
    entry equal to the file size. len(offsets) - 1 == number of lines.
    Binary-mode pass, so this is exact regardless of encoding -- shard
    boundaries only ever fall between lines, never mid-character.
    """
    offsets = [0]
    pos = 0
    with open(filepath, "rb") as f:
        for raw_line in f:
            pos += len(raw_line)
            offsets.append(pos)
    return offsets


def compute_shard_boundaries(offsets: list[int], n_shards: int) -> list[tuple[int, int]]:
    """
    Split a file's line range into up to n_shards contiguous (start_byte,
    end_byte) chunks of roughly equal line count. May return fewer than
    n_shards entries if the file has fewer lines than requested shards
    (harmless -- just means fewer, slightly larger shards for tiny files).
    """
    n_lines = len(offsets) - 1
    if n_shards <= 1 or n_lines == 0:
        return [(0, offsets[-1])]

    lines_per_shard = -(-n_lines // n_shards)  # ceil division
    boundaries = []
    for k in range(n_shards):
        start_line = k * lines_per_shard
        if start_line >= n_lines:
            break
        end_line = min(start_line + lines_per_shard, n_lines)
        boundaries.append((offsets[start_line], offsets[end_line]))
    return boundaries


def read_shard_lines(filepath: Path, start_byte: int, end_byte: int) -> list[str]:
    with open(filepath, "rb") as f:
        f.seek(start_byte)
        raw = f.read(end_byte - start_byte)
    return raw.decode("utf-8", errors="replace").splitlines()


# -------------------------------------------------------------------
# Token-level prepare_text
# -------------------------------------------------------------------

@dataclass
class PrepareResult:
    input_ids: Optional[list[int]]
    outcome: str
    token_count: int


def prepare_text(row: dict, tokenizer, log: logging.Logger) -> PrepareResult:
    article_id      = row.get("article_id", row.get("record_id", "<unknown>"))
    prev_sentence   = row.get("prev_sentence")   or ""
    masked_sentence = row.get("masked_sentence") or ""
    next_sentence   = row.get("next_sentence")   or ""

    if "[MASK]" not in masked_sentence:
        return PrepareResult(input_ids=None, outcome="skip_no_mask", token_count=0)

    mask_id = tokenizer.mask_token_id

    def encode(text: str) -> list[int]:
        return tokenizer(
            text, add_special_tokens=True, truncation=False,
            return_attention_mask=False, return_token_type_ids=False,
        )["input_ids"]

    full_text = " ".join(filter(None, [prev_sentence, masked_sentence, next_sentence])).strip()
    full_ids  = encode(full_text)

    if len(full_ids) <= BERT_MAX_TOKENS:
        return PrepareResult(input_ids=full_ids, outcome="ok_full", token_count=len(full_ids))

    log.warning(
        f"Long context — dropping prev_sentence | article_id={article_id} | "
        f"tokens={len(full_ids)} | threshold={BERT_MAX_TOKENS}"
    )

    reduced_text  = " ".join(filter(None, [masked_sentence, next_sentence])).strip()
    reduced_ids   = encode(reduced_text)
    truncated_ids = reduced_ids[:BERT_MAX_TOKENS]

    if mask_id not in truncated_ids:
        log.warning(
            f"Dropping row — [MASK] lost after truncation | article_id={article_id} | "
            f"reduced_tokens={len(reduced_ids)} | truncated_to={len(truncated_ids)}"
        )
        return PrepareResult(input_ids=None, outcome="skip_mask_lost", token_count=0)

    return PrepareResult(input_ids=truncated_ids, outcome="ok_truncated", token_count=len(truncated_ids))


def build_batch_tensors(batch_ids: list[list[int]], pad_token_id: int, device: torch.device) -> dict[str, torch.Tensor]:
    max_len = max(len(ids) for ids in batch_ids)
    padded, att_mask = [], []
    for ids in batch_ids:
        pad_len = max_len - len(ids)
        padded.append(ids + [pad_token_id] * pad_len)
        att_mask.append([1] * len(ids) + [0] * pad_len)
    return {
        "input_ids":      torch.tensor(padded,   dtype=torch.long, device=device),
        "attention_mask": torch.tensor(att_mask, dtype=torch.long, device=device),
    }


# -------------------------------------------------------------------
# Worker
# -------------------------------------------------------------------

def worker(gpu_idx: int, file_queue: mp.Queue, result_queue: mp.Queue, log_queue: mp.Queue, shard_dir: Path):

    device = torch.device(f"cuda:{gpu_idx}")
    label  = gpu_label(device)
    log    = make_queue_logger(f"worker_{gpu_idx}", log_queue)

    log.info(f"Worker started | device={label} | pid={os.getpid()} | dtype={MODEL_DTYPE} | batch_size={BATCH_SIZE}")

    models, tokenizers = {}, {}
    for col_name, model_path in MODELS.items():
        tok = AutoTokenizer.from_pretrained(model_path)
        tok.model_max_length = BERT_MAX_TOKENS
        mdl = AutoModelForMaskedLM.from_pretrained(model_path, torch_dtype=MODEL_DTYPE).to(device).eval()
        tokenizers[col_name] = tok
        models[col_name]     = mdl
        log.info(f"Loaded model | device={label} | col={col_name} | dtype={MODEL_DTYPE}")

    ref_tokenizer = next(iter(tokenizers.values()))
    for col_name, tok in tokenizers.items():
        log.info(f"same_vocab check | col={col_name} | same_as_ref={tok is ref_tokenizer}")

    while True:
        item = file_queue.get()
        if item is None:
            log.info(f"Worker shutting down | device={label}")
            break

        filepath, word, shard_idx, n_shards, start_byte, end_byte = item
        filepath = Path(filepath)

        shard_start_ts = time.time()
        shard_tag = f"{word}[{shard_idx+1}/{n_shards}]"
        log.info(
            f"Starting shard | device={label} | {shard_tag} | file={filepath.name} | "
            f"bytes=[{start_byte}:{end_byte}] ({end_byte - start_byte} bytes)"
        )

        if n_shards == 1:
            out_path = shard_dir.parent / f"blmicrosoft_final_{word}{filepath.suffix}"
        else:
            out_path = shard_dir / f"blmicrosoft_final_{word}.shard{shard_idx:03d}of{n_shards:03d}{filepath.suffix}"

        lines = read_shard_lines(filepath, start_byte, end_byte)

        valid_rows: list[dict] = []
        valid_ids:  list[list[int]] = []

        n_no_mask = n_mask_lost = n_ok_full = n_ok_trunc = n_filter_dropped = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue

            if not row_matches_filter(row):
                n_filter_dropped += 1
                continue

            result = prepare_text(row, ref_tokenizer, log)

            if result.outcome == "skip_no_mask":
                n_no_mask += 1
                continue
            if result.outcome == "skip_mask_lost":
                n_mask_lost += 1
                send_unprocessable(log_queue, "skip_mask_lost", row, {
                    "source_file": filepath.name, "shard": shard_idx,
                })
                continue
            if result.outcome == "ok_truncated":
                n_ok_trunc += 1
            if result.outcome == "ok_full":
                n_ok_full += 1

            valid_rows.append(row)
            valid_ids.append(result.input_ids)

        log.info(
            f"Shard parsed | device={label} | {shard_tag} | ok_full={n_ok_full} | "
            f"ok_truncated={n_ok_trunc} | skipped_no_mask={n_no_mask} | "
            f"skipped_mask_lost={n_mask_lost} | filter_dropped={n_filter_dropped}"
        )

        if not valid_rows:
            # Always write the file (even empty) when sharding, so the merge
            # step doesn't have to guess whether a shard was skipped.
            if n_shards > 1:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.touch()
            log.info(f"No valid rows in shard — skipping inference | device={label} | {shard_tag}")
            result_queue.put({
                "word": word, "shard_idx": shard_idx, "n_shards": n_shards,
                "rows_written": 0, "gpu_label": label,
                "start_ts": shard_start_ts, "end_ts": time.time(),
            })
            continue

        log.info(f"Running inference | device={label} | {shard_tag} | rows={len(valid_rows)}")

        order = sorted(range(len(valid_rows)), key=lambda i: len(valid_ids[i]))
        predictions: dict[str, list] = {col_name: [None] * len(valid_rows) for col_name in models}

        for col_name, mdl in models.items():
            tokenizer = tokenizers[col_name]
            same_vocab = (tokenizer is ref_tokenizer)

            for batch_start in range(0, len(order), BATCH_SIZE):
                batch_order = order[batch_start: batch_start + BATCH_SIZE]
                batch_rows  = [valid_rows[i] for i in batch_order]

                if same_vocab:
                    batch_ids = [valid_ids[i] for i in batch_order]
                    keep_idx  = list(range(len(batch_rows)))
                else:
                    batch_ids, keep_idx = [], []
                    for ridx, row in enumerate(batch_rows):
                        result = prepare_text(row, tokenizer, log)
                        if result.input_ids is None:
                            send_unprocessable(log_queue, "mask_lost_other_tokenizer", row, {
                                "col": col_name, "source_file": filepath.name, "shard": shard_idx,
                            })
                            continue
                        batch_ids.append(result.input_ids)
                        keep_idx.append(ridx)

                batch_preds = [[] for _ in batch_rows]

                if batch_ids:
                    encoded = build_batch_tensors(batch_ids, pad_token_id=tokenizer.pad_token_id, device=device)
                    with torch.inference_mode():
                        outputs = mdl(**encoded)
                    input_ids_tensor = encoded["input_ids"]

                    for j in range(input_ids_tensor.shape[0]):
                        mask_positions = (input_ids_tensor[j] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
                        if len(mask_positions) == 0:
                            row = batch_rows[keep_idx[j]]
                            send_unprocessable(log_queue, "no_mask_in_encoded", row, {
                                "col": col_name, "source_file": filepath.name, "shard": shard_idx,
                            })
                            continue
                        mask_pos = mask_positions[0].item()
                        logits = outputs.logits[j, mask_pos]
                        probs  = torch.softmax(logits.float(), dim=-1)
                        top    = torch.topk(probs, TOP_K)
                        batch_preds[keep_idx[j]] = [
                            (tokenizer.decode([tok_id]).strip(), round(score, 4))
                            for tok_id, score in zip(top.indices.tolist(), top.values.tolist())
                        ]

                for local_pos, orig_idx in enumerate(batch_order):
                    predictions[col_name][orig_idx] = batch_preds[local_pos]

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fout:
            for i, row in enumerate(valid_rows):
                for col_name in models:
                    row[col_name] = predictions[col_name][i]
                fout.write(json.dumps(row) + "\n")

        shard_end_ts = time.time()
        log.info(
            f"Shard complete | device={label} | {shard_tag} | "
            f"rows_written={len(valid_rows)} | elapsed={shard_end_ts - shard_start_ts:.1f}s"
        )

        result_queue.put({
            "word": word, "shard_idx": shard_idx, "n_shards": n_shards,
            "rows_written": len(valid_rows), "gpu_label": label,
            "start_ts": shard_start_ts, "end_ts": shard_end_ts,
        })


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    log = make_main_logger(LOG_FILE)

    log.info("=" * 80)
    log.info(f"Run started:        {RUN_TIMESTAMP}")
    log.info(f"Input dir:          {INPUT_DIR}")
    log.info(f"Output dir:         {OUTPUT_DIR}")
    log.info(f"Suffix:             {SUFFIX}")
    log.info(f"Query words:        {QUERY_WORDS}")
    log.info(f"Shard multiplier:   {SHARD_MULTIPLIER}")
    log.info(f"Batch size:         {BATCH_SIZE}")
    log.info(f"Model dtype:        {MODEL_DTYPE}")

    input_path  = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    shard_dir   = output_path / "_shards"

    n_gpus_requested = min(torch.cuda.device_count(), MAX_GPUS)
    log.info(f"GPUs requested:     {n_gpus_requested}")

    # Which words have step-5 input, and which already have a completed
    # merged output (skip those entirely -- coarse resume, see docstring).
    to_process: list[tuple[Path, str]] = []
    for word in QUERY_WORDS:
        fp = input_path / f"bl_microsoft_{word}_{SUFFIX}_step_5.jsonl"
        if not (fp.exists() and fp.stat().st_size > 0):
            log.warning(f"Expected step-5 file not found or empty, skipping: {fp}")
            continue
        final_out = output_path / f"blmicrosoft_final_{word}{fp.suffix}"
        marker = output_path / f"blmicrosoft_final_{word}{fp.suffix}.done"
        if marker.exists():
            log.info(f"Already complete (marker present), skipping: {final_out}")
            continue
        to_process.append((fp, word))

    if not to_process:
        log.warning("No input files to process. Exiting.")
        return

    n_files = len(to_process)
    shards_per_file = max(1, math.ceil((n_gpus_requested * SHARD_MULTIPLIER) / n_files))
    log.info(f"Files to process:  {n_files}  |  shards_per_file: {shards_per_file}")

    # Build shard tasks. word_n_shards records the ACTUAL shard count used
    # per word (may be < shards_per_file for tiny files), needed for the
    # merge step to know how many shard files to look for.
    tasks = []
    word_n_shards: dict[str, int] = {}
    for fp, word in to_process:
        offsets = compute_line_offsets(fp)
        boundaries = compute_shard_boundaries(offsets, shards_per_file)
        word_n_shards[word] = len(boundaries)
        for shard_idx, (start_byte, end_byte) in enumerate(boundaries):
            tasks.append((str(fp), word, shard_idx, len(boundaries), start_byte, end_byte))

    log.info(f"Total shard tasks: {len(tasks)}")
    for t in tasks:
        log.info(f"  word={t[1]}  shard={t[2]+1}/{t[3]}  bytes=[{t[4]}:{t[5]}]")

    n_gpus = min(n_gpus_requested, len(tasks)) or 1
    log.info(f"GPUs to use:       {n_gpus}  (capped to {len(tasks)} shard task(s))")

    ctx = mp.get_context("spawn")
    log_queue, file_queue, result_queue = ctx.Queue(), ctx.Queue(), ctx.Queue()
    log_listener = start_log_listener(log_queue, LOG_FILE, UNPROCESSABLE_LOG)

    for t in tasks:
        file_queue.put(t)
    for _ in range(n_gpus):
        file_queue.put(None)

    run_start_ts = time.time()

    workers = []
    for gpu_idx in range(n_gpus):
        p = ctx.Process(target=worker, args=(gpu_idx, file_queue, result_queue, log_queue, shard_dir))
        p.start()
        workers.append(p)

    results = [result_queue.get() for _ in range(len(tasks))]

    for p in workers:
        p.join()

    run_end_ts = time.time()
    log_queue.put(None)
    log_listener.join()

    # -----------------------------------------------------------------
    # Merge: for each word with n_shards > 1, concatenate its shard
    # files (in shard-index order) into the final output, write a
    # ".done" marker, and optionally clean up the shard files.
    # -----------------------------------------------------------------
    log.info("=" * 80)
    log.info("MERGING SHARDS")
    for fp, word in to_process:
        n_shards = word_n_shards[word]
        final_out = output_path / f"blmicrosoft_final_{word}{fp.suffix}"
        marker = output_path / f"blmicrosoft_final_{word}{fp.suffix}.done"

        if n_shards == 1:
            # worker already wrote directly to final_out (or wrote nothing
            # if the file had zero valid rows -- matches non-sharded script)
            if final_out.exists():
                marker.touch()
            continue

        shard_files = [
            shard_dir / f"blmicrosoft_final_{word}.shard{i:03d}of{n_shards:03d}{fp.suffix}"
            for i in range(n_shards)
        ]
        existing = [s for s in shard_files if s.exists()]
        if not existing:
            log.warning(f"No shard files found for word={word}, nothing to merge")
            continue

        total_rows = 0
        with open(final_out, "w", encoding="utf-8") as fout:
            for s in shard_files:
                if not s.exists():
                    continue
                with open(s, "r", encoding="utf-8") as fin:
                    for line in fin:
                        if line.strip():
                            fout.write(line if line.endswith("\n") else line + "\n")
                            total_rows += 1

        marker.touch()
        log.info(f"Merged {len(existing)} shard(s) -> {final_out}  ({total_rows} rows)")

        if CLEANUP_SHARDS:
            for s in shard_files:
                s.unlink(missing_ok=True)

    # -----------------------------------------------------------------
    # Timing summary
    # -----------------------------------------------------------------
    results.sort(key=lambda r: r["start_ts"])
    log.info("=" * 80)
    log.info("TIMING SUMMARY (per shard)")
    log.info(f"{'word':<12} {'shard':<8} {'gpu':<28} {'start':<10} {'end':<10} {'elapsed_s':>10} {'rows':>8}")
    for r in results:
        start_str = datetime.fromtimestamp(r["start_ts"]).strftime("%H:%M:%S")
        end_str   = datetime.fromtimestamp(r["end_ts"]).strftime("%H:%M:%S")
        elapsed   = r["end_ts"] - r["start_ts"]
        shard_str = f"{r['shard_idx']+1}/{r['n_shards']}"
        log.info(
            f"{r['word']:<12} {shard_str:<8} {r['gpu_label']:<28} {start_str:<10} "
            f"{end_str:<10} {elapsed:>10.1f} {r['rows_written']:>8}"
        )

    wall_clock = run_end_ts - run_start_ts
    sum_of_elapsed = sum(r["end_ts"] - r["start_ts"] for r in results)
    log.info("-" * 80)
    log.info(f"Total wall-clock time (parallel): {wall_clock:.1f}s")
    log.info(f"Sum of individual shard times (if run sequentially): {sum_of_elapsed:.1f}s")
    if wall_clock > 0:
        log.info(f"Approximate speedup factor: {sum_of_elapsed / wall_clock:.2f}x")
    log.info("=" * 80)
    log.info("All workers finished.")


if __name__ == "__main__":
    main()