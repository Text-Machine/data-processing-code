"""
BERT Masked Word Prediction Script (multi-GPU, batch_size configurable, fp16, length-bucketed)
=====================================================================
BL Microsoft only. For the dataset-agnostic (BL Microsoft/HMD/LWM)
version see bert_predictions.py -- this is a focused variant for timing
and parallelism verification on BL Microsoft specifically.

File selection
--------------
Input files are selected explicitly by query word rather than by a
directory-wide glob:

    {BERT_INPUT_DIR}/bl_microsoft_{word}_{BERT_SUFFIX}_step_5.jsonl

Query-word resolution (highest priority first):
  1. BERT_QUERY_WORDS env var, comma-separated (e.g. "machine,machines") --
     useful for one-off subset tests without touching the shared config.
  2. query_word_config.sh (path via BERT_QUERY_CONFIG, default
     "query_word_config.sh" in the working dir) -- the real one:
         QUERY_WORDS=(mind man men woman women child children)
  3. Hardcoded fallback: machine, machines, slave, slaves

QUERY_WORDS drives both FILTER_WORDS and which step-5 files get opened.
A run for a given word set never opens, checks, or reprocesses files
belonging to words outside that set.

GPU count & scheduling
-----------------------
n_gpus = min(available GPUs, BERT_MAX_GPUS, number of matched files).
This single formula gives you both behaviours you asked about, since
they're really the same mechanism at different scales:

  - words <= GPUs available (e.g. 2 words, 2 GPUs requested): n_gpus is
    capped DOWN to 2. Without this cap, extra idle workers would still
    pay the full model-loading cost per GPU before finding nothing left
    in the queue -- this is what makes it genuinely "one GPU per word"
    rather than just "some GPUs get a word, others sit there having
    loaded models for nothing."
  - words > GPUs available (e.g. 7 words, 4 GPUs): n_gpus stays at 4.
    file_queue is a shared multiprocessing queue that each GPU worker
    pulls from in a loop (see worker()) -- whichever GPU finishes its
    current file first immediately grabs the next one. This is already
    dynamic, pull-based scheduling: 4 words run concurrently, and the
    moment any one finishes, the 5th starts on that same GPU, and so on.
    No separate script is needed for this case.

Timing / parallelism verification
-----------------------------------
Each file's wall-clock start/end time and elapsed seconds are recorded
in the result dict put on result_queue. main() now drains result_queue
after workers finish and prints a summary table sorted by start time --
overlapping [start, end] windows across different GPU labels is direct
evidence that words were processed in parallel, not sequentially.

Performance changes vs. the original batch64 version
------------------------------------------------------
1. Models loaded in fp16 (BERT_DTYPE=float16|bfloat16|float32, default float16).
2. BATCH_SIZE is now overridable via env (BERT_BATCH_SIZE, default 256).
3. Rows are length-bucketed (sorted by reference-tokenizer token count)
   before chunking into batches, then unsorted before writing output.
4. torch.no_grad() -> torch.inference_mode().
5. GPU count now defaults to all visible devices, capped by MAX_GPUS and
   (new) by the number of matched files.
"""

import json
import logging
import logging.handlers
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
# Config  (all paths/words overridable via environment variables)
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

# Sentence-splitting suffix used by step 4/5 filenames (spacy or regex).
SUFFIX = os.environ.get("BERT_SUFFIX", "spacy")

# -------------------------------------------------------------------
# Query-word resolution: env override > query_word_config.sh > hardcoded
# -------------------------------------------------------------------

QUERY_CONFIG = Path(os.environ.get("BERT_QUERY_CONFIG", "query_word_config.sh"))


def load_query_words_from_shell(config_path: Path) -> list[str]:
    """
    Read the bash array QUERY_WORDS=(...) from config_path, e.g.:
        QUERY_WORDS=(mind man men woman women child children)
    Falls back to a singular QUERY_WORD="..." if no array is set.
    Returns [] if the file doesn't exist or neither variable is set.
    """
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

LOG_FILE          = f"bert_predictions_batch64_{RUN_TIMESTAMP}.log"
UNPROCESSABLE_LOG = f"bert_predictions_batch64_{RUN_TIMESTAMP}_unprocessable.jsonl"


# -------------------------------------------------------------------
# Shared logging via a queue
# -------------------------------------------------------------------

# Sentinel types sent over log_queue
_LOG_RECORD    = "log"       # a logging.LogRecord
_UNPROCESSABLE = "unproc"    # a dict row to append to the unprocessable log


def make_queue_logger(name: str, log_queue: mp.Queue) -> logging.Logger:
    """Logger for worker processes: sends records to the shared queue."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.handlers.QueueHandler(log_queue)
        logger.addHandler(handler)
    return logger


def send_unprocessable(log_queue: mp.Queue, reason: str, row: dict, extra: dict | None = None):
    """
    Send an unprocessable-row record to the listener via the shared queue.
    `reason` is one of: "skip_mask_lost" | "no_mask_in_encoded" | "mask_lost_other_tokenizer"
    """
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


def start_log_listener(
    log_queue: mp.Queue,
    log_file: str,
    unprocessable_file: str,
) -> mp.Process:
    """
    Listener process: drains the queue.
    - logging.LogRecord  → written to log_file + stderr
    - (_UNPROCESSABLE, dict) → appended to unprocessable_file as JSONL
    Returns the started Process; caller must .join() after workers finish.
    """
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
                if item is None:          # shutdown sentinel
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
    """Direct logger for the main process (not queue-based)."""
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


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def gpu_label(device: torch.device) -> str:
    if device.type == "cuda":
        idx = device.index or 0
        return f"GPU:{idx} ({torch.cuda.get_device_name(idx)})"
    return "CPU"


def is_file_complete(input_path: Path, output_path: Path) -> bool:
    if not output_path.exists():
        return False
    def count_lines(p: Path) -> int:
        with open(p, "r", encoding="utf-8") as f:
            return sum(1 for ln in f if ln.strip())
    return count_lines(input_path) == count_lines(output_path)


def row_matches_filter(row: dict) -> bool:
    """
    Returns True if at least one [MASK] token in masked_sentence corresponds
    to a word in FILTER_WORDS in the original sentence.
    """
    masked_sentence   = row.get("masked_sentence") or ""
    original_sentence = row.get("sentence") or ""
    for orig, masked in zip(original_sentence.split(), masked_sentence.split()):
        if "[MASK]" in masked:
            if orig.strip(".,;:!?\"'()-").lower() in FILTER_WORDS:
                return True
    return False


# -------------------------------------------------------------------
# Token-level prepare_text — guarantees <=512 tokens reaching BERT
# -------------------------------------------------------------------

@dataclass
class PrepareResult:
    input_ids: Optional[list[int]]   # None when outcome is a skip
    outcome: str      # "ok_full" | "ok_truncated" | "skip_no_mask" | "skip_mask_lost"
    token_count: int  # token count of the ids actually used (0 if skipped)


def prepare_text(row: dict, tokenizer, log: logging.Logger) -> PrepareResult:
    """
    Tokenise the context and apply truncation rules at the *token* level,
    using THIS tokenizer's own vocabulary.

    Rules:
      1. full_context = prev + masked + next
         - if <=512 tokens  →  use as-is               (outcome: ok_full)
         - if  >512 tokens  →  drop prev, try masked + next  (log warning)
      2. reduced = masked + next, hard-truncated to 512 tokens
         - if [MASK] id still present  →  use           (outcome: ok_truncated)
         - if [MASK] id gone           →  drop row      (outcome: skip_mask_lost)
    """
    article_id      = row.get("article_id", row.get("record_id", "<unknown>"))
    prev_sentence   = row.get("prev_sentence")   or ""
    masked_sentence = row.get("masked_sentence") or ""
    next_sentence   = row.get("next_sentence")   or ""

    if "[MASK]" not in masked_sentence:
        return PrepareResult(input_ids=None, outcome="skip_no_mask", token_count=0)

    mask_id = tokenizer.mask_token_id

    def encode(text: str) -> list[int]:
        return tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]

    # --- Rule 1: try full context ---
    full_text = " ".join(filter(None, [prev_sentence, masked_sentence, next_sentence])).strip()
    full_ids  = encode(full_text)

    if len(full_ids) <= BERT_MAX_TOKENS:
        return PrepareResult(input_ids=full_ids, outcome="ok_full", token_count=len(full_ids))

    # --- Rule 2: drop prev_sentence, hard-truncate masked + next ---
    log.warning(
        f"Long context — dropping prev_sentence | "
        f"article_id={article_id} | tokens={len(full_ids)} | threshold={BERT_MAX_TOKENS}"
    )

    reduced_text  = " ".join(filter(None, [masked_sentence, next_sentence])).strip()
    reduced_ids   = encode(reduced_text)
    truncated_ids = reduced_ids[:BERT_MAX_TOKENS]

    if mask_id not in truncated_ids:
        log.warning(
            f"Dropping row — [MASK] lost after truncation | "
            f"article_id={article_id} | "
            f"reduced_tokens={len(reduced_ids)} | truncated_to={len(truncated_ids)}"
        )
        return PrepareResult(input_ids=None, outcome="skip_mask_lost", token_count=0)

    return PrepareResult(
        input_ids=truncated_ids,
        outcome="ok_truncated",
        token_count=len(truncated_ids),
    )


# -------------------------------------------------------------------
# Batch builder — pads pre-computed input_ids directly into tensors
# -------------------------------------------------------------------

def build_batch_tensors(
    batch_ids: list[list[int]],
    pad_token_id: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    max_len = max(len(ids) for ids in batch_ids)

    padded   = []
    att_mask = []
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

def worker(gpu_idx: int, file_queue: mp.Queue, result_queue: mp.Queue, log_queue: mp.Queue):

    device = torch.device(f"cuda:{gpu_idx}")
    label  = gpu_label(device)
    log    = make_queue_logger(f"worker_{gpu_idx}", log_queue)

    log.info(f"Worker started | device={label} | pid={os.getpid()} | dtype={MODEL_DTYPE} | batch_size={BATCH_SIZE}")

    models     = {}
    tokenizers = {}

    for col_name, model_path in MODELS.items():
        tok = AutoTokenizer.from_pretrained(model_path)
        tok.model_max_length = BERT_MAX_TOKENS
        mdl = AutoModelForMaskedLM.from_pretrained(
            model_path, torch_dtype=MODEL_DTYPE
        ).to(device).eval()
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

        # word is resolved at file-discovery time and carried through the
        # queue -- NOT re-derived from the filename (the old
        # `filepath.stem.split("_")[2]` only worked by coincidence for the
        # two-token "bl_microsoft_" prefix).
        filepath, output_dir, word = item
        filepath   = Path(filepath)
        output_dir = Path(output_dir)

        file_start_ts = time.time()
        log.info(
            f"Starting file | device={label} | word={word} | "
            f"file={filepath.name} | start={datetime.fromtimestamp(file_start_ts).strftime('%H:%M:%S')}"
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        out_name = f"blmicrosoft_final_{word}{filepath.suffix}"
        out_path = output_dir / out_name

        if is_file_complete(filepath, out_path):
            log.info(f"Skipping — already complete | device={label} | file={filepath.name}")
            result_queue.put({
                "skipped": True, "filepath": str(filepath), "word": word,
                "gpu_label": label, "start_ts": file_start_ts, "end_ts": time.time(),
            })
            continue

        valid_rows: list[dict]       = []
        valid_ids:  list[list[int]]  = []

        n_no_mask        = 0
        n_mask_lost      = 0
        n_ok_full        = 0
        n_ok_trunc       = 0
        n_too_long       = 0
        n_filter_dropped = 0

        dropped_rows: list[tuple[int, dict]] = []

        with open(filepath, "r", encoding="utf-8") as fin:
            for line_num, line in enumerate(fin, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue

                if not row_matches_filter(row):
                    n_filter_dropped += 1
                    dropped_rows.append((line_num, row))
                    continue

                result = prepare_text(row, ref_tokenizer, log)

                if result.outcome == "skip_no_mask":
                    n_no_mask += 1
                    continue

                if result.outcome == "skip_mask_lost":
                    n_mask_lost += 1
                    send_unprocessable(log_queue, "skip_mask_lost", row, {
                        "source_file": filepath.name,
                        "line_num":    line_num,
                    })
                    continue

                if result.outcome == "ok_truncated":
                    n_ok_trunc += 1
                    n_too_long += 1
                if result.outcome == "ok_full":
                    n_ok_full += 1

                valid_rows.append(row)
                valid_ids.append(result.input_ids)

        if dropped_rows:
            dropped_path = output_dir / (filepath.stem + "_dropped_by_filter.jsonl")
            with open(dropped_path, "w", encoding="utf-8") as fdr:
                for line_num, row in dropped_rows:
                    fdr.write(json.dumps({
                        "line_num":        line_num,
                        "record_id":       row.get("record_id", ""),
                        "query":           row.get("query", ""),
                        "sentence":        row.get("sentence", ""),
                        "masked_sentence": row.get("masked_sentence", ""),
                    }, ensure_ascii=False) + "\n")
            log.warning(
                f"Filter-dropped rows → {dropped_path.name} ({len(dropped_rows)} rows)"
            )

        log.info(
            f"File parsed | device={label} | file={filepath.name} | "
            f"ok_full={n_ok_full} | ok_truncated={n_ok_trunc} | "
            f"skipped_no_mask={n_no_mask} | skipped_mask_lost={n_mask_lost} | "
            f"filter_dropped={n_filter_dropped}"
        )

        if not valid_rows:
            log.info(f"No valid rows — skipping inference | device={label} | file={filepath.name}")
            result_queue.put({
                "skipped": False, "filepath": str(filepath), "word": word,
                "rows_written": 0, "gpu_label": label,
                "start_ts": file_start_ts, "end_ts": time.time(),
            })
            continue

        log.info(
            f"Running inference | device={label} | file={filepath.name} | rows={len(valid_rows)}"
        )

        order = sorted(range(len(valid_rows)), key=lambda i: len(valid_ids[i]))

        predictions: dict[str, list] = {
            col_name: [None] * len(valid_rows) for col_name in models
        }

        for col_name, mdl in models.items():
            tokenizer = tokenizers[col_name]
            same_vocab = (tokenizer is ref_tokenizer)

            for batch_start in range(0, len(order), BATCH_SIZE):
                batch_order = order[batch_start : batch_start + BATCH_SIZE]
                batch_rows  = [valid_rows[i] for i in batch_order]

                if same_vocab:
                    batch_ids = [valid_ids[i] for i in batch_order]
                    keep_idx  = list(range(len(batch_rows)))
                else:
                    batch_ids = []
                    keep_idx  = []
                    for ridx, row in enumerate(batch_rows):
                        result = prepare_text(row, tokenizer, log)
                        if result.input_ids is None:
                            log.warning(
                                f"Mask lost for this tokenizer's vocab | "
                                f"col={col_name} | batch_row={ridx} | file={filepath.name}"
                            )
                            send_unprocessable(log_queue, "mask_lost_other_tokenizer", row, {
                                "col":         col_name,
                                "source_file": filepath.name,
                                "batch_start": batch_start,
                                "batch_row":   ridx,
                            })
                            continue
                        batch_ids.append(result.input_ids)
                        keep_idx.append(ridx)

                batch_preds = [[] for _ in batch_rows]

                if batch_ids:
                    encoded = build_batch_tensors(
                        batch_ids,
                        pad_token_id=tokenizer.pad_token_id,
                        device=device,
                    )

                    with torch.inference_mode():
                        outputs = mdl(**encoded)

                    input_ids_tensor = encoded["input_ids"]

                    for j in range(input_ids_tensor.shape[0]):
                        mask_positions = (
                            input_ids_tensor[j] == tokenizer.mask_token_id
                        ).nonzero(as_tuple=True)[0]

                        if len(mask_positions) == 0:
                            row = batch_rows[keep_idx[j]]
                            log.warning(
                                f"No [MASK] token found in encoded input (unexpected) | "
                                f"col={col_name} | batch_row={j} | file={filepath.name}"
                            )
                            send_unprocessable(log_queue, "no_mask_in_encoded", row, {
                                "col":         col_name,
                                "source_file": filepath.name,
                                "batch_start": batch_start,
                                "batch_row":   j,
                            })
                            continue

                        mask_pos = mask_positions[0].item()
                        logits   = outputs.logits[j, mask_pos]
                        probs    = torch.softmax(logits.float(), dim=-1)
                        top      = torch.topk(probs, TOP_K)

                        batch_preds[keep_idx[j]] = [
                            (tokenizer.decode([tok_id]).strip(), round(score, 4))
                            for tok_id, score in zip(
                                top.indices.tolist(), top.values.tolist()
                            )
                        ]

                for local_pos, orig_idx in enumerate(batch_order):
                    predictions[col_name][orig_idx] = batch_preds[local_pos]

        with open(out_path, "w", encoding="utf-8") as fout:
            for i, row in enumerate(valid_rows):
                for col_name in models:
                    row[col_name] = predictions[col_name][i]
                fout.write(json.dumps(row) + "\n")

        file_end_ts = time.time()
        elapsed = file_end_ts - file_start_ts

        log.info(
            f"File complete | device={label} | word={word} | file={filepath.name} | "
            f"rows_written={len(valid_rows)} | elapsed={elapsed:.1f}s"
        )

        result_queue.put({
            "skipped": False,
            "filepath": str(filepath),
            "word": word,
            "rows_written": len(valid_rows),
            "gpu_label": label,
            "start_ts": file_start_ts,
            "end_ts": file_end_ts,
        })


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    log = make_main_logger(LOG_FILE)

    log.info("=" * 80)
    log.info(f"Run started:       {RUN_TIMESTAMP}")
    log.info(f"Input dir:         {INPUT_DIR}")
    log.info(f"Output dir:        {OUTPUT_DIR}")
    log.info(f"Suffix:            {SUFFIX}")
    log.info(f"Query config:      {QUERY_CONFIG}")
    log.info(f"Query words:       {QUERY_WORDS}")
    log.info(f"Filter words:      {sorted(FILTER_WORDS)}")
    log.info(f"Batch size:        {BATCH_SIZE}")
    log.info(f"Model dtype:       {MODEL_DTYPE}")
    log.info(f"Log file:          {LOG_FILE}")
    log.info(f"Unprocessable log: {UNPROCESSABLE_LOG}")

    input_path  = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)

    # File selection: one explicit file per query word, rather than a
    # directory-wide glob. Word is carried alongside the filepath so the
    # worker never has to re-derive it from the filename.
    matched: list[tuple[Path, str]] = []
    for word in QUERY_WORDS:
        fp = input_path / f"bl_microsoft_{word}_{SUFFIX}_step_5.jsonl"
        if fp.exists() and fp.stat().st_size > 0:
            matched.append((fp, word))
        else:
            log.warning(f"Expected step-5 file not found or empty, skipping: {fp}")

    all_files = [(str(fp), str(output_path), word) for fp, word in matched]

    log.info(f"Files queued: {len(all_files)}")
    for fp, _, word in all_files:
        log.info(f"  {fp}  (word={word})")

    if not all_files:
        log.warning("No input files found for any of the requested query words. Exiting.")
        return

    # GPU count capped to the number of matched files (== number of query
    # words). This is what gives "one GPU per word" when words <= GPUs
    # requested, and correctly falls back to dynamic queue-based scheduling
    # when words > GPUs (see module docstring).
    n_gpus = min(torch.cuda.device_count(), MAX_GPUS, len(all_files)) or 1
    log.info(f"GPUs to use:       {n_gpus}  (capped to {len(all_files)} matched file(s))")

    ctx = mp.get_context("spawn")

    log_queue    = ctx.Queue()
    file_queue   = ctx.Queue()
    result_queue = ctx.Queue()

    log_listener = start_log_listener(log_queue, LOG_FILE, UNPROCESSABLE_LOG)

    for f in all_files:
        file_queue.put(f)
    for _ in range(n_gpus):
        file_queue.put(None)

    run_start_ts = time.time()

    workers = []
    for gpu_idx in range(n_gpus):
        p = ctx.Process(
            target=worker,
            args=(gpu_idx, file_queue, result_queue, log_queue),
        )
        p.start()
        workers.append(p)

    # Drain result_queue for the timing summary. Exactly one result is put
    # per queued file, so we know how many to expect up front.
    results = []
    for _ in range(len(all_files)):
        results.append(result_queue.get())

    for p in workers:
        p.join()

    run_end_ts = time.time()

    log_queue.put(None)
    log_listener.join()

    # ---------------------------------------------------------------
    # Timing summary -- this is the evidence of parallel speedup.
    # Overlapping [start, end] windows across different gpu_label values
    # mean those words genuinely ran concurrently, not back-to-back.
    # ---------------------------------------------------------------
    results.sort(key=lambda r: r["start_ts"])

    log.info("=" * 80)
    log.info("TIMING SUMMARY")
    log.info(
        f"{'word':<12} {'gpu':<28} {'start':<10} {'end':<10} {'elapsed_s':>10} {'rows':>8}"
    )
    for r in results:
        start_str = datetime.fromtimestamp(r["start_ts"]).strftime("%H:%M:%S")
        end_str   = datetime.fromtimestamp(r["end_ts"]).strftime("%H:%M:%S")
        elapsed   = r["end_ts"] - r["start_ts"]
        rows      = r.get("rows_written", "skipped" if r.get("skipped") else "?")
        log.info(
            f"{r['word']:<12} {r['gpu_label']:<28} {start_str:<10} {end_str:<10} "
            f"{elapsed:>10.1f} {str(rows):>8}"
        )

    wall_clock = run_end_ts - run_start_ts
    sum_of_elapsed = sum(r["end_ts"] - r["start_ts"] for r in results)
    log.info("-" * 80)
    log.info(f"Total wall-clock time (parallel): {wall_clock:.1f}s")
    log.info(f"Sum of individual file times (if run sequentially): {sum_of_elapsed:.1f}s")
    if wall_clock > 0:
        log.info(f"Approximate speedup factor: {sum_of_elapsed / wall_clock:.2f}x")
    log.info("=" * 80)

    log.info("All workers finished.")


if __name__ == "__main__":
    main()