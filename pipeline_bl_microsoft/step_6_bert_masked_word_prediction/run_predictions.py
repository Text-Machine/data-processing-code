"""
BERT Masked Word Prediction Script (multi-GPU, batch_size=64)
=====================================================================
"""

import json
import logging
import logging.handlers
import multiprocessing as mp
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

# -------------------------------------------------------------------
# Batch size
# -------------------------------------------------------------------

BATCH_SIZE = 64

# -------------------------------------------------------------------
# Config  (all paths overridable via environment variables)
# -------------------------------------------------------------------

MODELS_BASE = os.environ.get(
    "BERT_MODELS_BASE",
    "/gpfs/projects/bsc100/models/bert_textmachine",
)

MODELS = {
    "pred_bert_1760_1850":    f"{MODELS_BASE}/bert_1760_1850",
    "pred_bert_1890_1900":    f"{MODELS_BASE}/bert_1890_1900",
    "pred_bert_contemporary": f"{MODELS_BASE}/bert-base-uncased",
}

INPUT_DIR = os.environ.get(
    "BERT_INPUT_DIR",
    "/gpfs/projects/bsc100/textmachine-data/preprocessed_data/consolidated_metadata",
)
INPUT_GLOB = os.environ.get(
    "BERT_INPUT_GLOB",
    "bl_microsoft_{machine,machines,slave,slaves}_step_5.jsonl",
)
OUTPUT_DIR = os.environ.get(
    "BERT_OUTPUT_DIR",
    "/gpfs/projects/bsc100/textmachine-data/filtered_data_predictions_batch64",
)

FILTER_WORDS = {"machine", "machines", "slave", "slaves"}

TOP_K = 10
BERT_MAX_TOKENS = 512
MAX_GPUS = 4

LOG_FILE = f"bert_predictions_batch64_{RUN_TIMESTAMP}.log"


# -------------------------------------------------------------------
# Shared logging via a queue
# -------------------------------------------------------------------

def make_queue_logger(name: str, log_queue: mp.Queue) -> logging.Logger:
    """Logger for worker processes: sends records to the shared queue."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.handlers.QueueHandler(log_queue)
        logger.addHandler(handler)
    return logger


def start_log_listener(log_queue: mp.Queue, log_file: str) -> mp.Process:
    """
    Listener process: drains the queue and writes to file + stderr.
    Returns the started Process; caller must .join() it after workers finish.
    """
    def _listen(q, lf):
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        root = logging.getLogger("listener")
        root.setLevel(logging.INFO)
        fh = logging.FileHandler(lf)
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(fh)
        root.addHandler(sh)

        while True:
            try:
                record = q.get()
                if record is None:       # sentinel
                    break
                root.handle(record)
            except Exception:
                import traceback
                traceback.print_exc()

    p = mp.Process(target=_listen, args=(log_queue, log_file), daemon=False)
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
# Stats dataclass
# -------------------------------------------------------------------

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
    masked_sentence  = row.get("masked_sentence") or ""
    original_sentence = row.get("sentence") or ""
    for orig, masked in zip(original_sentence.split(), masked_sentence.split()):
        if masked == "[MASK]":
            if orig.strip(".,;:!?\"'()-").lower() in FILTER_WORDS:
                return True
    return False


# -------------------------------------------------------------------
# Token-level prepare_text — guarantees <=512 tokens reaching BERT
# -------------------------------------------------------------------

@dataclass
class PrepareResult:
    text: str | None
    input_ids: list[int] | None
    outcome: str      # "ok_full" | "ok_truncated" | "skip_no_mask" | "skip_mask_lost"
    token_count: int  # token count of the text actually sent (0 if skipped)


def prepare_text(row: dict, tokenizer, log: logging.Logger) -> PrepareResult:
    """
    Tokenise the context, apply truncation rules at the *token* level,
    then decode back to a string that is guaranteed <=512 tokens.

    Rules:
      1. full_context = prev + masked + next
         - if <=512 tokens  →  use as-is  (outcome: ok_full)
         - if  >512 tokens  →  drop prev, try masked + next  (log warning)
      2. reduced = masked + next, hard-truncated to 512 tokens
         - if [MASK] id still present  →  decode and use  (outcome: ok_truncated)
         - if [MASK] id gone           →  drop row        (outcome: skip_mask_lost)

    The returned text is decoded from token ids so that what hits the
    tokenizer during inference is as close as possible to what we measured.
    Inference re-tokenizes with truncation=True / max_length=512 as a
    hard safety net (see worker).
    """
    article_id      = row.get("article_id", "<unknown>")
    prev_sentence   = row.get("prev_sentence")   or ""
    masked_sentence = row.get("masked_sentence") or ""
    next_sentence   = row.get("next_sentence")   or ""

    if "[MASK]" not in masked_sentence:
        return PrepareResult(text=None, input_ids=None,
                             outcome="skip_no_mask", token_count=0)

    mask_id = tokenizer.mask_token_id

    # --- helper: tokenise with special tokens, return input_ids as list ---
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
        decoded = tokenizer.decode(full_ids, skip_special_tokens=False)
        return PrepareResult(text=decoded, input_ids=full_ids,
                             outcome="ok_full", token_count=len(full_ids))

    # --- Rule 2: drop prev_sentence, hard-truncate masked + next ---
    log.warning(
        f"Long context — dropping prev_sentence | "
        f"article_id={article_id} | tokens={len(full_ids)} | threshold={BERT_MAX_TOKENS}"
    )

    reduced_text = " ".join(filter(None, [masked_sentence, next_sentence])).strip()
    reduced_ids  = encode(reduced_text)
    truncated_ids = reduced_ids[:BERT_MAX_TOKENS]   # hard truncation at token level

    if mask_id not in truncated_ids:
        log.warning(
            f"Dropping row — [MASK] lost after truncation | "
            f"article_id={article_id} | "
            f"reduced_tokens={len(reduced_ids)} | truncated_to={len(truncated_ids)}"
        )
        return PrepareResult(text=None, input_ids=None,
                             outcome="skip_mask_lost", token_count=0)

    decoded = tokenizer.decode(truncated_ids, skip_special_tokens=False)
    return PrepareResult(text=decoded, input_ids=truncated_ids,
                         outcome="ok_truncated", token_count=len(truncated_ids))


# -------------------------------------------------------------------
# Worker
# -------------------------------------------------------------------

def worker(gpu_idx: int, file_queue: mp.Queue, result_queue: mp.Queue, log_queue: mp.Queue):

    device = torch.device(f"cuda:{gpu_idx}")
    label  = gpu_label(device)
    log    = make_queue_logger(f"worker_{gpu_idx}", log_queue)

    log.info(f"Worker started | device={label} | pid={os.getpid()}")

    # Load models and tokenizers directly — no pipeline in inference path.
    # This avoids the double-tokenization bug where pipeline re-tokenizes
    # decoded strings, producing 513/514-token sequences from <=512-token inputs.
    models     = {}
    tokenizers = {}

    for col_name, model_path in MODELS.items():
        tok = AutoTokenizer.from_pretrained(model_path)
        tok.model_max_length = BERT_MAX_TOKENS
        mdl = AutoModelForMaskedLM.from_pretrained(model_path).to(device).eval()
        tokenizers[col_name] = tok
        models[col_name]     = mdl
        log.info(f"Loaded model | device={label} | col={col_name}")

    ref_tokenizer = next(iter(tokenizers.values()))

    while True:
        item = file_queue.get()
        if item is None:
            log.info(f"Worker shutting down | device={label}")
            break

        filepath, output_dir = item
        filepath   = Path(filepath)
        output_dir = Path(output_dir)

        log.info(f"Starting file | device={label} | file={filepath.name} | path={filepath}")

        output_dir.mkdir(parents=True, exist_ok=True)
        out_name = filepath.stem.replace("_step_5", "") + "_step_6" + filepath.suffix
        out_path = output_dir / out_name

        if is_file_complete(filepath, out_path):
            log.info(f"Skipping — already complete | device={label} | file={filepath.name}")
            result_queue.put({"skipped": True, "filepath": str(filepath)})
            continue

        valid_rows  = []
        valid_texts = []

        # per-file counters
        n_no_mask   = 0
        n_too_long  = 0   # dropped prev
        n_mask_lost = 0   # [MASK] truncated away
        n_ok_full   = 0
        n_ok_trunc  = 0

        with open(filepath, "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue

                if not row_matches_filter(row):
                    continue

                result = prepare_text(row, ref_tokenizer, log)

                if result.outcome == "skip_no_mask":
                    n_no_mask += 1
                    continue
                if result.outcome == "skip_mask_lost":
                    n_mask_lost += 1
                    continue
                if result.outcome == "ok_truncated":
                    n_too_long += 1   # prev was dropped
                    n_ok_trunc += 1
                if result.outcome == "ok_full":
                    n_ok_full += 1

                valid_rows.append(row)
                valid_texts.append(result.text)

        log.info(
            f"File parsed | device={label} | file={filepath.name} | "
            f"ok_full={n_ok_full} | ok_truncated={n_ok_trunc} | "
            f"skipped_no_mask={n_no_mask} | skipped_mask_lost={n_mask_lost}"
        )

        if not valid_rows:
            log.info(f"No valid rows — skipping inference | device={label} | file={filepath.name}")
            result_queue.put({
                "skipped": False,
                "filepath": str(filepath),
                "rows_written": 0,
                "gpu_label": label,
            })
            continue

        log.info(
            f"Running inference | device={label} | file={filepath.name} | rows={len(valid_rows)}"
        )

        predictions = {}

        for col_name, mdl in models.items():
            tokenizer = tokenizers[col_name]
            col_preds = []

            for batch_start in range(0, len(valid_texts), BATCH_SIZE):
                batch_texts = valid_texts[batch_start : batch_start + BATCH_SIZE]

                encoded = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=BERT_MAX_TOKENS,
                ).to(device)

                with torch.no_grad():
                    outputs = mdl(**encoded)

                input_ids = encoded["input_ids"]

                for j in range(input_ids.shape[0]):
                    mask_positions = (
                        input_ids[j] == tokenizer.mask_token_id
                    ).nonzero(as_tuple=True)[0]

                    if len(mask_positions) == 0:
                        log.warning(
                            f"No [MASK] token found in encoded input | "
                            f"col={col_name} | batch_row={j} | "
                            f"file={filepath.name}"
                        )
                        col_preds.append([])
                        continue

                    mask_pos = mask_positions[0].item()
                    logits   = outputs.logits[j, mask_pos]
                    probs    = torch.softmax(logits, dim=-1)
                    top      = torch.topk(probs, TOP_K)

                    col_preds.append([
                        (tokenizer.decode([tok_id]).strip(), round(score, 4))
                        for tok_id, score in zip(
                            top.indices.tolist(), top.values.tolist()
                        )
                    ])

            predictions[col_name] = col_preds

        with open(out_path, "w", encoding="utf-8") as fout:
            for i, row in enumerate(valid_rows):
                for col_name in models:
                    row[col_name] = predictions[col_name][i]
                fout.write(json.dumps(row) + "\n")

        log.info(
            f"File complete | device={label} | file={filepath.name} | rows_written={len(valid_rows)}"
        )

        result_queue.put({
            "skipped": False,
            "filepath": str(filepath),
            "rows_written": len(valid_rows),
            "gpu_label": label,
        })


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    log = make_main_logger(LOG_FILE)

    log.info("=" * 80)
    log.info(f"Run started:  {RUN_TIMESTAMP}")
    log.info(f"Input dir:    {INPUT_DIR}")
    log.info(f"Input glob:   {INPUT_GLOB}")
    log.info(f"Output dir:   {OUTPUT_DIR}")
    log.info(f"Log file:     {LOG_FILE}")

    n_gpus = min(torch.cuda.device_count(), MAX_GPUS) or 1
    log.info(f"GPUs to use:  {n_gpus}")

    input_path  = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)

    # Path.glob supports brace expansion only in Python 3.12+;
    # use explicit word list as fallback for Python 3.11 and earlier.
    matched = sorted(input_path.glob(INPUT_GLOB))
    if not matched:
        words = ["machine", "machines", "slave", "slaves"]
        matched = sorted(
            fp
            for word in words
            for fp in input_path.glob(f"bl_microsoft_{word}_step_5.jsonl")
        )

    # Skip zero-byte files (e.g. night / morning placeholders)
    all_files = [
        (str(fp), str(output_path))
        for fp in matched
        if fp.stat().st_size > 0
    ]

    log.info(f"Files queued: {len(all_files)}")
    for fp, _ in all_files:
        log.info(f"  {fp}")

    ctx = mp.get_context("spawn")

    log_queue    = ctx.Queue()
    file_queue   = ctx.Queue()
    result_queue = ctx.Queue()

    # Start the log listener before workers
    log_listener = start_log_listener(log_queue, LOG_FILE)

    for f in all_files:
        file_queue.put(f)
    for _ in range(n_gpus):
        file_queue.put(None)

    workers = []
    for gpu_idx in range(n_gpus):
        p = ctx.Process(
            target=worker,
            args=(gpu_idx, file_queue, result_queue, log_queue),
        )
        p.start()
        workers.append(p)

    for p in workers:
        p.join()

    # Shut down log listener
    log_queue.put(None)
    log_listener.join()

    log.info("All workers finished.")


if __name__ == "__main__":
    main()
