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
from typing import Optional

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
    """
    Returns True if at least one [MASK] token in masked_sentence corresponds
    to a FILTER_WORD in the original sentence.

    Uses `"[MASK]" in masked` (substring check) so that tokens like "[MASK],"
    and "[MASK]." are correctly matched rather than silently dropped.
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

    IMPORTANT: token counts are vocabulary-specific. The same text can fit
    under 512 tokens for one tokenizer and overflow for another (e.g. rare
    archaic words fragment very differently across vocabularies). This
    function must therefore be called separately, with the matching
    tokenizer, for every model that will consume its output — it must never
    be called once and its result reused across different tokenizers.
    """
    article_id      = row.get("article_id", "<unknown>")
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
    """
    Pad a list of input_id sequences to the same length and return the
    attention-mask dict ready for model(**batch).
    """
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

    log.info(f"Worker started | device={label} | pid={os.getpid()}")

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

        valid_rows: list[dict]       = []
        valid_ids:  list[list[int]]  = []   # pre-computed input_ids, one per row (ref tokenizer)

        # per-file counters
        n_no_mask        = 0
        n_mask_lost      = 0
        n_ok_full        = 0
        n_ok_trunc       = 0
        n_too_long       = 0    # rows where prev_sentence was dropped (subset of ok_trunc)
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
                    # send to unprocessable log
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

        # write filter-dropped sidecar
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
                "skipped": False,
                "filepath": str(filepath),
                "rows_written": 0,
                "gpu_label": label,
            })
            continue

        log.info(
            f"Running inference | device={label} | file={filepath.name} | rows={len(valid_rows)}"
        )

        predictions: dict[str, list] = {}

        for col_name, mdl in models.items():
            tokenizer = tokenizers[col_name]
            col_preds = []

            # Re-encode with this model's tokenizer when it differs from the
            # reference tokenizer used to build valid_rows/valid_ids, otherwise
            # reuse the already-computed ids.
            same_vocab = (tokenizer is ref_tokenizer)

            for batch_start in range(0, len(valid_rows), BATCH_SIZE):
                batch_rows = valid_rows[batch_start : batch_start + BATCH_SIZE]
                batch_ref_ids = valid_ids[batch_start : batch_start + BATCH_SIZE]

                # keep_idx[k] = position within batch_rows that batch_ids[k]
                # corresponds to. Needed because, for a differing tokenizer,
                # some rows may need to be skipped (mask doesn't survive
                # truncation under THIS vocabulary) without shifting the
                # alignment of the rows that do succeed.
                if same_vocab:
                    batch_ids = batch_ref_ids
                    keep_idx  = list(range(len(batch_rows)))
                else:
                    # Token counts are vocabulary-specific: the same text can
                    # fit under 512 tokens for the reference tokenizer and
                    # overflow for this one (or vice versa). Re-run the same
                    # vetted prepare_text() logic — which drops prev_sentence
                    # and verifies the mask survives truncation — using THIS
                    # tokenizer, instead of a naive truncate-and-hope.
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

                # batch_preds defaults to [] for any row skipped above; rows
                # that succeed get overwritten with real predictions below.
                batch_preds = [[] for _ in batch_rows]

                if not batch_ids:
                    col_preds.extend(batch_preds)
                    continue

                # Pad pre-computed ids into tensors — no string re-encoding
                encoded = build_batch_tensors(
                    batch_ids,
                    pad_token_id=tokenizer.pad_token_id,
                    device=device,
                )

                with torch.no_grad():
                    outputs = mdl(**encoded)

                input_ids_tensor = encoded["input_ids"]

                for j in range(input_ids_tensor.shape[0]):
                    mask_positions = (
                        input_ids_tensor[j] == tokenizer.mask_token_id
                    ).nonzero(as_tuple=True)[0]

                    if len(mask_positions) == 0:
                        # Should not happen now that prepare_text() already
                        # verified the mask survives for this tokenizer, but
                        # guard and log to the unprocessable file just in case.
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
                    probs    = torch.softmax(logits, dim=-1)
                    top      = torch.topk(probs, TOP_K)

                    batch_preds[keep_idx[j]] = [
                        (tokenizer.decode([tok_id]).strip(), round(score, 4))
                        for tok_id, score in zip(
                            top.indices.tolist(), top.values.tolist()
                        )
                    ]

                col_preds.extend(batch_preds)

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
    log.info(f"Run started:       {RUN_TIMESTAMP}")
    log.info(f"Input dir:         {INPUT_DIR}")
    log.info(f"Input glob:        {INPUT_GLOB}")
    log.info(f"Output dir:        {OUTPUT_DIR}")
    log.info(f"Log file:          {LOG_FILE}")
    log.info(f"Unprocessable log: {UNPROCESSABLE_LOG}")

    n_gpus = min(torch.cuda.device_count(), MAX_GPUS) or 1
    log.info(f"GPUs to use:       {n_gpus}")

    input_path  = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)

    matched = sorted(input_path.glob(INPUT_GLOB))
    if not matched:
        words = ["machine", "machines", "slave", "slaves"]
        matched = sorted(
            fp
            for word in words
            for fp in input_path.glob(f"bl_microsoft_{word}_spacy_step_5.jsonl")
        )

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

    log_listener = start_log_listener(log_queue, LOG_FILE, UNPROCESSABLE_LOG)

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

    log_queue.put(None)
    log_listener.join()

    log.info("All workers finished.")


if __name__ == "__main__":
    main()
