import json
import re
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/gpfs/projects/bsc100/models/Qwen/Qwen3.5-9B/"
BATCH_SIZE = 16
MAX_NEW_TOKENS = 1024

PROMPT = "Clean up the paragraphs of this text, keep only the paragraphs that belong to the narrative"

# Output-format instruction only; the task prompt above is unchanged
FORMAT = (
    'Reply with a single JSON object and nothing else: '
    '{"keep": true or false, "text": "<the cleaned paragraph, or empty string if keep is false>"}'
)


def build_messages(paragraph: str):
    return [{"role": "user", "content": f"{PROMPT}\n\n{FORMAT}\n\nParagraph:\n{paragraph}"}]


def parse(raw: str):
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        if isinstance(obj.get("keep"), bool) and isinstance(obj.get("text"), str):
            return obj
    except json.JSONDecodeError:
        pass
    return None


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="left")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
        )
    except (ValueError, KeyError):
        # Some Qwen3.5 checkpoints are multimodal and need this class instead
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
        )
    model.eval()
    return tokenizer, model


def main(in_path: str, out_path: str):
    with open(in_path, encoding="utf-8") as f:
        paragraphs = json.load(f)

    tokenizer, model = load_model()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = [
        tokenizer.apply_chat_template(
            build_messages(p), tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        for p in paragraphs
    ]

    # Sort by length so batches have similar sizes (less padding), restore order afterwards
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    results = [None] * len(prompts)

    for start in range(0, len(order), BATCH_SIZE):
        idx = order[start:start + BATCH_SIZE]
        batch = [prompts[i] for i in idx]
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen = out[:, inputs["input_ids"].shape[1]:]
        texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
        for i, t in zip(idx, texts):
            results[i] = t
        print(f"processed {min(start + BATCH_SIZE, len(order))}/{len(order)}", flush=True)

    cleaned, review = [], []
    for i, (p, raw) in enumerate(zip(paragraphs, results)):
        res = parse(raw)
        if res is None:
            review.append({"index": i, "reason": "invalid JSON", "original": p, "raw": raw})
            print(f"[{i}] invalid JSON, flagged")
            continue
        if not res["keep"]:
            print(f"[{i}] dropped")
            continue
        ratio = len(res["text"]) / max(len(p), 1)
        if ratio < 0.6 or ratio > 1.1:
            review.append({"index": i, "reason": f"length ratio {ratio:.2f}",
                           "original": p, "cleaned": res["text"]})
            print(f"[{i}] kept (flagged, ratio {ratio:.2f})")
        else:
            print(f"[{i}] kept")
        cleaned.append(res["text"])

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)
    if review:
        with open(out_path.replace(".json", "_review.json"), "w", encoding="utf-8") as f:
            json.dump(review, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # usage: python postcorrect.py paragraphs.json cleaned.json
    main(sys.argv[1], sys.argv[2])