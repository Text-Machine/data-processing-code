import os
import pandas as pd
import torch
import torch.nn as nn

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


# INPUT_CSV is the results_prompts.csv written by preprocess_instruction_tuning.py:
# one row per paragraph, with a "prompt" column (the LLM-generated one-sentence
# summary) and an "expected_output" column (the original paragraph text).
INPUT_CSV = "/gpfs/scratch/bsc100/paolo/tm96_finetuning/data-processing-code/finetuning/outputs/results_prompts.csv"
MODEL_ID = "/gpfs/projects/bsc100/models/Qwen/Qwen3.5-9B/"
OUTPUT_DIR = "/gpfs/scratch/bsc100/paolo/tm96_finetuning/data-processing-code/finetuning/outputs/"
MAX_SEQ_LENGTH = 1024
TEST_SIZE = 0.05
SEED = 42

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

df = pd.read_csv(INPUT_CSV)

# Turn each (summary, paragraph) pair into an instruction: "write the full
# paragraph, in the source author's style, that matches this summary."
instruction = "Write a paragraph, in the style of the source novel, based on this one-sentence summary:"
df['prompt'] = df.apply(lambda x: instruction + " " + x.prompt, axis=1)
df['completions'] = df['expected_output']
df[['prompt', 'completions']].to_csv('oliver_twist_sft_formatted.csv')

df = pd.read_csv('oliver_twist_sft_formatted.csv', index_col=0)
required_cols = {"prompt", "completions"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

df = df.dropna(subset=["prompt", "completions"]).copy()
df["prompt"] = df["prompt"].astype(str)
df["completions"] = df["completions"].astype(str)

print("Rows after cleaning:", len(df))
df.head()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

START_USER = "<start_of_turn>user\n"
START_MODEL = "<start_of_turn>model\n"
END_TURN = "<end_of_turn>\n"

def to_text(example):
    prompt = example["prompt"].strip()
    completion = example["completions"].strip()
    text = f"{START_USER}{prompt}{END_TURN}{START_MODEL}{completion}{END_TURN}"
    return {"text": text}

dataset = Dataset.from_pandas(df[["prompt", "completions"]], preserve_index=False)
dataset = dataset.map(to_text, remove_columns=dataset.column_names)
dataset = dataset.train_test_split(test_size=TEST_SIZE, seed=SEED)

train_dataset = dataset["train"]
eval_dataset = dataset["test"]

print(train_dataset)
print(eval_dataset)
print("Sample formatted record:\n", train_dataset[0]["text"][:500])

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="eager")
model.config.use_cache = False

try:
    import bitsandbytes as bnb
    LinearTypes = (nn.Linear, bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)
except Exception:
    LinearTypes = (nn.Linear,)

TEXT_LORA_SUFFIXES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj")

SKIP_BRANCHES = (
    "vision", "visual", "audio",
    "multimodal", "multi_modal", "mm_projector")

target_modules = []
for name, module in model.named_modules():
    lname = name.lower()
    if any(branch in lname for branch in SKIP_BRANCHES):
        continue
    if not name.endswith(TEXT_LORA_SUFFIXES):
        continue
    if isinstance(module, LinearTypes):
        target_modules.append(name)
print(f"Found {len(target_modules)} LoRA target modules")
print("\n".join(target_modules[:40]))

assert target_modules, "No LoRA target modules found; inspect model.named_modules()."

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=target_modules)

sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=5,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=0.03,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_steps=100,
    save_total_limit=2,
    bf16=True,
    fp16=False,
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    report_to="none",
    seed=SEED,
    max_length=MAX_SEQ_LENGTH,
    dataset_text_field="text",
    packing=False)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    args=sft_config,
    processing_class=tokenizer)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Training complete. Adapter saved to:", OUTPUT_DIR)

model.eval()

test_prompt = "Write a paragraph, in the style of the source novel, based on this one-sentence summary: A boy is born into hardship in a workhouse town."
formatted = f"{START_USER}{test_prompt}{END_TURN}{START_MODEL}"
inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=180,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id)

print(tokenizer.decode(output[0], skip_special_tokens=False))