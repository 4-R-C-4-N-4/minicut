"""Train the MiniCut LoRA adapter on a local GPU.

Runs on Ivy's GPU machine, NOT on the CPU-only VM where the dataset
was built. Framework: plain transformers Trainer + PEFT (no TRL),
chosen so the training loop is short, readable, and version-stable —
one file, no framework-specific config dialect.

What it does:
- Loads openbmb/MiniCPM5-2B (bf16) and its tokenizer.
- Renders each train_v1.jsonl example with the tokenizer's own chat
  template (the same template llama.cpp uses at inference, thinking
  off — the examples carry no reasoning, assistant = JSON only).
- Trains on the assistant completion only: prompt tokens are masked
  with -100, so loss never teaches the model to predict the prompt.
- LoRA r=16, alpha=32, lr=1e-4, 2 epochs (see lora_config.yaml).

Run:
    pip install -r training/requirements-gpu.txt
    python -m training.train_lora --data training/train_v1.jsonl --out training/output/lora_v1

Then convert the adapter to GGUF and eval it (see training/README.md):
    python llama.cpp/convert_lora_to_gguf.py --base models/MiniCPM5-2B-F16.gguf \
        --out training/output/lora_v1.gguf training/output/lora_v1
    llama-server -m models/MiniCPM5-2B-F16.gguf --lora training/output/lora_v1.gguf \
        --jinja -c 8192 --port 8080
    python -m eval.run_eval --all --backend http --label lora_v1 --out eval/results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


class ChatDataset(Dataset):
    """One row per train_v1.jsonl line; completion-only labels."""

    def __init__(self, path: Path, tokenizer, max_len: int = 1024):
        self.rows = []
        for line in open(path):
            line = line.strip()
            if line:
                self.rows.append(json.loads(line))
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        messages = self.rows[i]["messages"]
        # Prompt = system + user, with the generation prompt appended.
        prompt_text = self.tok.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )
        # Full = prompt + assistant reply + EOS.
        full_text = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        eos = self.tok.eos_token or ""
        if eos and not full_text.endswith(eos):
            full_text += eos
        prompt_ids = self.tok(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = self.tok(full_text, add_special_tokens=False)["input_ids"][: self.max_len]
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        labels = labels[: self.max_len]
        return {"input_ids": full_ids, "labels": labels}


def _collate(batch: list[dict], pad_id: int) -> dict:
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        n = len(b["input_ids"])
        pad = maxlen - n
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        attn.append([1] * n + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="training/train_v1.jsonl")
    ap.add_argument("--base", default="openbmb/MiniCPM5-2B")
    ap.add_argument("--out", default="training/output/lora_v1")
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=1024)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU found — this script is meant for the local GPU machine.")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        r=args.r, lora_alpha=args.alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    ds = ChatDataset(Path(args.data), tok, max_len=args.max_len)
    print(f"training examples: {len(ds)}")
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to=[],
    )
    trainer = Trainer(
        model=model, args=targs, train_dataset=ds,
        data_collator=lambda b: _collate(b, pad_id),
    )
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"adapter saved to {args.out}")


if __name__ == "__main__":
    main()
