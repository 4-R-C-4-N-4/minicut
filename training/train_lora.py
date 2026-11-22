"""CPU tokenizer preflight and BF16 LoRA training; see training/README.md."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from functools import partial
from pathlib import Path

from minicut.llm.prompts import SYSTEM_PROMPT, edit_messages

TURN_END = "<|im_end|>"


def completion_example(messages: list[dict], tok, max_len: int) -> dict:
    answer = messages[-1]["content"]
    if not isinstance(answer, str) or not answer.strip() or TURN_END in answer:
        raise ValueError("empty or invalid assistant target")
    kwargs = {"tokenize": False, "enable_thinking": False}
    prefix = tok.apply_chat_template(messages[:-1], add_generation_prompt=True, **kwargs)
    complete = tok.apply_chat_template(messages, add_generation_prompt=False, **kwargs)
    target = answer + TURN_END
    if complete != prefix + target + "\n":
        raise ValueError("unexpected MiniCPM chat template or assistant boundary")
    prefix_ids = tok(prefix, add_special_tokens=False)["input_ids"]
    ids = tok(complete.removesuffix("\n"), add_special_tokens=False)["input_ids"]
    if not prefix_ids or ids[:len(prefix_ids)] != prefix_ids:
        raise ValueError("tokenized serving prompt is not an exact prefix")
    if len(ids) > max_len:
        raise ValueError(f"row has {len(ids)} tokens, exceeds max length {max_len}")
    suffix = ids[len(prefix_ids):]
    if not suffix or tok.decode(suffix, skip_special_tokens=False,
                                clean_up_tokenization_spaces=False) != target:
        raise ValueError("supervised tokens do not decode to answer and turn end")
    return {"input_ids": ids, "labels": [-100] * len(prefix_ids) + suffix}


def prepare_rows(data: bytes, tok, max_len: int) -> list[dict]:
    rows = []
    for line_no, line in enumerate(data.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            system, user, assistant = json.loads(line)["messages"]
            if system != {"role": "system", "content": SYSTEM_PROMPT} or (
                user["role"], assistant["role"]
            ) != ("user", "assistant"):
                raise ValueError("row does not follow the deployed chat contract")
            call = json.loads(assistant["content"])
            if not isinstance(call, dict) or not isinstance(call.get("tool"), str):
                raise ValueError("assistant target must be a JSON tool call")
            rows.append(completion_example([*edit_messages(user["content"]), assistant],
                                           tok, max_len))
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"row {line_no}: {exc}") from exc
    if not rows:
        raise ValueError("no training rows")
    return rows


def _collate(batch: list[dict], pad_id: int) -> dict:
    import torch

    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        n = len(b["input_ids"])
        pad = maxlen - n
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        attn.append([1] * n + [0] * pad)
    return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("training/train_pilot_v1.jsonl"))
    ap.add_argument("--base", required=True, help="Local HF snapshot directory or cached model ID")
    ap.add_argument("--revision", help="Pinned HF commit, required for a model ID")
    ap.add_argument("--out", type=Path, default=Path("training/output/lora_pilot_v1"))
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--preflight", action="store_true", help="Cached tokenizer only, no model or CUDA")
    ap.add_argument("--max-steps", type=int, help="Bound optimizer updates and verify LoRA changed")
    ap.add_argument("--max-seconds", type=float, help="Stop training at the next update boundary")
    ap.add_argument("--checkpoint-steps", type=int, help="Save intermediate bounded-run checkpoints")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if not Path(args.base).is_dir() and not args.revision:
        ap.error("provide a local --base directory or a cached --base ID with --revision")
    if args.max_steps is not None and args.max_steps < 1:
        ap.error("--max-steps must be positive")
    if any(v is not None and (not math.isfinite(v) or v <= 0)
           for v in (args.max_seconds, args.checkpoint_steps)):
        ap.error("time and checkpoint limits must be finite and positive")

    from transformers import AutoTokenizer

    checkpoint = {"revision": args.revision, "trust_remote_code": False, "local_files_only": True}
    tok = AutoTokenizer.from_pretrained(args.base, **checkpoint)
    data = args.data.read_bytes()
    ds = prepare_rows(data, tok, args.max_len)
    digest = hashlib.sha256(data).hexdigest()
    print(f"preflight: {len(ds)} rows; max sequence {max(len(r['input_ids']) for r in ds)} tokens; "
          f"min supervised {min(sum(i != -100 for i in r['labels']) for r in ds)} tokens; "
          f"data sha256={digest}")
    if args.preflight:
        return
    if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
        ap.error("--out must be a new or empty directory")

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, Trainer, TrainerCallback, TrainingArguments, set_seed

    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise SystemExit("Expose exactly one BF16-capable CUDA device for this run.")
    gpu = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu}")
    set_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, attn_implementation="sdpa", **checkpoint,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(
        r=args.r, lora_alpha=args.alpha, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    model.print_trainable_parameters()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if pad_id is None:
        raise ValueError("tokenizer has no pad or EOS token")

    class RunGuard(TrainerCallback):
        timed_out = False

        def on_train_begin(self, args, state, control, **kwargs):
            self.started = time.monotonic()

        def on_step_end(self, args, state, control, **kwargs):
            if time_budget is not None and time.monotonic() - self.started >= time_budget:
                self.timed_out = True
                control.should_training_stop = True

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and any(not math.isfinite(logs[k]) for k in ("loss", "train_loss") if k in logs):
                raise RuntimeError("nonfinite training loss")

    smoke = args.max_steps is not None
    time_budget = args.max_seconds
    guard = RunGuard()
    lora_b = [(p, p.detach().cpu().clone()) for name, p in model.named_parameters()
              if smoke and p.requires_grad and ".lora_B." in name]
    targs = TrainingArguments(
        output_dir=str(args.out), per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum, num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps is not None else -1,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.03, bf16=True,
        logging_steps=1 if smoke else 10, logging_nan_inf_filter=False,
        save_strategy="steps" if args.checkpoint_steps else ("no" if smoke else "epoch"),
        save_steps=args.checkpoint_steps or 500, save_total_limit=5, report_to=[], seed=args.seed,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=partial(_collate, pad_id=pad_id), callbacks=[guard])
    train_started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    result = trainer.train()
    train_seconds = time.monotonic() - train_started
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    if smoke:
        changed = False
        for param, before in lora_b:
            after = param.detach().cpu()
            if not torch.isfinite(after).all():
                raise RuntimeError("nonfinite LoRA weights after smoke")
            changed |= not torch.equal(before, after)
        if not changed or (not guard.timed_out and trainer.state.global_step != args.max_steps):
            raise RuntimeError("smoke did not finish requested updates with a changed LoRA weight")
        print(f"bounded run: {trainer.state.global_step} updates; LoRA B changed")
    trainer.save_model(str(args.out))
    tok.save_pretrained(args.out)
    manifest = {**vars(args), "gpu": gpu, "data_sha256": digest, "rows": len(ds),
                "optimizer_steps": trainer.state.global_step,
                "stop_reason": "time_limit" if guard.timed_out else "completed",
                "train_seconds": train_seconds, "train_loss": result.training_loss,
                "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved}
    (args.out / "run.json").write_text(json.dumps(manifest, default=str, indent=2) + "\n")
    print(f"adapter saved to {args.out}")


if __name__ == "__main__":
    main()
