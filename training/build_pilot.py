"""Derive the JSON-only pilot without altering historical v1 artifacts."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def build(root: Path):
    source = root / "training"
    freeze = json.loads((root / "eval/freeze.json").read_text())
    combos = freeze["combo_arg_pools"]["convert"]
    frozen = {r["prompt"] for p in (root / "eval").glob("eval_*.jsonl") for r in _read(p)}
    rows, meta, seen = [], [], {}
    for line, (row, m) in enumerate(zip(_read(source / "train_v1.jsonl"),
                                         _read(source / "train_v1_meta.jsonl"), strict=True), 1):
        if m["kind"] == "general" or m["template_id"] == "train_clarify_invalid":
            continue
        messages = row["messages"]
        prompt = messages[1]["content"]
        call = json.loads(messages[2]["content"])
        tool = call["tool"]
        if tool in {"trim", "cut_out"}:
            if "%" in prompt or "half" in prompt:
                continue
            # v1's mm:ss renderer truncates fractional seconds.
            if re.search(r"\d+:\d{2}", prompt) and any(call[k] % 1 for k in ("start", "end")):
                continue
        if tool == "scale" and not re.search(rf"\b{call['width']}(?:x| by ){call['height']}\b", prompt):
            continue
        if tool == "concat" and len(call["inputs"]) == 3 and not re.search(r"\bthree\b", prompt):
            continue
        if tool == "convert":
            call["crf"] = 23  # None of the v1 conversion prompts requests a CRF.
            if [call["codec"], call["container"], call["crf"]] in combos:
                continue
        if tool == "extract_audio" and (call["input"] not in prompt or
                                       not re.search(rf"(?<!\.)\b{call['format']}\b", prompt)):
            continue
        if (prompt in frozen or m["template_id"] in freeze["heldout_template_ids"]
                or m["fixture"] == freeze["reserved_fixture"]):
            raise ValueError(f"pilot inherited a frozen split collision on source line {line}")
        target = {k: v for k, v in call.items() if k not in {"output", "question"}}
        if prompt in seen:
            if seen[prompt] != target:
                raise ValueError(f"conflicting pilot targets on source line {line}: {prompt!r}")
            continue
        seen[prompt] = target
        rows.append({"messages": [*messages[:-1], {"role": "assistant", "content": json.dumps(call)}]})
        meta.append({**m, "source_line": line})
    return rows, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "training")
    args = parser.parse_args()
    rows, meta = build(ROOT)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, values in (("train_pilot_v1", rows), ("train_pilot_v1_meta", meta)):
        (args.out / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in values))
    print(f"Pilot rows: {len(rows)}; {dict(sorted(Counter(m['tool'] for m in meta).items()))}")


if __name__ == "__main__":
    main()
