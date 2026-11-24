"""Drop-only freeze gate between rows/meta from the Rust generator and a
frozen train_verified_v1 split: cuts anything that collides with the eval
freeze (eval/freeze.json) or that is malformed, otherwise passes rows
through byte-identical.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from training.generate_dataset import _canon, _load_eval_collisions

ROOT = Path(__file__).resolve().parents[1]


def _read_lines(path: Path) -> list[str]:
    return path.read_text().splitlines()


def build(rows_path: Path, meta_path: Path, root: Path = ROOT):
    from minicut.llm.prompts import SYSTEM_PROMPT

    freeze = json.loads((root / "eval/freeze.json").read_text())
    heldout = set(freeze["heldout_template_ids"])
    reserved_fixture = freeze["reserved_fixture"]
    prompt_canon, tid_canon, eval_prompts = _load_eval_collisions()

    row_lines = _read_lines(rows_path)
    meta_lines = _read_lines(meta_path)
    if len(row_lines) != len(meta_lines):
        raise ValueError(f"rows/meta line count mismatch: {len(row_lines)} vs {len(meta_lines)}")

    kept_rows, kept_meta = [], []
    dropped_by_reason: Counter[str] = Counter()
    kept_by_tool: Counter[str] = Counter()
    seen_dupe: set[tuple[str, str]] = set()

    for row_line, meta_line in zip(row_lines, meta_lines):
        try:
            row = json.loads(row_line)
            meta = json.loads(meta_line)
            messages = row["messages"]
            if (len(messages) != 3 or messages[0] != {"role": "system", "content": SYSTEM_PROMPT}
                    or messages[0]["role"] != "system" or messages[1]["role"] != "user"
                    or messages[2]["role"] != "assistant"):
                raise ValueError("bad shape")
            prompt = messages[1]["content"]
            label = json.loads(messages[2]["content"])
        except Exception:
            dropped_by_reason["malformed"] += 1
            continue

        canon = _canon(label)
        template_id = meta.get("template_id")
        fixture = meta.get("fixture")
        dupe_key = (prompt, json.dumps(label, sort_keys=True))

        reason = None
        if template_id in heldout:
            reason = "heldout_template"
        elif fixture == reserved_fixture:
            reason = "reserved_fixture"
        elif (prompt, canon) in prompt_canon:
            reason = "eval_prompt_canon_collision"
        elif (template_id, canon) in tid_canon:
            reason = "eval_tid_canon_collision"
        elif prompt in eval_prompts:
            reason = "eval_prompt_collision"
        elif dupe_key in seen_dupe:
            reason = "duplicate"

        if reason is not None:
            dropped_by_reason[reason] += 1
            continue

        seen_dupe.add(dupe_key)
        kept_rows.append(row_line)
        kept_meta.append(meta_line)
        kept_by_tool[label.get("tool")] += 1

    stats = {
        "input": len(row_lines),
        "kept": len(kept_rows),
        "dropped_by_reason": dict(sorted(dropped_by_reason.items())),
        "kept_by_tool": dict(sorted(kept_by_tool.items())),
    }
    return kept_rows, kept_meta, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    kept_rows, kept_meta, stats = build(args.rows, args.meta)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "train_verified_v1.jsonl").write_text("".join(line + "\n" for line in kept_rows))
    (args.out / "train_verified_v1_meta.jsonl").write_text("".join(line + "\n" for line in kept_meta))
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
