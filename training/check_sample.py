"""Wrapper execution check on a random sample of training examples.

For each sampled tool example (clarify and general are excluded —
they involve no ffmpeg), copy the example's fixture to clip.mp4 in a
fresh workdir, run wrapper.edit() with the example's label, and check
the result is ok. The input filename in the label is rewritten to
clip.mp4 so the harness convention holds; the label's args are
unchanged.

A failure is either a bad label (generator bug) or a wrapper bug —
print every failure with its label so the cause can be told apart.

Usage:
    python -m training.check_sample --n 200 --seed 20261122
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minicut.wrapper import edit  # noqa: E402

FIXTURES_DIR = ROOT / "factory" / "fixtures"
FILE = "clip.mp4"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261122)
    ap.add_argument("--data", default="training/train_v1.jsonl")
    ap.add_argument("--meta", default="training/train_v1_meta.jsonl")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.data).read_text().splitlines()]
    meta = [json.loads(l) for l in Path(args.meta).read_text().splitlines()]
    assert len(rows) == len(meta)
    pool = [(r, m) for r, m in zip(rows, meta) if m["kind"] == "tool"]
    rng = random.Random(args.seed)
    sample = rng.sample(pool, min(args.n, len(pool)))
    ok, failures = 0, []
    for i, (row, m) in enumerate(sample):
        label = json.loads(row["messages"][2]["content"])
        workdir = Path(tempfile.mkdtemp())
        try:
            shutil.copy(FIXTURES_DIR / m["fixture"] / "input.mp4", workdir / FILE)
            call = dict(label)
            if "input" in call:
                call["input"] = FILE
            if "inputs" in call:
                call["inputs"] = [FILE] * len(call["inputs"])
            r = edit(call, workdir)
            if r.get("ok"):
                ok += 1
            else:
                failures.append((m["tool"], m["fixture"], label, str(r.get("error"))[:200]))
        except Exception as e:  # noqa: BLE001 — report, don't crash the check
            failures.append((m["tool"], m["fixture"], label, f"exception: {e}"))
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(sample)} checked, {ok} ok", flush=True)
    print(f"\nwrapper sample check: {ok}/{len(sample)} passed "
          f"({100.0 * ok / len(sample):.1f}%)")
    for tool, fixture, label, err in failures:
        print(f"FAIL {tool} [{fixture}] {label} -> {err}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
