"""Differential test: Rust `minicut validate` vs Python `minicut.validate`.

Samples (call, probe) pairs on the ms / 0.001-factor grid, including
edges, and requires the same accept/reject verdict from both. Known,
intended divergences are listed in DIVERGENCES and skipped by name.

    python scripts/parity_validate.py --bin target/debug/minicut --n 20000
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from minicut.probe import Probe  # noqa: E402
from minicut.validate import validate  # noqa: E402

# Intended divergences: (name, predicate on the call, expected rust verdict,
# expected python verdict). A mismatch matching one of these is reported by
# name and does not fail the run; any other mismatch does.
DIVERGENCES = [
    ("extra 'inputs' key on a single-input tool: Rust rejects unknown fields "
     "(docs/rust-port.md open question 5)",
     lambda call: call["tool"] not in ("concat", "clarify") and "inputs" in call, False, True),
]

TOOLS = ["trim", "cut_out", "concat", "scale", "speed", "convert", "extract_audio", "mute", "clarify"]


def ms(rng, hi_ms):
    return rng.randint(0, hi_ms) / 1000


def sample(rng: random.Random):
    dur_ms = rng.choice([1000, 10000, 30000, 30023, 60000, rng.randint(500, 120000)])
    has_video = rng.random() > 0.1
    probe = {"duration_s": dur_ms / 1000,
             "width": rng.choice([1280, 640, 1920]) if has_video else None,
             "height": rng.choice([720, 360, 1080]) if has_video else None,
             "has_audio": rng.random() > 0.2}
    tool = rng.choice(TOOLS)
    call: dict = {"tool": tool}
    if tool == "concat":
        call["inputs"] = ["a.mp4"] * rng.choice([1, 2, 3])
    elif tool != "clarify":
        call["input"] = rng.choice(["clip.mp4", "clip.mp4", "../x.mp4", "/abs.mp4", ""])
    if tool in ("trim", "cut_out"):
        edge = [0, 50, dur_ms - 50, dur_ms, dur_ms + 1]
        pick = lambda: rng.choice(edge) / 1000 if rng.random() < 0.4 else ms(rng, dur_ms + 2000)
        call["start"], call["end"] = pick(), pick()
    elif tool == "scale":
        call["width"] = rng.choice([0, 2, 639, 640, 1280, 1920, probe["width"] or 2])
        call["height"] = rng.choice([0, 2, 359, 360, 720, 1080, probe["height"] or 2])
    elif tool == "speed":
        call["factor"] = rng.choice([0.249, 0.25, 0.5, 1.0, 1.25, 2.0, 4.0, 4.001, rng.randint(200, 4100) / 1000])
    elif tool == "convert":
        call["codec"] = rng.choice(["libx264", "libx265", "vp9"])
        call["container"] = rng.choice(["mp4", "mkv", "webm"])
        if rng.random() < 0.5:
            call["crf"] = rng.choice([0, 23, 51])
    elif tool == "extract_audio":
        call["format"] = rng.choice(["mp3", "wav", "aac"])
    elif tool == "clarify":
        call["question"] = rng.choice(["Which times?", "   "])
        call["missing"] = ["start", "end"]
    if tool not in ("concat", "clarify") and rng.random() < 0.02:
        call["inputs"] = ["a.mp4", "b.mp4"]  # see DIVERGENCES
    if tool != "clarify" and rng.random() < 0.15:
        # validate.py path-checks `output` too; both sides must agree. (The
        # Python CLI/wrapper strip `output` before validating; Phase 2's Rust
        # CLI must do the same: docs/rust-port.md, Phase 2.)
        call["output"] = rng.choice(["out.mp4", "renders/out.mp4", "/abs.mp4", "../up.mp4", ""])
    return call, probe


def py_ok(call, probe) -> bool:
    p = Probe(path=Path("clip.mp4"), duration_s=probe["duration_s"], width=probe["width"],
              height=probe["height"], fps=30.0, video_codec=None, audio_codec=None,
              has_audio=probe["has_audio"], has_video=probe["width"] is not None, format_name=None)
    return validate(call, p, Path(".")) == []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rows = [sample(rng) for _ in range(args.n)]
    feed = "".join(json.dumps({"call": json.dumps(c), "probe": p}) + "\n" for c, p in rows)
    out = subprocess.run([args.bin, "validate", "--batch"], input=feed, capture_output=True,
                         text=True, check=True).stdout.splitlines()
    assert len(out) == len(rows)
    mismatches, known = Counter(), Counter()
    examples = {}
    verdicts = Counter()
    for (call, probe), line in zip(rows, out):
        rust, py = json.loads(line)["ok"], py_ok(call, probe)
        verdicts[(call["tool"], py)] += 1
        if rust == py:
            continue
        name = next((n for n, pred, r, p in DIVERGENCES if pred(call) and (rust, py) == (r, p)), None)
        if name:
            known[name] += 1
            continue
        key = f"{call['tool']}: rust={rust} python={py}"
        mismatches[key] += 1
        examples.setdefault(key, (call, probe))
    for tool in TOOLS:
        print(f"  {tool:14s} python accepts {verdicts[(tool, True)]:5d}, rejects {verdicts[(tool, False)]:5d}")
    for name, n in known.items():
        print(f"  known divergence x{n}: {name}")
    print(f"{len(rows)} cases, {sum(mismatches.values())} mismatches")
    for k, v in mismatches.most_common():
        print(f"  {v:5d}  {k}  e.g. {json.dumps(examples[k][0])} probe={json.dumps(examples[k][1])}")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
