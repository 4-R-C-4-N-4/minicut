"""Generate MiniCut Phase 3 eval sets — frozen before any training.

Writes four JSONL files under eval/:
  eval_seen.jsonl            — train-pool paraphrases, new arg values
  eval_unseen_paraphrase.jsonl — held-out paraphrases (never in training)
  eval_unseen_combo.jsonl    — train-pool paraphrases, arg combos never in training
  eval_unseen_fixture.jsonl  — long_60s_1080p fixture only (reserved; never in training)

Freeze rules (see eval/README.md + freeze.json written by this script):
  * Seed is fixed (20261120). Regenerating with the same seed must
    reproduce the same files byte-for-byte.
  * Held-out paraphrase templates (ids ending _heldout) must NEVER be
    used in Phase 4 training data.
  * The long_60s_1080p fixture must NEVER be used in training data.
  * Unseen-combo arg values (the COMBO_* pools below) must NEVER be
    used in training data.
  * Every case's ground-truth JSON is validated against its fixture's
    probe here (no ffmpeg run) — a case that fails validation is a
    generator bug, and generation aborts.

Filename convention: every prompt says "clip.mp4" and the expected
call uses input "clip.mp4", matching the few-shot examples in
minicut/llm/prompts.py. The harness copies the fixture to clip.mp4
in a fresh workdir per case, so the model is never punished for the
filename the few-shot examples taught it.

Usage:
    python -m eval.generate_eval --out eval
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minicut.probe import probe_file  # noqa: E402
from minicut.validate import validate  # noqa: E402

SEED = 20261120
FIXTURES_DIR = ROOT / "factory" / "fixtures"
FILE = "clip.mp4"

# Fixture durations, from factory manifests (read at generation time
# for validation; these are the known values, not guesses).
FIXTURE_DURATIONS = {
    "base_30s": 30.0,
    "short_10s": 10.0,
    "silent_30s": 30.0,
    "portrait_30s": 30.0,
    "long_60s_1080p": 60.0,
}
# Fixtures allowed per split. long_60s_1080p is reserved for unseen_fixture.
SEEN_FIXTURES = ["base_30s", "short_10s", "silent_30s", "portrait_30s"]
UNSEEN_FIXTURE = "long_60s_1080p"

# --- Prompt templates -------------------------------------------------
# Each template: (template_id, render function). render(args, fixture)
# returns the prompt string. Train-pool ids end in _tN; held-out ids
# end in _hN and must never appear in training data.


def _mmss(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def _fmt_secs(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else str(v)


# trim: args {start, end}
TRIM_TRAIN = [
    ("trim_t1", lambda a, f: f"trim {FILE} from {_fmt_secs(a['start'])} seconds to {_fmt_secs(a['end'])} seconds"),
    ("trim_t2", lambda a, f: f"trim {FILE} from {_mmss(a['start'])} to {_mmss(a['end'])}"),
    ("trim_t3", lambda a, f: f"keep only {_fmt_secs(a['start'])} to {_fmt_secs(a['end'])} seconds of {FILE}"),
    ("trim_t4", lambda a, f: f"cut {FILE} down to {_fmt_secs(a['start'])}-{_fmt_secs(a['end'])} seconds"),
    ("trim_t5", lambda a, f: f"trim {FILE} to the first {_fmt_secs(a['end'])} seconds" if a["start"] == 0 else f"trim {FILE} from {_fmt_secs(a['start'])} to {_fmt_secs(a['end'])}"),
]
TRIM_HELDOUT = [
    ("trim_h1", lambda a, f: f"give me just the {_fmt_secs(a['start'])}–{_fmt_secs(a['end'])} second stretch of {FILE}"),
    ("trim_h2", lambda a, f: f"{FILE}, only {_fmt_secs(a['start'])} to {_fmt_secs(a['end'])}"),
    ("trim_h3", lambda a, f: f"isolate seconds {_fmt_secs(a['start'])} through {_fmt_secs(a['end'])} in {FILE}"),
]

# cut_out: args {start, end}
CUT_TRAIN = [
    ("cut_t1", lambda a, f: f"cut the first {_fmt_secs(a['end'])} seconds off {FILE}" if a["start"] == 0 else f"remove {_fmt_secs(a['start'])} to {_fmt_secs(a['end'])} seconds from {FILE}"),
    ("cut_t2", lambda a, f: f"remove {_fmt_secs(a['start'])} to {_fmt_secs(a['end'])} seconds from {FILE}"),
    ("cut_t3", lambda a, f: f"cut out {_fmt_secs(a['start'])}-{_fmt_secs(a['end'])} from {FILE}"),
    ("cut_t4", lambda a, f: f"delete the middle {_fmt_secs(a['start'])} to {_fmt_secs(a['end'])} seconds of {FILE}"),
    ("cut_t5", lambda a, f: f"drop seconds {_fmt_secs(a['start'])} through {_fmt_secs(a['end'])} of {FILE}"),
]
CUT_HELDOUT = [
    ("cut_h1", lambda a, f: f"snip out {_fmt_secs(a['start'])}–{_fmt_secs(a['end'])} from {FILE}"),
    ("cut_h2", lambda a, f: f"take the {_fmt_secs(a['start'])} to {_fmt_secs(a['end'])} second bit out of {FILE}"),
    ("cut_h3", lambda a, f: f"{FILE} without the part from {_fmt_secs(a['start'])} to {_fmt_secs(a['end'])} seconds"),
]

# concat: args {inputs} — always clip.mp4 repeated; n = len(inputs)
CONCAT_TRAIN = [
    ("concat_t1", lambda a, f: f"join {FILE} to itself" if len(a["inputs"]) == 2 else f"join {FILE} to itself three times"),
    ("concat_t2", lambda a, f: f"make one video from {FILE} twice" if len(a["inputs"]) == 2 else f"make one video from {FILE} three times in a row"),
    ("concat_t3", lambda a, f: f"concatenate {FILE} with itself"),
    ("concat_t4", lambda a, f: f"combine {FILE} with itself into one file"),
]
CONCAT_HELDOUT = [
    ("concat_h1", lambda a, f: f"stick two copies of {FILE} together" if len(a["inputs"]) == 2 else f"stick three copies of {FILE} together"),
    ("concat_h2", lambda a, f: f"play {FILE} twice back to back" if len(a["inputs"]) == 2 else f"play {FILE} three times back to back"),
    ("concat_h3", lambda a, f: f"combine {FILE} with itself into one file"),
]

# scale: args {width, height}
SCALE_TRAIN = [
    ("scale_t1", lambda a, f: f"make {FILE} {a['width']}x{a['height']}"),
    ("scale_t2", lambda a, f: f"resize {FILE} to {a['width']} by {a['height']}"),
    ("scale_t3", lambda a, f: "scale {FILE} to 720p".format(FILE=FILE) if (a["width"], a["height"]) == (1280, 720) else f"resize {FILE} to {a['width']}x{a['height']}"),
    ("scale_t4", lambda a, f: "shrink {FILE} to half size".format(FILE=FILE) if (a["width"], a["height"]) == (640, 360) else f"make {FILE} {a['width']}x{a['height']}"),
]
SCALE_HELDOUT = [
    ("scale_h1", lambda a, f: f"fit {FILE} into a {a['width']}x{a['height']} frame"),
    ("scale_h2", lambda a, f: f"change {FILE} to {a['width']} pixels wide and {a['height']} tall"),
    ("scale_h3", lambda a, f: f"set the resolution of {FILE} to {a['width']}x{a['height']}"),
]

# speed: args {factor}
SPEED_TRAIN = [
    ("speed_t1", lambda a, f: f"speed up {FILE} by {a['factor']}x" if a["factor"] > 1 else f"slow {FILE} down to {a['factor']}x"),
    ("speed_t2", lambda a, f: f"make {FILE} {a['factor']} times as fast" if a["factor"] > 1 else f"play {FILE} at {a['factor']}x speed"),
    ("speed_t3", lambda a, f: "make {FILE} twice as fast".format(FILE=FILE) if a["factor"] == 2.0 else f"play {FILE} at {a['factor']}x speed" if a["factor"] == 0.5 else f"speed {FILE} up to {a['factor']}x"),
    ("speed_t4", lambda a, f: f"run {FILE} at {a['factor']}x speed"),
]
SPEED_HELDOUT = [
    ("speed_h1", lambda a, f: f"play {FILE} {a['factor']}× as fast" if a["factor"] > 1 else f"play {FILE} at {a['factor']}× speed"),
    ("speed_h2", lambda a, f: f"change the playback speed of {FILE} to {a['factor']}x"),
    ("speed_h3", lambda a, f: f"make {FILE} run at {a['factor']}x" if a["factor"] > 1 else f"slow {FILE} to {a['factor']}x"),
]

# convert: args {codec, container, crf}
CONVERT_TRAIN = [
    ("convert_t1", lambda a, f: f"convert {FILE} to {a['container']} with {a['codec']}"),
    ("convert_t2", lambda a, f: f"re-encode {FILE} as {a['codec']} in a .{a['container']} file"),
    ("convert_t3", lambda a, f: f"save {FILE} as {a['container']} using {a['codec']}"),
]
CONVERT_HELDOUT = [
    ("convert_h1", lambda a, f: f"transcode {FILE} to {a['container']} ({a['codec']})"),
    ("convert_h2", lambda a, f: f"turn {FILE} into a {a['codec']} {a['container']}"),
    ("convert_h3", lambda a, f: f"re-encode {FILE} with {a['codec']} and save it as {a['container']}"),
]

# extract_audio: args {format}
EXTRACT_TRAIN = [
    ("extract_t1", lambda a, f: f"extract the audio from {FILE} as {a['format']}"),
    ("extract_t2", lambda a, f: f"save the audio from {FILE} to {a['format']}"),
    ("extract_t3", lambda a, f: f"pull the soundtrack out of {FILE} as an {a['format']} file"),
]
EXTRACT_HELDOUT = [
    ("extract_h1", lambda a, f: f"rip the audio from {FILE} ({a['format']})"),
    ("extract_h2", lambda a, f: f"get just the sound from {FILE}, as {a['format']}"),
    ("extract_h3", lambda a, f: f"export the audio track of {FILE} to {a['format']}"),
]

# mute: no args
MUTE_TRAIN = [
    ("mute_t1", lambda a, f: f"mute {FILE}"),
    ("mute_t2", lambda a, f: f"remove the audio from {FILE}"),
    ("mute_t3", lambda a, f: f"make a silent version of {FILE}"),
    ("mute_t4", lambda a, f: f"drop the audio track from {FILE}"),
]
MUTE_HELDOUT = [
    ("mute_h1", lambda a, f: f"strip the sound from {FILE}"),
    ("mute_h2", lambda a, f: f"take the audio track off {FILE}"),
    ("mute_h3", lambda a, f: f"silence {FILE}"),
    ("mute_h4", lambda a, f: f"{FILE} with no sound"),
]

# clarify: args {question, missing} — prompt is fixed per template
CLARIFY_TRAIN = [
    ("clarify_t1", "trim it", {"question": "What start and end times should I trim to?", "missing": ["start", "end"]}),
    ("clarify_t2", "cut out the bad part", {"question": "Which start and end times should I cut out?", "missing": ["start", "end"]}),
    ("clarify_t3", "convert it", {"question": "Which codec and container should I convert to?", "missing": ["codec", "container"]}),
]
CLARIFY_HELDOUT = [
    ("clarify_h1", "trim clip.mp4 for me", {"question": "What start and end times should I trim to?", "missing": ["start", "end"]}),
    ("clarify_h2", "make it smaller", {"question": "What width and height should I scale to?", "missing": ["width", "height"]}),
]

TEMPLATES = {
    "trim": (TRIM_TRAIN, TRIM_HELDOUT),
    "cut_out": (CUT_TRAIN, CUT_HELDOUT),
    "concat": (CONCAT_TRAIN, CONCAT_HELDOUT),
    "scale": (SCALE_TRAIN, SCALE_HELDOUT),
    "speed": (SPEED_TRAIN, SPEED_HELDOUT),
    "convert": (CONVERT_TRAIN, CONVERT_HELDOUT),
    "extract_audio": (EXTRACT_TRAIN, EXTRACT_HELDOUT),
    "mute": (MUTE_TRAIN, MUTE_HELDOUT),
    "clarify": (CLARIFY_TRAIN, CLARIFY_HELDOUT),
}

# Per-split tool counts — 50 cases each.
COUNTS = {"trim": 9, "cut_out": 7, "concat": 4, "scale": 7, "speed": 6,
          "convert": 6, "extract_audio": 5, "mute": 4, "clarify": 2}

# Arg pools. SEEN_* are the ordinary values; COMBO_* values are reserved
# for the unseen_combo split and must never be used in training.
TRIM_SEEN = [(2.0, 8.0), (5.0, 15.0), (12.0, 27.0), (0.0, 5.0), (20.0, 28.0), (3.0, 9.0)]
TRIM_COMBO = [(5.0, 15.0), (8.0, 22.0), (0.0, 10.0), (15.0, 25.0)]  # cross segment boundaries
CUT_SEEN = [(10.0, 20.0), (0.0, 5.0), (5.0, 10.0), (15.0, 25.0)]
CUT_COMBO = [(10.0, 20.0), (0.0, 10.0), (5.0, 25.0), (2.0, 8.0)]  # (2,8) is the only combo entry valid on the 10s fixture
SCALE_SEEN = [(640, 360), (1280, 720), (854, 480)]
SCALE_COMBO = [(854, 480), (1920, 1080), (720, 1280)]
SPEED_SEEN = [2.0, 0.5, 1.5]
SPEED_COMBO = [0.25, 4.0, 1.5]
CONVERT_SEEN = [("libx264", "mp4", 23), ("libx265", "mkv", 23), ("libx264", "mkv", 28)]
CONVERT_COMBO = [("vp9", "webm", 23), ("libx265", "mkv", 18), ("vp9", "mkv", 23)]
EXTRACT_SEEN = ["mp3", "wav"]
EXTRACT_COMBO = ["aac", "wav"]


def _pick(rng: random.Random, pool: list):
    return pool[rng.randrange(len(pool))]


def _time_args(rng: random.Random, duration: float, pool: list,
               min_keep: float = 1.0) -> dict:
    """Sample a valid (start, end). min_keep is the shortest output the
    edit must leave behind: for trim it is end-start; for cut_out it is
    duration minus the removed span — a cut_out that removes the whole
    fixture leaves an empty concat that ffmpeg cannot build."""
    for _ in range(50):
        s, e = _pick(rng, pool)
        if 0 <= s < e <= duration and (e - s) >= min_keep:
            return {"start": float(s), "end": float(e)}
    raise RuntimeError(f"no valid time args for duration {duration} from {pool}")


def _args_for(rng: random.Random, tool: str, fixture: str, combo: bool) -> dict:
    dur = FIXTURE_DURATIONS[fixture]
    if tool == "trim":
        return _time_args(rng, dur, TRIM_COMBO if combo else TRIM_SEEN)
    if tool == "cut_out":
        # Kept footage is dur - (end - start); require >= 1s of it.
        for _ in range(50):
            args = _time_args(rng, dur, CUT_COMBO if combo else CUT_SEEN)
            if dur - (args["end"] - args["start"]) >= 1.0:
                return args
        raise RuntimeError(f"no valid cut_out args for duration {dur}")
    if tool == "concat":
        # 3-input concat is the unseen-combo variant; other splits mix
        # 2 and 3 inputs so the template pool has enough distinct cases.
        n = 3 if combo else (3 if rng.random() < 0.5 else 2)
        return {"inputs": [FILE] * n}
    if tool == "scale":
        w, h = _pick(rng, SCALE_COMBO if combo else SCALE_SEEN)
        return {"width": w, "height": h}
    if tool == "speed":
        return {"factor": _pick(rng, SPEED_COMBO if combo else SPEED_SEEN)}
    if tool == "convert":
        codec, container, crf = _pick(rng, CONVERT_COMBO if combo else CONVERT_SEEN)
        return {"codec": codec, "container": container, "crf": crf}
    if tool == "extract_audio":
        return {"format": _pick(rng, EXTRACT_COMBO if combo else EXTRACT_SEEN)}
    if tool == "mute":
        return {}
    raise ValueError(tool)


def _expected(tool: str, args: dict) -> dict:
    if tool == "concat":
        return {"tool": "concat", "inputs": list(args["inputs"])}
    if tool == "clarify":
        return {"tool": "clarify", "question": args["question"], "missing": list(args["missing"])}
    call = {"tool": tool, "input": FILE}
    call.update(args)
    return call


def _contrast_pair(fixture: str, split: str, n: int) -> list[dict]:
    """The cut-first-10s vs trim-to-10s trap, on a 30s+ fixture.

    Same wording family, different labels — the pair the dataset must
    teach, so the baseline must be measured on it.
    """
    return [
        {"id": f"{split}-{n:03d}", "split": split, "prompt": f"cut the first 10 seconds off {FILE}",
         "fixture": fixture, "template_id": "cut_t1",
         "expected": {"tool": "cut_out", "input": FILE, "start": 0.0, "end": 10.0},
         "notes": "contrast pair: cut first 10s (cut_out) vs trim to first 10s (trim)"},
        {"id": f"{split}-{n + 1:03d}", "split": split, "prompt": f"trim {FILE} to the first 10 seconds",
         "fixture": fixture, "template_id": "trim_t5",
         "expected": {"tool": "trim", "input": FILE, "start": 0.0, "end": 10.0},
         "notes": "contrast pair: trim to first 10s (trim) vs cut first 10s (cut_out)"},
    ]


def _silent_ok(tool: str) -> bool:
    # extract_audio / mute on a silent fixture are degenerate (no audio
    # to extract / already silent) — keep those tools off silent_30s.
    return tool not in ("extract_audio", "mute")


def generate_split(split: str, rng: random.Random) -> list[dict]:
    """Build one 50-case split. `split` is one of the four file stems."""
    heldout_phrasing = split == "eval_unseen_paraphrase"
    combo = split == "eval_unseen_combo"
    unseen_fixture = split == "eval_unseen_fixture"
    cases: list[dict] = []
    seen_keys: set[tuple] = set()
    n = 0

    def add(prompt: str, fixture: str, template_id: str, expected: dict, notes: str = "") -> bool:
        nonlocal n
        key = (prompt, json.dumps(expected, sort_keys=True))
        if key in seen_keys:
            return False
        seen_keys.add(key)
        n += 1
        cases.append({"id": f"{split}-{n:03d}", "split": split, "prompt": prompt,
                      "fixture": fixture, "template_id": template_id,
                      "expected": expected, "notes": notes})
        return True

    # Contrast pairs first, on a fixture long enough for both labels.
    pair_fixture = UNSEEN_FIXTURE if unseen_fixture else "base_30s"
    for c in _contrast_pair(pair_fixture, split, n + 1):
        cases.append(c)
        seen_keys.add((c["prompt"], json.dumps(c["expected"], sort_keys=True)))
        n += 1
    # The pair consumed one trim and one cut_out slot.
    remaining = dict(COUNTS)
    remaining["trim"] -= 1
    remaining["cut_out"] -= 1

    for tool, count in remaining.items():
        train_pool, heldout_pool = TEMPLATES[tool]
        pool = heldout_pool if heldout_phrasing else train_pool
        made = 0
        guard = 0
        while made < count and guard < 200:
            guard += 1
            fixture = UNSEEN_FIXTURE if unseen_fixture else _pick(rng, SEEN_FIXTURES)
            if fixture == "silent_30s" and not _silent_ok(tool):
                continue
            if tool == "clarify":
                tid, prompt, cargs = _pick(rng, pool)
                expected = _expected("clarify", cargs)
                if add(prompt, fixture, tid, expected, "ambiguous prompt -> clarify, do not guess"):
                    made += 1
                continue
            args = _args_for(rng, tool, fixture, combo)
            tid, render = _pick(rng, pool)
            try:
                prompt = render(args, fixture)
            except Exception:
                continue
            expected = _expected(tool, args)
            if add(prompt, fixture, tid, expected):
                made += 1
        if made < count:
            raise RuntimeError(f"{split}: only made {made}/{count} {tool} cases")
    assert len(cases) == 50, f"{split}: {len(cases)} cases, expected 50"
    return cases


def validate_cases(cases: list[dict]) -> None:
    """Every ground-truth call must validate against its fixture probe.

    No ffmpeg is run here; run_eval --backend oracle executes them all.
    """
    probes = {}
    for c in cases:
        fixture = c["fixture"]
        if fixture not in probes:
            probes[fixture] = probe_file(FIXTURES_DIR / fixture / "input.mp4")
        exp = c["expected"]
        if exp["tool"] == "clarify":
            errs = validate(exp, None, Path("."))
        elif exp["tool"] == "concat":
            errs = validate(exp, probes[fixture], Path("."))
        else:
            errs = validate(exp, probes[fixture], Path("."))
        if errs:
            raise RuntimeError(f"{c['id']} {c['prompt']!r} invalid: {errs}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval")
    args = ap.parse_args()
    out = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    all_cases: list[dict] = []
    counts: dict[str, dict[str, int]] = {}
    for split in ("eval_seen", "eval_unseen_paraphrase", "eval_unseen_combo", "eval_unseen_fixture"):
        cases = generate_split(split, rng)
        validate_cases(cases)
        all_cases.extend(cases)
        counts[split] = {t: sum(1 for c in cases if c["expected"]["tool"] == t) for t in COUNTS}
        path = out / f"{split}.jsonl"
        path.write_text("".join(json.dumps(c) + "\n" for c in cases))
        print(f"{split}: {len(cases)} cases -> {path}")
    freeze = {
        "seed": SEED,
        "generated": "2026-11-20",
        "splits": list(counts),
        "per_split_tool_counts": counts,
        "heldout_template_ids": sorted(
            tid for _, held in TEMPLATES.values() for tid, *_ in held),
        "train_template_ids": sorted(
            tid for train, _ in TEMPLATES.values() for tid, *_ in train),
        "reserved_fixture": UNSEEN_FIXTURE,
        "combo_arg_pools": {
            "trim": TRIM_COMBO, "cut_out": CUT_COMBO, "scale": SCALE_COMBO,
            "speed": SPEED_COMBO, "convert": CONVERT_COMBO, "extract_audio": EXTRACT_COMBO,
        },
        "rules": [
            "Held-out template ids must never be used in Phase 4 training data.",
            "The reserved fixture must never be used in training data.",
            "Combo arg pools must never be used in training data.",
            "Eval cases must never be sampled into the training pool.",
            "Do not edit these files after the baseline run; new runs get new files.",
        ],
    }
    (out / "freeze.json").write_text(json.dumps(freeze, indent=2) + "\n")
    print(f"freeze record -> {out / 'freeze.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
