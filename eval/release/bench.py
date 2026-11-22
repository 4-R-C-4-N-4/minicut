"""MiniCut independent release benchmark (CPU only, no model).

The roster (cases.jsonl) and fixture specs (fixtures.json) are frozen and
pinned below. Every case is graded on two independent axes:

  intent_pass        -- the published derivative has the effect the request
                        asked for, measured on the output media against the
                        expected edit (not against the call that produced it),
                        or, for an ambiguous request, a clarification with the
                        exact missing fields and nothing published.
  sources_unchanged  -- every staged source file is byte-for-byte identical
                        after the attempt, whatever the attempt did.

`run_case` accepts any candidate tool call, so a later model run can reuse
it; this module only runs calibration controls. See eval/release/README.md.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RELEASE_DIR = Path(__file__).resolve().parent
DEFAULT_FIXTURES = RELEASE_DIR / "fixtures"
FROZEN_FILES = ("cases.jsonl", "fixtures.json")

# Pinned when the roster was frozen. Never recompute these from the files
# being checked: a replaced roster would then pass its own gate.
PIN = {
    "cases.jsonl": "3629c7edbe0adeeed6fca162314cc5c12469d9ef2be7802035dc0a5bd8e4e312",
    "fixtures.json": "69c26cb477c77e113f6721a7c0196c30ccf32fe6c37e0fec4efa0f83456df32b",
    "cases": 38,
    "categories": {"keep_remove": 16, "ambiguous": 8, "concat_order": 6, "other_tools": 8},
}

PALETTE = {"yellow": (255, 255, 0), "cyan": (0, 255, 255), "magenta": (255, 0, 255),
           "black": (0, 0, 0), "white": (255, 255, 255), "red": (255, 0, 0),
           "lime": (0, 255, 0), "blue": (0, 0, 255)}
AUDIO_CODECS = {"mp3": "mp3", "wav": "pcm_s16le", "aac": "aac"}
VIDEO_CODECS = {"libx264": "h264", "libx265": "hevc", "vp9": "vp9"}
VIDEO_TOOLS = {"trim", "cut_out", "concat", "scale", "speed", "convert", "mute"}
DURATION_TOL = 0.25


class RosterError(ValueError):
    """The frozen roster is missing, changed, or malformed."""


@dataclass
class Roster:
    cases: list[dict]
    fixtures: dict[str, dict]


def sha256_file(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def load_roster(release_dir: Path, pin: dict = PIN) -> Roster:
    """Load the roster only if it matches the pin and the schema."""
    release_dir = Path(release_dir)
    for name in FROZEN_FILES:
        path = release_dir / name
        if not path.is_file():
            raise RosterError(f"missing frozen file: {name}")
        if sha256_file(path) != pin[name]:
            raise RosterError(f"{name}: digest does not match the pinned roster")
    fixtures = json.loads((release_dir / "fixtures.json").read_text())
    cases = [json.loads(line) for line in (release_dir / "cases.jsonl").read_text().splitlines()
             if line.strip()]
    ids = [c.get("id") for c in cases]
    if len(cases) != pin["cases"] or len(set(ids)) != len(ids):
        raise RosterError(f"expected {pin['cases']} uniquely identified cases, got {len(cases)}")
    counts: dict[str, int] = {}
    for case in cases:
        counts[case.get("category")] = counts.get(case.get("category"), 0) + 1
        _check_case(case, fixtures)
    if counts != pin["categories"]:
        raise RosterError(f"category counts {counts} != pinned {pin['categories']}")
    return Roster(cases, fixtures)


def _check_case(case: dict, fixtures: dict) -> None:
    from minicut.validate import validate

    cid = case.get("id")
    files = case.get("files")
    if not isinstance(files, dict) or not files or any(f not in fixtures for f in files.values()):
        raise RosterError(f"{cid}: files must map names to frozen fixtures")
    expected, wrong = case.get("expected"), case.get("control_wrong")
    if not isinstance(expected, dict):
        raise RosterError(f"{cid}: missing expected outcome")
    calls = [wrong]
    if expected.get("kind") == "edit":
        calls.append(expected.get("call"))
    elif expected.get("kind") != "clarify" or not expected.get("missing"):
        raise RosterError(f"{cid}: expected must be an edit or a clarification")
    for call in calls:
        if not isinstance(call, dict) or call.get("tool") == "clarify" or validate(call, None, Path(".")):
            raise RosterError(f"{cid}: controls and edits must be valid edit calls")
        names = call["inputs"] if call["tool"] == "concat" else [call["input"]]
        if any(n not in files for n in names):
            raise RosterError(f"{cid}: call names a file that is not staged")
    if expected.get("kind") == "edit" and wrong == expected["call"]:
        raise RosterError(f"{cid}: wrong control equals the expected edit")


# --- Separation from development and training data -----------------------
def _template(prompt: str) -> str:
    text = re.sub(r"\S+\.(?:mp4|mkv|webm|mp3|wav|m4a)\b", "<file>", prompt.lower())
    text = re.sub(r"\d+(?:[.:]\d+)?", "<n>", text)
    return " ".join(text.split())


def separation_problems(roster: Roster) -> list[str]:
    """Release prompts, filenames and fixtures must not appear in dev or training data."""
    from factory.generate import FIXTURES as DEV_FIXTURES
    from training.generate_dataset import FILENAMES as TRAIN_FILENAMES

    seen: set[str] = set()
    for path in sorted((ROOT / "eval").glob("eval_*.jsonl")):
        seen.update(_template(json.loads(line)["prompt"]) for line in path.read_text().splitlines())
    for name in ("train_v1.jsonl", "train_pilot_v1.jsonl"):
        path = ROOT / "training" / name
        if path.exists():
            seen.update(_template(json.loads(line)["messages"][1]["content"])
                        for line in path.read_text().splitlines())
    problems = []
    for case in roster.cases:
        if _template(case["prompt"]) in seen:
            problems.append(f"{case['id']}: prompt template appears in dev or training data")
        reused = set(case["files"]) & set(TRAIN_FILENAMES)
        if reused:
            problems.append(f"{case['id']}: training filename reused: {sorted(reused)}")
    dev_specs = [[c for c, _ in spec["segments"]] for spec in DEV_FIXTURES.values()]
    for fid, spec in roster.fixtures.items():
        if fid in DEV_FIXTURES or [c for c, _ in spec["segments"]] in dev_specs:
            problems.append(f"{fid}: fixture duplicates a development fixture")
    return problems


# --- Fixtures --------------------------------------------------------------
def _ffmpeg_bytes(argv: list[str]) -> bytes:
    r = subprocess.run(["ffmpeg", "-v", "error", *argv], capture_output=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode(errors="replace").strip()[-300:])
    return r.stdout


def colour_at(path: Path, t: float) -> str | None:
    """Mean frame colour at t, named from PALETTE, or None if nothing is close."""
    raw = _ffmpeg_bytes(["-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1",
                         "-vf", "scale=1:1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    if len(raw) < 3:
        return None
    rgb = raw[:3]
    name, dist = min(((n, sum((a - b) ** 2 for a, b in zip(rgb, c))) for n, c in PALETTE.items()),
                     key=lambda item: item[1])
    return name if dist < 3 * 60 ** 2 else None


def tone_at(path: Path, t: float) -> float:
    raw = _ffmpeg_bytes(["-ss", f"{t:.3f}", "-t", "1", "-i", str(path), "-vn", "-ac", "1",
                         "-ar", "48000", "-f", "s16le", "-"])
    samples = np.frombuffer(raw, dtype=np.int16).astype(float)
    if len(samples) < 4000:
        raise RuntimeError("audio slice too short")
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
    spectrum[:5] = 0
    return float(np.fft.rfftfreq(len(samples), 1 / 48000)[int(np.argmax(spectrum))])


def _probe(path: Path) -> dict:
    r = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format",
                        "-show_streams", str(path)], capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        raise RuntimeError(f"unprobeable: {path.name}")
    data = json.loads(r.stdout)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    return {"duration": float(data.get("format", {}).get("duration", 0) or 0),
            "format": data.get("format", {}).get("format_name", ""),
            "video": video and video.get("codec_name"), "audio": audio and audio.get("codec_name"),
            "size": (video["width"], video["height"]) if video else None}


def _timeline(spec: dict) -> list[tuple[float, float, str]]:
    n = len(spec["segments"])
    base, extra = divmod(spec["duration"], n)
    out, start = [], 0
    for i, (colour, _tone) in enumerate(spec["segments"]):
        end = start + base + (1 if i < extra else 0)
        out.append((start, end, colour))
        start = end
    return out


def ensure_fixtures(roster: Roster, root: Path = DEFAULT_FIXTURES) -> Path:
    """Generate any missing release fixture, then check every one against its spec."""
    from factory.generate import generate_fixture

    root = Path(root)
    for fid, spec in roster.fixtures.items():
        clip = root / fid / "input.mp4"
        manifest = root / fid / "manifest.json"
        if not clip.is_file() or not manifest.is_file() or \
                json.loads(manifest.read_text()).get("release_spec") != spec:
            generate_fixture(fid, dict(spec, segments=[tuple(s) for s in spec["segments"]]), root)
            data = json.loads(manifest.read_text())
            manifest.write_text(json.dumps(dict(data, release_spec=spec), indent=2))
        info = _probe(clip)
        problems = []
        if abs(info["duration"] - spec["duration"]) > DURATION_TOL:
            problems.append(f"duration {info['duration']:.2f}")
        if info["size"] != (spec["width"], spec["height"]) or bool(info["audio"]) != spec["audio"]:
            problems.append(f"streams {info['size']} audio={info['audio']}")
        for start, end, colour in _timeline(spec):
            if colour_at(clip, (start + end) / 2) != colour:
                problems.append(f"colour at {(start + end) / 2}s")
        if problems:
            raise RuntimeError(f"fixture {fid} does not match its spec: {'; '.join(problems)}")
    return root


# --- Expected effects -----------------------------------------------------
def expected_timeline(call: dict, timelines: dict, primary: str) -> list[tuple[float, float, str]]:
    """Output-time pieces (start, end, colour) the expected edit must produce."""
    tool = call["tool"]
    if tool == "concat":
        pieces, offset = [], 0.0
        for name in call["inputs"]:
            for s, e, c in timelines[name]:
                pieces.append((offset + s, offset + e, c))
            offset += timelines[name][-1][1]
        return pieces
    source = timelines[primary]
    if tool == "trim":
        lo, hi = float(call["start"]), float(call["end"])
        return [(max(s, lo) - lo, min(e, hi) - lo, c) for s, e, c in source if min(e, hi) > max(s, lo)]
    if tool == "cut_out":
        lo, hi = float(call["start"]), float(call["end"])
        before = [(s, min(e, lo), c) for s, e, c in source if s < lo]
        after = [(max(s, hi) - (hi - lo), e - (hi - lo), c) for s, e, c in source if e > hi]
        return before + after
    if tool == "speed":
        f = float(call["factor"])
        return [(s / f, e / f, c) for s, e, c in source]
    return list(source)


def _effect_problems(expected: dict, output: Path, roster: Roster, files: dict) -> list[str]:
    specs = {name: roster.fixtures[fid] for name, fid in files.items()}
    timelines = {name: _timeline(spec) for name, spec in specs.items()}
    tool = expected["tool"]
    primary = expected["inputs"][0] if tool == "concat" else expected["input"]
    pieces = expected_timeline(expected, timelines, primary)
    info = _probe(output)
    total = pieces[-1][1] if pieces else 0.0
    tol = 0.35 if tool == "concat" else DURATION_TOL
    problems = []
    if abs(info["duration"] - total) > tol:
        problems.append(f"duration {info['duration']:.2f}s, expected {total:.2f}s")
    if tool == "extract_audio":
        if info["video"] or info["audio"] != AUDIO_CODECS[expected["format"]]:
            problems.append(f"expected {expected['format']} audio only, got "
                            f"video={info['video']} audio={info['audio']}")
            return problems
        tones = {c: t for c, t in specs[primary]["segments"]}
        for s, e, c in pieces:
            if e - s >= 1.5 and abs(tone_at(output, s + 0.25) - tones[c]) > 25:
                problems.append(f"tone at {s + 0.25:.2f}s is not {tones[c]}Hz")
        return problems
    if not info["video"]:
        return problems + ["no video stream"]
    spec = specs[primary]
    size = (expected["width"], expected["height"]) if tool == "scale" else (spec["width"], spec["height"])
    if info["size"] != size:
        problems.append(f"size {info['size']}, expected {size}")
    has_audio = tool != "mute" and all(specs[n]["audio"] for n in
                                       (expected["inputs"] if tool == "concat" else [primary]))
    if bool(info["audio"]) != has_audio:
        problems.append(f"audio present={bool(info['audio'])}, expected {has_audio}")
    if tool == "convert":
        if info["video"] != VIDEO_CODECS[expected["codec"]] or output.suffix != f".{expected['container']}":
            problems.append(f"codec {info['video']} in {output.suffix}")
    for s, e, c in pieces:
        if e - s < 0.8:
            continue
        for t in (s + (e - s) * 0.25, s + (e - s) * 0.75):
            got = colour_at(output, t)
            if got != c:
                problems.append(f"colour at {t:.2f}s is {got}, expected {c}")
    return problems


# --- Running one attempt ---------------------------------------------------
def _snapshot(workdir: Path) -> dict[str, str]:
    return {str(p.relative_to(workdir)): sha256_file(p) if p.is_file() else "<dir>"
            for p in sorted(workdir.rglob("*"))}


def run_case(case: dict, call, fixtures_root: Path, editor=None, roster: Roster | None = None) -> dict:
    """Stage the case's sources, apply one candidate call, and grade both axes."""
    if editor is None:
        from minicut.wrapper import edit as editor
    roster = roster or load_roster(RELEASE_DIR)
    expected = case["expected"]
    with tempfile.TemporaryDirectory(prefix="minicut-release-") as tmp:
        workdir = Path(tmp)
        for name, fid in case["files"].items():
            shutil.copyfile(Path(fixtures_root) / fid / "input.mp4", workdir / name)
        before = _snapshot(workdir)
        try:
            result = editor(call, workdir)
        except Exception as exc:  # a crashing editor is a failed attempt, not a crashed benchmark
            result = {"ok": False, "error": f"editor raised {type(exc).__name__}: {exc}"}
        after = _snapshot(workdir)
        sources_unchanged = all(after.get(name) == before[name] for name in case["files"])
        published = sorted(set(after) - set(before))
        reason = ""
        if expected["kind"] == "clarify":
            missing = call.get("missing") if isinstance(call, dict) else None
            intent = (isinstance(call, dict) and call.get("tool") == "clarify" and bool(result.get("ok"))
                      and isinstance(missing, list) and len(missing) == len(set(missing))
                      and set(missing) == set(expected["missing"]) and not published)
            if not intent:
                reason = "expected a clarification of exactly the missing fields with nothing published"
        elif not result.get("ok") or len(published) != 1:
            intent = False
            reason = f"no single derivative published: {result.get('error', published)}"
        else:
            output = workdir / published[0]
            if Path(result.get("output", "")).resolve() != output.resolve():
                intent, reason = False, "reported output is not the published file"
            else:
                problems = _effect_problems(expected["call"], output, roster, case["files"])
                intent, reason = not problems, "; ".join(problems)
        return {"id": case["id"], "category": case["category"], "intent_pass": intent,
                "sources_unchanged": sources_unchanged, "published": published, "reason": reason}
