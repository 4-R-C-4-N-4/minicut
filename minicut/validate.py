"""Validation: schema first, then semantic checks against the probe.

Returns a list of human-readable errors. Empty list = valid.
The grammar already constrains shape; this is the layer that can see
the actual input file (duration, resolution, streams).
"""
from __future__ import annotations

from pathlib import Path

from .probe import Probe

TOOLS = {
    "trim": {"start", "end"},
    "cut_out": {"start", "end"},
    "concat": {"inputs"},
    "scale": {"width", "height"},
    "speed": {"factor"},
    "convert": {"codec", "container", "crf"},
    "extract_audio": {"format"},
    "mute": set(),
    "clarify": {"question", "missing"},
}
CODECS = {"libx264", "libx265", "vp9"}
CONTAINERS = {"mp4", "mkv", "webm"}
AUDIO_FORMATS = {"mp3", "wav", "aac"}
# A trim this close to the full clip at both ends keeps everything.
NOOP_TRIM_TOL_S = 0.05


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def validate(tool_call: dict, probe: Probe | None, workdir: Path) -> list[str]:
    errors: list[str] = []
    if not isinstance(tool_call, dict):
        return ["tool call must be a JSON object"]
    tool = tool_call.get("tool")
    if not isinstance(tool, str) or tool not in TOOLS:
        return [f"unknown tool: {tool!r}"]
    if tool == "clarify":
        if not isinstance(tool_call.get("question"), str) or not tool_call["question"].strip():
            errors.append("clarify needs a non-empty question")
        return errors

    # Unknown / missing keys (input/output are common keys, not per-tool args).
    allowed = TOOLS[tool] | {"tool", "input", "output", "inputs"}
    for k in tool_call:
        if k not in allowed:
            errors.append(f"unknown key for {tool}: {k!r}")
    # Per-tool required args (crf has a default, so not required).
    required = TOOLS[tool] - ({"crf"} if tool == "convert" else set())
    if tool == "concat":
        if not isinstance(tool_call.get("inputs"), list) or len(tool_call.get("inputs", [])) < 2:
            errors.append("concat needs inputs: a list of at least 2 files")
    else:
        if "input" not in tool_call:
            errors.append(f"{tool} needs 'input'")
        for a in required:
            if a not in tool_call:
                errors.append(f"{tool} needs '{a}'")
    if errors:
        return errors

    files = tool_call["inputs"] if tool == "concat" else [tool_call["input"]]
    if "output" in tool_call:
        files = [*files, tool_call["output"]]
    for f in files:
        if not isinstance(f, str) or not f or "\0" in f:
            errors.append("paths must be non-empty strings")
        elif Path(f).is_absolute() or ".." in Path(f).parts:
            errors.append(f"path must be relative and inside the workdir: {f!r}")
    if errors:
        return errors
    if probe is None:
        return errors  # cannot do semantic checks without a probe

    if tool in ("trim", "cut_out"):
        s, e = tool_call["start"], tool_call["end"]
        if not _is_number(s) or not _is_number(e):
            errors.append("start/end must be numbers (seconds)")
        elif not (0 <= s < e <= probe.duration_s):
            errors.append(f"need 0 <= start < end <= duration ({probe.duration_s:.2f}s), got {s}..{e}")
        elif tool == "trim" and s <= NOOP_TRIM_TOL_S and e >= probe.duration_s - NOOP_TRIM_TOL_S:
            errors.append("this trim keeps the whole clip, so it would not change anything")
    if tool == "scale":
        w, h = tool_call["width"], tool_call["height"]
        if not isinstance(w, int) or not isinstance(h, int) or w <= 0 or h <= 0:
            errors.append("width/height must be positive integers")
        elif w % 2 or h % 2:
            errors.append("width/height must be even (yuv420p)")
        elif (w, h) == (probe.width, probe.height):
            errors.append(f"the clip is already {w}x{h}, so this would not change anything")
    if tool == "speed":
        f = tool_call["factor"]
        if not _is_number(f) or not (0.25 <= f <= 4.0):
            errors.append(f"speed factor must be 0.25..4.0, got {f!r}")
        elif f == 1:
            errors.append("speed 1x would not change anything")
    if tool == "convert":
        if not isinstance(tool_call["codec"], str) or tool_call["codec"] not in CODECS:
            errors.append(f"codec not allowlisted: {tool_call['codec']!r}")
        if not isinstance(tool_call["container"], str) or tool_call["container"] not in CONTAINERS:
            errors.append(f"container not allowlisted: {tool_call['container']!r}")
        crf = tool_call.get("crf", 23)
        if not isinstance(crf, int) or not (0 <= crf <= 51):
            errors.append(f"crf must be int 0..51, got {crf!r}")
    if tool == "extract_audio":
        if not isinstance(tool_call["format"], str) or tool_call["format"] not in AUDIO_FORMATS:
            errors.append(f"audio format not allowlisted: {tool_call['format']!r}")
    return errors
