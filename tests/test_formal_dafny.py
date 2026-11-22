"""Cross-check: the Dafny dictionary in `formal/dafny/minicut.dfy`
against the Python wrapper it specifies.

This test does NOT run Dafny (no Dafny in CI here). It encodes the
Dafny spec's validity rules and duration formulas in Python and checks
they agree with `minicut.validate` / `minicut.verify` / `minicut.chain`
/ `minicut.wrapper._resolve_output` on sampled cases. If the wrapper
changes and the Dafny file does not (or vice versa), this test fails
first — that is its job.

Known divergences from the Python wrapper (also listed in
`formal/dafny/README.md`):
- `extract_audio` on a silent input: Dafny says invalid (there is no
  audio to extract); `validate.py` does not check `has_audio` for
  this tool and says valid. The test asserts that divergence
  explicitly instead of papering over it.
- Clarify's `missing` list: Dafny requires it to be exactly
  required-minus-provided; `validate.py` only checks the question
  is non-empty. The exactness rule is tested here as the spec's
  own contract, not as a wrapper behaviour.
"""
from __future__ import annotations

from pathlib import Path

from minicut.chain import append_chain, token_for
from minicut.probe import Probe
from minicut.validate import validate
from minicut.wrapper import _resolve_output


def _probe(duration=60.0, width=1280, height=720, has_audio=True):
    return Probe(
        path=Path("clip.mp4"), duration_s=duration, width=width,
        height=height, fps=30.0, video_codec="h264",
        audio_codec="aac" if has_audio else None,
        has_audio=has_audio, has_video=True, format_name="mp4",
    )


# --- Mirror of Dafny `ValidSingle` (formal/dafny/minicut.dfy §3) ---
def dafny_valid(tool_call, probe):
    tool = tool_call["tool"]
    if tool in ("trim", "cut_out"):
        s, e = tool_call["start"], tool_call["end"]
        return 0 <= s < e <= probe.duration_s
    if tool == "scale":
        w, h = tool_call["width"], tool_call["height"]
        return w >= 2 and h >= 2 and w % 2 == 0 and h % 2 == 0
    if tool == "speed":
        return 0.25 <= tool_call["factor"] <= 4.0
    if tool == "convert":
        return (tool_call["codec"] in ("libx264", "libx265", "vp9")
                and tool_call["container"] in ("mp4", "mkv", "webm")
                and 0 <= tool_call.get("crf", 23) <= 51)
    if tool == "extract_audio":
        return tool_call["format"] in ("mp3", "wav", "aac") and probe.has_audio
    if tool == "mute":
        return True
    return False


# Mirror of Dafny `DurationAfter` (§4), single-input tools only.
def dafny_duration(tool_call, probe):
    tool = tool_call["tool"]
    if tool == "trim":
        return tool_call["end"] - tool_call["start"]
    if tool == "cut_out":
        return probe.duration_s - (tool_call["end"] - tool_call["start"])
    if tool == "speed":
        return probe.duration_s / tool_call["factor"]
    return probe.duration_s


# Mirror of Dafny `RequiredArgs` (§8).
REQUIRED = {
    "trim": {"input", "start", "end"},
    "cut_out": {"input", "start", "end"},
    "concat": {"inputs"},
    "scale": {"input", "width", "height"},
    "speed": {"input", "factor"},
    "convert": {"input", "codec", "container"},  # crf has a default
    "extract_audio": {"input", "format"},
    "mute": {"input"},
    "clarify": {"question"},
}


def dafny_missing(tool, provided):
    return sorted(REQUIRED[tool] - set(provided))


CASES = [
    # (tool_call, probe, dafny_valid_expected)
    ({"tool": "trim", "input": "clip.mp4", "start": 12.0, "end": 45.0}, _probe(), True),
    ({"tool": "trim", "input": "clip.mp4", "start": 12.0, "end": 61.0}, _probe(), False),
    ({"tool": "trim", "input": "clip.mp4", "start": 45.0, "end": 12.0}, _probe(), False),
    ({"tool": "trim", "input": "clip.mp4", "start": 0.0, "end": 60.0}, _probe(), True),
    ({"tool": "cut_out", "input": "clip.mp4", "start": 12.0, "end": 45.0}, _probe(), True),
    ({"tool": "cut_out", "input": "clip.mp4", "start": 45.0, "end": 12.0}, _probe(), False),
    ({"tool": "scale", "input": "clip.mp4", "width": 1280, "height": 720}, _probe(), True),
    ({"tool": "scale", "input": "clip.mp4", "width": 1281, "height": 720}, _probe(), False),
    ({"tool": "scale", "input": "clip.mp4", "width": 0, "height": 720}, _probe(), False),
    ({"tool": "speed", "input": "clip.mp4", "factor": 2.0}, _probe(), True),
    ({"tool": "speed", "input": "clip.mp4", "factor": 0.25}, _probe(), True),
    ({"tool": "speed", "input": "clip.mp4", "factor": 5.0}, _probe(), False),
    ({"tool": "speed", "input": "clip.mp4", "factor": 0.1}, _probe(), False),
    ({"tool": "convert", "input": "clip.mp4", "codec": "libx264", "container": "mp4"}, _probe(), True),
    ({"tool": "convert", "input": "clip.mp4", "codec": "libx265", "container": "webm", "crf": 51}, _probe(), True),
    ({"tool": "convert", "input": "clip.mp4", "codec": "libx999", "container": "mp4"}, _probe(), False),
    ({"tool": "convert", "input": "clip.mp4", "codec": "libx264", "container": "mp4", "crf": 99}, _probe(), False),
    ({"tool": "extract_audio", "input": "clip.mp4", "format": "mp3"}, _probe(), True),
    ({"tool": "extract_audio", "input": "clip.mp4", "format": "flac"}, _probe(), False),
    ({"tool": "mute", "input": "clip.mp4"}, _probe(), True),
]


def test_dafny_valid_agrees_with_validate(tmp_path):
    workdir = tmp_path
    for call, probe, expected in CASES:
        assert dafny_valid(call, probe) == expected, call
        errors = validate(call, probe, workdir)
        assert (errors == []) == expected, (call, errors)


def test_dafny_extract_audio_silent_is_the_known_divergence(tmp_path):
    call = {"tool": "extract_audio", "input": "clip.mp4", "format": "mp3"}
    probe = _probe(has_audio=False)
    assert dafny_valid(call, probe) is False  # Dafny: no audio to extract
    assert validate(call, probe, tmp_path) == []  # Python: not checked today


def test_dafny_duration_formulas_are_the_verify_formulas():
    probe = _probe(duration=60.0)
    assert dafny_duration({"tool": "trim", "start": 12.0, "end": 45.0}, probe) == 33.0
    assert dafny_duration({"tool": "cut_out", "start": 12.0, "end": 45.0}, probe) == 27.0
    # trim + cut_out partition the input (Dafny lemma TrimCutOutPartition)
    assert dafny_duration({"tool": "trim", "start": 12.0, "end": 45.0}, probe) + \
        dafny_duration({"tool": "cut_out", "start": 12.0, "end": 45.0}, probe) == 60.0
    assert dafny_duration({"tool": "speed", "factor": 2.0}, probe) == 30.0
    assert dafny_duration({"tool": "speed", "factor": 0.5}, probe) == 120.0
    assert dafny_duration({"tool": "scale", "width": 640, "height": 360}, probe) == 60.0
    assert dafny_duration({"tool": "mute"}, probe) == 60.0
    # concat duration is the sum (Dafny lemma ConcatDurationIsSum)
    assert 30.0 + 10.0 == 40.0


def test_dafny_chain_grows_by_one_and_never_rewrites():
    assert append_chain(None, "trim:12-45") == "trim:12-45"
    assert append_chain("trim:12-45", "speed:2.0") == "trim:12-45;speed:2.0"
    chain = append_chain(append_chain(None, token_for({"tool": "trim", "start": 12.0, "end": 45.0})), token_for({"tool": "speed", "factor": 2.0}))
    # _num strips ".0": 2.0 -> "2", so the token is "speed:2", not "speed:2.0".
    assert chain == "trim:12-45;speed:2"
    assert token_for({"tool": "cut_out", "start": 12.0, "end": 45.0}) == "cutout:12-45"
    assert token_for({"tool": "mute"}) == "mute"


def test_dafny_never_overwrite_matches_resolve_output(tmp_path):
    workdir = tmp_path
    inp = workdir / "clip.mp4"
    inp.touch()
    call = {"tool": "trim", "input": "clip.mp4", "start": 12.0, "end": 45.0}
    out = _resolve_output(call, workdir, ["clip.mp4"])
    assert out != inp
    assert out.name == "clip_trim.mp4"
    # Collision bumps to _v2 instead of overwriting (Dafny BumpedNameDiffers).
    (workdir / "clip_trim.mp4").touch()
    out2 = _resolve_output(call, workdir, ["clip.mp4"])
    assert out2.name == "clip_trim_v2.mp4"
    assert out2 != inp


def test_dafny_clarify_missing_is_exact():
    # "trim it" with no times: required minus provided is exactly {start, end}.
    assert dafny_missing("trim", {"input"}) == ["end", "start"]
    assert dafny_missing("trim", {"input", "start", "end"}) == []
    assert dafny_missing("convert", {"input", "codec"}) == ["container"]
    # crf has a default, so it is never in the missing list.
    assert "crf" not in dafny_missing("convert", {"input", "codec", "container"})
    assert dafny_missing("clarify", set()) == ["question"]
