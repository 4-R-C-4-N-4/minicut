"""Verification: ffprobe + frame/audio checks against expected values.

Exit code 0 is NOT success. Every check here is programmatic.
Tolerances are part of the spec — do not tighten them silently.
"""
from __future__ import annotations

import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from .probe import Probe, probe_file

# Segment colours used by the factory (ffmpeg color names).
SEGMENT_COLORS = {
    "red": (255, 0, 0),
    "lime": (0, 255, 0),
    "blue": (0, 0, 255),
}
DURATION_TOL = 0.15


def _extract_frame(video: Path, t: float, out: Path) -> None:
    cmd = ["ffmpeg", "-y", "-v", "error", "-ss", str(t), "-i", str(video),
           "-frames:v", "1", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        raise RuntimeError(f"frame extract failed: {r.stderr.strip()}")


def colour_at(video: str | Path, t: float) -> str:
    """Centre-pixel colour, nearest of red/lime/blue."""
    with tempfile.TemporaryDirectory() as td:
        png = Path(td) / "frame.png"
        _extract_frame(Path(video), t, png)
        px = Image.open(png).convert("RGB")
        w, h = px.size
        r, g, b = px.getpixel((w // 2, h // 2))
    best, best_d = None, 1e18
    for name, (cr, cg, cb) in SEGMENT_COLORS.items():
        d = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2
        if d < best_d:
            best, best_d = name, d
    return best  # type: ignore[return-value]


def tone_at(audio_or_video: str | Path, t: float, duration: float = 1.0) -> float:
    """Dominant frequency (Hz) of a 1s audio slice starting at t."""
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "slice.wav"
        cmd = ["ffmpeg", "-y", "-v", "error", "-ss", str(t), "-t", str(duration),
               "-i", str(audio_or_video), "-vn", "-ac", "1", "-ar", "48000", str(wav)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            raise RuntimeError(f"audio extract failed: {r.stderr.strip()}")
        import wave
        with wave.open(str(wav)) as wf:
            n = wf.getnframes()
            raw = wf.readframes(n)
        samples = np.frombuffer(raw, dtype=np.int16).astype(float)
        if len(samples) < 1000:
            raise RuntimeError("audio slice too short for tone check")
        # Window + FFT, ignore DC.
        spec = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
        freqs = np.fft.rfftfreq(len(samples), 1 / 48000)
        spec[:5] = 0
        return float(freqs[int(np.argmax(spec))])


def mean_luma(video: str | Path, t: float) -> float:
    with tempfile.TemporaryDirectory() as td:
        png = Path(td) / "frame.png"
        _extract_frame(Path(video), t, png)
        img = Image.open(png).convert("L")
        return float(np.asarray(img).mean())


def _close(a: float, b: float, tol: float = DURATION_TOL) -> bool:
    return abs(a - b) <= tol


def verify_output(tool_call: dict, output: str | Path, input_probe: Probe,
                  input_probes: dict | None = None) -> tuple[bool, str]:
    """Check an edit output against what the tool call should have produced.

    Returns (passed, reason). `input_probes` maps input filename -> Probe,
    used by concat.
    """
    out = Path(output)
    if not out.exists():
        return False, f"output missing: {out}"
    try:
        p = probe_file(out)
    except Exception as e:
        return False, f"output unprobeable: {e}"

    tool = tool_call.get("tool")
    if tool == "trim":
        exp = float(tool_call["end"]) - float(tool_call["start"])
        if not _close(p.duration_s, exp):
            return False, f"trim duration {p.duration_s:.2f} != {exp:.2f}"
        return True, f"trim ok: duration {p.duration_s:.2f}s"
    if tool == "cut_out":
        exp = input_probe.duration_s - (float(tool_call["end"]) - float(tool_call["start"]))
        if not _close(p.duration_s, exp):
            return False, f"cut_out duration {p.duration_s:.2f} != {exp:.2f}"
        return True, f"cut_out ok: duration {p.duration_s:.2f}s"
    if tool == "concat":
        total = sum((input_probes or {}).get(f, input_probe).duration_s
                    for f in tool_call.get("inputs", []))
        # Fallback: if probes not supplied, use input_probe duration * count.
        if not input_probes:
            total = input_probe.duration_s * len(tool_call.get("inputs", []))
        if not _close(p.duration_s, total, tol=0.3):
            return False, f"concat duration {p.duration_s:.2f} != {total:.2f}"
        return True, f"concat ok: duration {p.duration_s:.2f}s"
    if tool == "scale":
        if p.width != int(tool_call["width"]) or p.height != int(tool_call["height"]):
            return False, f"scale dims {p.width}x{p.height} != {tool_call['width']}x{tool_call['height']}"
        return True, f"scale ok: {p.width}x{p.height}"
    if tool == "speed":
        exp = input_probe.duration_s / float(tool_call["factor"])
        if not _close(p.duration_s, exp, tol=0.3):
            return False, f"speed duration {p.duration_s:.2f} != {exp:.2f}"
        return True, f"speed ok: duration {p.duration_s:.2f}s"
    if tool == "convert":
        codec = {"libx264": "h264", "libx265": "hevc", "vp9": "vp9"}[tool_call["codec"]]
        if p.video_codec != codec:
            return False, f"convert codec {p.video_codec} != {codec}"
        return True, f"convert ok: codec {p.video_codec}"
    if tool == "extract_audio":
        if not p.has_audio:
            return False, "extract_audio: no audio stream in output"
        if not _close(p.duration_s, input_probe.duration_s, tol=0.3):
            return False, f"extract_audio duration {p.duration_s:.2f} != {input_probe.duration_s:.2f}"
        return True, f"extract_audio ok: duration {p.duration_s:.2f}s"
    if tool == "mute":
        if p.has_audio:
            return False, "mute: audio stream still present"
        if not _close(p.duration_s, input_probe.duration_s):
            return False, f"mute duration {p.duration_s:.2f} != {input_probe.duration_s:.2f}"
        return True, "mute ok: no audio stream"
    return False, f"unknown tool for verification: {tool}"
