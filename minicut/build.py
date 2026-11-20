"""Build ffmpeg argv lists — one deterministic template per tool.

No shell, no string concatenation of user text into a command.
Re-encode (not -c copy) for trim/cut_out/speed so durations are exact
to the frame, not keyframe-bound. See IMPLEMENTATION_GUIDE.md §5.3.
"""
from __future__ import annotations

from pathlib import Path

from .chain import CHAIN_KEY, append_chain, token_for
from .probe import Probe


def _atempo_chain(factor: float) -> str:
    """Decompose a speed factor into atempo steps (each 0.5..100)."""
    if factor == 1.0:
        return "atempo=1.0"
    steps: list[float] = []
    remaining = factor
    # atempo max is 100, min 0.5; for 0.25..4.0 one or two steps suffice.
    while remaining > 2.0:
        steps.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        steps.append(0.5)
        remaining /= 0.5
    steps.append(round(remaining, 4))
    return ",".join(f"atempo={s}" for s in steps)


def default_output(tool_call: dict) -> str:
    tool = tool_call["tool"]
    if tool == "concat":
        return "concat_output.mp4"
    if tool == "convert":
        return f"{Path(tool_call['input']).stem}_converted.{tool_call['container']}"
    if tool == "extract_audio":
        # AAC in a raw .aac (ADTS) file has no container duration
        # metadata — ffprobe under-reports it by ~0.35s, which breaks
        # verification for no real reason. .m4a is the same AAC codec
        # in an MP4 container, with an accurate duration.
        ext = {"aac": "m4a"}.get(tool_call["format"], tool_call["format"])
        return f"{Path(tool_call['input']).stem}.{ext}"
    return f"{Path(tool_call['input']).stem}_{tool}.{Path(tool_call['input']).suffix.lstrip('.') or 'mp4'}"


def build(tool_call: dict, probe: Probe, workdir: Path) -> list[str]:
    """Build argv, then stamp the edit-history chain into the output's
    metadata. The output path is always the last element, and -metadata
    is an output option, so it goes just before it. If the input had no
    chain tag, the chain starts with this op's token."""
    argv = _build(tool_call, probe, workdir)
    chain = append_chain(probe.chain if probe is not None else None,
                         token_for(tool_call))
    # MP4/M4A's muxer drops arbitrary metadata keys unless told to keep
    # them (verified 2026-11-20: a custom tag written without the flag
    # simply vanishes from the output). MKV/WebM keep custom keys
    # natively, so the flag is only added for .mp4/.m4a outputs.
    out_name = tool_call.get("output") or default_output(tool_call)
    extra: list[str] = []
    if Path(out_name).suffix.lower() in (".mp4", ".m4a"):
        extra = ["-movflags", "use_metadata_tags"]
    return argv[:-1] + ["-metadata", f"{CHAIN_KEY}={chain}"] + extra + [argv[-1]]


def _build(tool_call: dict, probe: Probe, workdir: Path) -> list[str]:
    tool = tool_call["tool"]
    inp = str(workdir / tool_call["input"]) if "input" in tool_call else None
    out = str(workdir / tool_call.get("output", default_output(tool_call)))
    if tool == "trim":
        return ["ffmpeg", "-y", "-ss", str(tool_call["start"]), "-to", str(tool_call["end"]),
                "-i", inp, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "aac", out]
    if tool == "cut_out":
        s, e = float(tool_call["start"]), float(tool_call["end"])
        dur = probe.duration_s
        if probe.has_audio:
            fc = (f"[0:v]trim=0:{s},setpts=PTS-STARTPTS[v0];"
                  f"[0:a]atrim=0:{s},asetpts=PTS-STARTPTS[a0];"
                  f"[0:v]trim={e}:{dur},setpts=PTS-STARTPTS[v1];"
                  f"[0:a]atrim={e}:{dur},asetpts=PTS-STARTPTS[a1];"
                  f"[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]")
            return ["ffmpeg", "-y", "-i", inp, "-filter_complex", fc,
                    "-map", "[v]", "-map", "[a]", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", out]
        fc = (f"[0:v]trim=0:{s},setpts=PTS-STARTPTS[v0];"
              f"[0:v]trim={e}:{dur},setpts=PTS-STARTPTS[v1];"
              f"[v0][v1]concat=n=2:v=1:a=0[v]")
        return ["ffmpeg", "-y", "-i", inp, "-filter_complex", fc,
                "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p", out]
    if tool == "concat":
        # Concat demuxer list file, written next to the output by the wrapper.
        # build() returns the argv; the list file path is passed via _list_file.
        list_file = tool_call["_list_file"]
        return ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file,
                "-c", "copy", out]
    if tool == "scale":
        return ["ffmpeg", "-y", "-i", inp,
                "-vf", f"scale={tool_call['width']}:{tool_call['height']}",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "copy", out]
    if tool == "speed":
        f = float(tool_call["factor"])
        vf = f"setpts=PTS/{f}"
        if probe.has_audio:
            af = _atempo_chain(f)
            return ["ffmpeg", "-y", "-i", inp, "-vf", vf, "-af", af,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", out]
        return ["ffmpeg", "-y", "-i", inp, "-vf", vf,
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", out]
    if tool == "convert":
        crf = tool_call.get("crf", 23)
        argv = ["ffmpeg", "-y", "-i", inp, "-c:v", tool_call["codec"],
                "-crf", str(crf), "-pix_fmt", "yuv420p"]
        # WebM only allows Vorbis/Opus audio — AAC (the default) fails
        # the mux outright. mkv/mp4 keep AAC.
        audio_codec = "libopus" if tool_call["container"] == "webm" else "aac"
        argv += ["-c:a", audio_codec] if probe.has_audio else ["-an"]
        argv.append(out)
        return argv
    if tool == "extract_audio":
        fmt = tool_call["format"]
        codec = {"mp3": "libmp3lame", "wav": "pcm_s16le", "aac": "aac"}[fmt]
        return ["ffmpeg", "-y", "-i", inp, "-vn", "-c:a", codec, out]
    if tool == "mute":
        return ["ffmpeg", "-y", "-i", inp, "-c:v", "copy", "-an", out]
    raise ValueError(f"no template for tool: {tool}")
