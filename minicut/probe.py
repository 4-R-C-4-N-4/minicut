"""ffprobe wrapper. The only module that calls ffprobe directly."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


from .chain import CHAIN_KEY


@dataclass
class Probe:
    path: Path
    duration_s: float
    width: int | None
    height: int | None
    fps: float | None
    video_codec: str | None
    audio_codec: str | None
    has_audio: bool
    has_video: bool
    format_name: str | None
    # Edit-history chain from the minicut_chain metadata tag, if present.
    # None on a fresh file or one whose metadata was stripped elsewhere.
    chain: str | None = None


def probe_file(path: str | Path) -> Probe:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"input not found: {path}")
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if out.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {out.stderr.strip()}")
    data = json.loads(out.stdout)
    fmt = data.get("format", {})
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    audio = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
    fps = None
    if video and video.get("avg_frame_rate"):
        try:
            num, den = video["avg_frame_rate"].split("/")
            fps = float(num) / float(den) if float(den) else None
        except Exception:
            fps = None
    # Metadata tags are case-insensitive in practice (MP4 stores them
    # lowercased, MKV keeps the case written); match case-insensitively.
    tags = fmt.get("tags", {}) or {}
    chain = next((v for k, v in tags.items() if k.lower() == CHAIN_KEY), None)
    return Probe(
        path=path,
        duration_s=float(fmt.get("duration", 0) or 0),
        width=int(video["width"]) if video else None,
        height=int(video["height"]) if video else None,
        fps=fps,
        video_codec=video.get("codec_name") if video else None,
        audio_codec=audio.get("codec_name") if audio else None,
        has_audio=audio is not None,
        has_video=video is not None,
        format_name=fmt.get("format_name"),
        chain=chain,
    )
