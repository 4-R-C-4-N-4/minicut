"""MiniCut test factory — generate verifiable videos + manifests.

v1 note: fixtures are solid-colour segments (red / lime / blue) with a
sine tone per segment, not testsrc2. Colour-at-time + tone-at-time are
unambiguous with a centre-pixel / FFT check, which is what the verifier
needs. testsrc2 timecode is a v2 refinement, not a v1 requirement.
Each segment is generated with ffmpeg lavfi, then joined with the
concat demuxer (same codec settings throughout, so -c copy is safe).

Usage:
    python -m factory.generate --out factory/fixtures --all
    python -m factory.generate --out factory/fixtures --fixture base_30s
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

FIXTURES = {
    "base_30s": dict(duration=30, width=1280, height=720, fps=30, audio=True,
                     segments=[("red", 440), ("lime", 880), ("blue", 440)]),
    "short_10s": dict(duration=10, width=1280, height=720, fps=30, audio=True,
                      segments=[("red", 440), ("lime", 880)]),
    "long_60s_1080p": dict(duration=60, width=1920, height=1080, fps=30, audio=True,
                           segments=[("red", 440), ("lime", 880), ("blue", 440),
                                     ("red", 880), ("lime", 440), ("blue", 880)]),
    "silent_30s": dict(duration=30, width=1280, height=720, fps=30, audio=False,
                       segments=[("red", 440), ("lime", 880), ("blue", 440)]),
    "portrait_30s": dict(duration=30, width=720, height=1280, fps=30, audio=True,
                         segments=[("red", 440), ("lime", 880), ("blue", 440)]),
}


def _run(cmd: list[str]) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)}\n{r.stderr.strip()[-500:]}")


def _segment_durations(total: int, n: int) -> list[int]:
    base, rem = divmod(total, n)
    return [base + (1 if i < rem else 0) for i in range(n)]


def generate_fixture(fixture_id: str, spec: dict, out_root: Path) -> Path:
    out_dir = out_root / fixture_id
    out_dir.mkdir(parents=True, exist_ok=True)
    durations = _segment_durations(spec["duration"], len(spec["segments"]))
    segments_meta = []
    start = 0
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        seg_files = []
        for i, ((color, tone), dur) in enumerate(zip(spec["segments"], durations)):
            seg = td / f"seg{i}.mp4"
            cmd = ["ffmpeg", "-y", "-v", "error",
                   "-f", "lavfi", "-i", f"color=c={color}:s={spec['width']}x{spec['height']}:r={spec['fps']}:d={dur}"]
            if spec["audio"]:
                cmd += ["-f", "lavfi", "-i", f"sine=frequency={tone}:sample_rate=48000:duration={dur}",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest"]
            else:
                cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-an"]
            cmd.append(str(seg))
            _run(cmd)
            seg_files.append(seg)
            segments_meta.append({"start": start, "end": start + dur,
                                  "color": color, "tone_hz": tone if spec["audio"] else None})
            start += dur
        # Concat demuxer.
        lst = td / "list.txt"
        lst.write_text("".join(f"file '{f}'\n" for f in seg_files))
        out = out_dir / "input.mp4"
        _run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
              "-i", str(lst), "-c", "copy", str(out)])
    manifest = {
        "fixture_id": fixture_id,
        "duration_s": float(spec["duration"]),
        "width": spec["width"], "height": spec["height"], "fps": spec["fps"],
        "video_codec": "h264", "audio_codec": "aac" if spec["audio"] else None,
        "has_audio": spec["audio"],
        "segments": segments_meta,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="factory/fixtures")
    ap.add_argument("--fixture", choices=list(FIXTURES))
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    out_root = Path(args.out)
    names = list(FIXTURES) if args.all or not args.fixture else [args.fixture]
    for name in names:
        p = generate_fixture(name, FIXTURES[name], out_root)
        print(f"{name}: {p} ({FIXTURES[name]['duration']}s "
              f"{FIXTURES[name]['width']}x{FIXTURES[name]['height']})")


if __name__ == "__main__":
    main()
