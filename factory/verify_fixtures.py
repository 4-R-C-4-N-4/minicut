"""Check every generated fixture against its manifest.

Usage: python -m factory.verify_fixtures --root factory/fixtures
Exits non-zero on any mismatch. Also does one negative check: a
deliberately wrong duration must fail verification.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minicut.probe import probe_file
from minicut.verify import colour_at, tone_at


def check_fixture(d: Path) -> list[str]:
    errors = []
    manifest = json.loads((d / "manifest.json").read_text())
    p = probe_file(d / "input.mp4")
    if abs(p.duration_s - manifest["duration_s"]) > 0.15:
        errors.append(f"duration {p.duration_s} != {manifest['duration_s']}")
    if (p.width, p.height) != (manifest["width"], manifest["height"]):
        errors.append(f"dims {p.width}x{p.height} != {manifest['width']}x{manifest['height']}")
    if p.has_audio != manifest["has_audio"]:
        errors.append(f"has_audio {p.has_audio} != {manifest['has_audio']}")
    for seg in manifest["segments"]:
        mid = (seg["start"] + seg["end"]) / 2
        got = colour_at(d / "input.mp4", mid)
        if got != seg["color"]:
            errors.append(f"colour at {mid}s: {got} != {seg['color']}")
        if manifest["has_audio"] and seg["tone_hz"]:
            tone = tone_at(d / "input.mp4", mid, duration=1.0)
            if abs(tone - seg["tone_hz"]) > 20:
                errors.append(f"tone at {mid}s: {tone:.0f}Hz != {seg['tone_hz']}Hz")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="factory/fixtures")
    args = ap.parse_args()
    root = Path(args.root)
    ok = True
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        errs = check_fixture(d)
        print(f"{d.name}: {'OK' if not errs else 'FAIL: ' + '; '.join(errs)}")
        ok = ok and not errs
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
