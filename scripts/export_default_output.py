"""Export golden cases for minicut.build.default_output, for the Rust port.

    PYTHONPATH=<repo root> <venv>/bin/python scripts/export_default_output.py
"""
from __future__ import annotations

import json
from pathlib import Path

from minicut.build import default_output

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "crates/minicut/tests/golden/default_output.json"

INPUTS = [
    "clip.mp4",
    "video.mp4",
    "my_video.mp4",
    "a.b.mp4",
    "noext",
    ".hidden",
    "clip_01.mp4",
    "take1.mkv",
    "movie.webm",
]
CONTAINERS = ["mp4", "mkv", "webm"]
FORMATS = ["mp3", "wav", "aac"]
TOOLS = [
    "trim",
    "cut_out",
    "concat",
    "scale",
    "speed",
    "convert",
    "extract_audio",
    "mute",
]


def main() -> None:
    rows: list[dict] = []
    for tool in TOOLS:
        if tool == "concat":
            call = {"tool": "concat"}
            rows.append(
                {
                    "tool": tool,
                    "input": None,
                    "container": None,
                    "format": None,
                    "expected": default_output(call),
                }
            )
            continue
        for inp in INPUTS:
            if tool == "convert":
                for container in CONTAINERS:
                    call = {"tool": tool, "input": inp, "container": container}
                    rows.append(
                        {
                            "tool": tool,
                            "input": inp,
                            "container": container,
                            "format": None,
                            "expected": default_output(call),
                        }
                    )
            elif tool == "extract_audio":
                for fmt in FORMATS:
                    call = {"tool": tool, "input": inp, "format": fmt}
                    rows.append(
                        {
                            "tool": tool,
                            "input": inp,
                            "container": None,
                            "format": fmt,
                            "expected": default_output(call),
                        }
                    )
            else:
                call = {"tool": tool, "input": inp}
                rows.append(
                    {
                        "tool": tool,
                        "input": inp,
                        "container": None,
                        "format": None,
                        "expected": default_output(call),
                    }
                )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
