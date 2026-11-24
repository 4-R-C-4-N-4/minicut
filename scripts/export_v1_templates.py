"""Export v1 template renderings for the Rust template-table test.

Renders every v1 train/extra template lambda (eval/generate_eval.py and
training/generate_dataset.py) with fixed sentinel arguments and writes
crates/minicut/tests/golden/v1_templates.json: {id: [rendered texts]}.
The Rust test renders each non-"v3_" row of templates.rs with the same
sentinels and requires its text to be one of v1's renderings for that id,
so reused ids keep v1's exact wording.

    python scripts/export_v1_templates.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval import generate_eval as ge  # noqa: E402
from training import generate_dataset as gd  # noqa: E402

OUT = ROOT / "crates/minicut/tests/golden/v1_templates.json"

ARGS = [
    {"start": 0.0, "end": 222.0}, {"start": 111.0, "end": 222.0},
    {"width": 1280, "height": 720}, {"width": 960, "height": 540}, {"width": 333, "height": 444},
    {"width": 1920, "height": 1080},
    {"factor": 1.25}, {"factor": 0.8},
    {"codec": "libx264", "container": "mkv"}, {"format": "mp3"},
    {"inputs": ["clip.mp4", "clip.mp4"]}, {"inputs": ["clip.mp4", "clip.mp4", "clip.mp4"]},
    {},
]
POOLS = [ge.TRIM_TRAIN, ge.CUT_TRAIN, ge.SCALE_TRAIN, ge.SPEED_TRAIN, ge.CONVERT_TRAIN,
         ge.EXTRACT_TRAIN, ge.MUTE_TRAIN, ge.CONCAT_TRAIN, gd.TRIM_EXTRA, gd.CUT_EXTRA,
         gd.SCALE_EXTRA, gd.SPEED_EXTRA, gd.CONVERT_EXTRA, gd.EXTRACT_EXTRA, gd.MUTE_NEW,
         gd.CONCAT_NEW]


def main() -> int:
    out: dict[str, set[str]] = {}
    for pool in POOLS:
        for tid, render in pool:
            for args in ARGS:
                try:
                    text = render(args, "base_30s")
                except (KeyError, TypeError, ValueError):
                    continue  # this sentinel set does not fit the template
                out.setdefault(tid, set()).add(text)
    data = {k: sorted(v) for k, v in sorted(out.items())}
    OUT.write_text(json.dumps(data, indent=1) + "\n")
    print(f"wrote {OUT} ({len(data)} ids)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
