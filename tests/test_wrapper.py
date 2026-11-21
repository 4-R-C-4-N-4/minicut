"""Wrapper tests — hand-written JSON, no model.

Run: python -m pytest tests/ -q   (or python -m unittest discover)
Requires fixtures in factory/fixtures (run factory/generate.py --all first).
Each of the 8 v1 tools must pass verification on base_30s, plus
negative checks: invalid JSON must be rejected before ffmpeg runs.
"""
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minicut.wrapper import edit  # noqa: E402

FIXTURE = ROOT / "factory" / "fixtures" / "base_30s" / "input.mp4"

CASES = [
    ({"tool": "trim", "input": "input.mp4", "start": 12.0, "end": 27.0}, "trim"),
    ({"tool": "cut_out", "input": "input.mp4", "start": 10.0, "end": 20.0}, "cut_out"),
    ({"tool": "concat", "inputs": ["input.mp4", "input.mp4"]}, "concat"),
    ({"tool": "scale", "input": "input.mp4", "width": 640, "height": 360}, "scale"),
    ({"tool": "speed", "input": "input.mp4", "factor": 2.0}, "speed"),
    ({"tool": "convert", "input": "input.mp4", "codec": "libx264", "container": "mkv", "crf": 23}, "convert"),
    ({"tool": "extract_audio", "input": "input.mp4", "format": "mp3"}, "extract_audio"),
    ({"tool": "mute", "input": "input.mp4"}, "mute"),
    ({"tool": "clarify", "question": "What times?", "missing": ["start", "end"]}, "clarify"),
]


class TestWrapper(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        shutil.copy(FIXTURE, self.tmp / "input.mp4")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_all_v1_tools(self):
        for call, name in CASES:
            with self.subTest(tool=name):
                r = edit(dict(call), self.tmp)
                self.assertTrue(r["ok"], f"{name} failed: {r}")
                if name != "clarify":
                    self.assertTrue(Path(r["output"]).exists())

    def test_invalid_rejected_before_ffmpeg(self):
        bad = [
            {"tool": "trim", "input": "input.mp4", "start": 25.0, "end": 10.0},
            {"tool": "trim", "input": "input.mp4", "start": 0.0, "end": 99.0},
            {"tool": "scale", "input": "input.mp4", "width": 641, "height": 360},
            {"tool": "speed", "input": "input.mp4", "factor": 9.0},
            {"tool": "nope", "input": "input.mp4"},
            {"tool": "trim", "input": "../escape.mp4", "start": 0.0, "end": 5.0},
        ]
        for call in bad:
            with self.subTest(call=call):
                r = edit(dict(call), self.tmp)
                self.assertFalse(r["ok"], f"should have rejected {call}")


if __name__ == "__main__":
    unittest.main()
