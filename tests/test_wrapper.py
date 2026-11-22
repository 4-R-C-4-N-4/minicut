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
from hashlib import sha256
from pathlib import Path
from subprocess import TimeoutExpired
from unittest.mock import patch

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

    def test_publish_never_replaces_existing_name(self):
        from minicut.wrapper import verify_output

        source = self.tmp / "input.mp4"
        digest = sha256(source.read_bytes()).digest()
        occupied = self.tmp / "input_trim.mp4"

        def verify(*args):
            self.assertFalse(occupied.exists(), "final path visible before verification")
            occupied.write_bytes(b"concurrent writer")
            return verify_output(*args)

        with patch("minicut.wrapper.verify_output", side_effect=verify):
            result = edit({**CASES[0][0], "output": source.name}, self.tmp)
        self.assertTrue(result["ok"], result)
        self.assertEqual(Path(result["output"]), self.tmp / "input_trim_v2.mp4")
        self.assertEqual(occupied.read_bytes(), b"concurrent writer")
        self.assertEqual(sha256(source.read_bytes()).digest(), digest)

    def test_failed_verification_never_publishes_render(self):
        with patch("minicut.wrapper.verify_output", return_value=(False, "test failure")):
            result = edit(CASES[0][0], self.tmp)
        self.assertFalse(result["ok"])
        self.assertNotIn("output", result)
        self.assertEqual({p.name for p in self.tmp.iterdir()}, {"input.mp4"})

    def test_render_timeout_removes_partial_file(self):
        def timeout(argv):
            Path(argv[-1]).write_bytes(b"partial render")
            raise TimeoutExpired(argv, 120)

        with patch("minicut.wrapper.run", side_effect=timeout):
            result = edit(CASES[0][0], self.tmp)
        self.assertFalse(result["ok"])
        self.assertEqual({p.name for p in self.tmp.iterdir()}, {"input.mp4"})

    def test_derivative_can_go_in_another_directory(self):
        with tempfile.TemporaryDirectory() as destination:
            result = edit(CASES[0][0], self.tmp, output_dir=destination)
            self.assertTrue(result["ok"], result)
            self.assertEqual(Path(result["output"]).parent, Path(destination))
            self.assertEqual({p.name for p in Path(destination).iterdir()}, {"input_trim.mp4"})
        self.assertEqual({p.name for p in self.tmp.iterdir()}, {"input.mp4"})

    def test_publish_preserves_dangling_symlink(self):
        occupied = self.tmp / "input_trim.mp4"
        target = self.tmp / "absent.mp4"
        occupied.symlink_to(target)
        result = edit(CASES[0][0], self.tmp)
        self.assertTrue(result["ok"], result)
        self.assertEqual(Path(result["output"]).name, "input_trim_v2.mp4")
        self.assertEqual(occupied.readlink(), target)
        self.assertFalse(target.exists())

    def test_retry_is_validated_before_a_second_render(self):
        with patch("minicut.wrapper.verify_output", return_value=(False, "retry")) as verify:
            result = edit(CASES[0][0], self.tmp,
                          llm=lambda _: {**CASES[0][0], "end": 999})
        self.assertFalse(result["ok"])
        verify.assert_called_once()
        self.assertEqual({p.name for p in self.tmp.iterdir()}, {"input.mp4"})

    def test_outside_symlink_is_rejected_before_probe(self):
        alias = self.tmp / "alias.mp4"
        alias.symlink_to(FIXTURE)
        with patch("minicut.wrapper.probe_file") as probe:
            result = edit({**CASES[0][0], "input": alias.name}, self.tmp)
        self.assertFalse(result["ok"])
        probe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
