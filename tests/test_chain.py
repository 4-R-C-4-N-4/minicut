"""Metadata-chain tests — tokens, collision bumps, chain survival.

Run: python -m unittest discover -s tests
Requires fixtures in factory/fixtures (run factory/generate.py --all first).
"""
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minicut.chain import append_chain, read_chain, token_for  # noqa: E402
from minicut.wrapper import edit  # noqa: E402

FIXTURE = ROOT / "factory" / "fixtures" / "base_30s" / "input.mp4"


class TestTokens(unittest.TestCase):
    def test_tokens(self):
        self.assertEqual(token_for({"tool": "trim", "start": 12.0, "end": 45.0}), "trim:12-45")
        self.assertEqual(token_for({"tool": "cut_out", "start": 0.0, "end": 10.0}), "cutout:0-10")
        self.assertEqual(token_for({"tool": "concat", "inputs": ["a.mp4", "b.mp4"]}), "concat:2")
        self.assertEqual(token_for({"tool": "scale", "width": 1280, "height": 720}), "scale:1280x720")
        self.assertEqual(token_for({"tool": "speed", "factor": 2.0}), "speed:2")
        self.assertEqual(token_for({"tool": "convert", "codec": "libx264", "container": "mkv"}), "convert:libx264-mkv-crf23")
        self.assertEqual(token_for({"tool": "extract_audio", "format": "mp3"}), "extract:mp3")
        self.assertEqual(token_for({"tool": "mute"}), "mute")

    def test_append(self):
        self.assertEqual(append_chain(None, "trim:12-45"), "trim:12-45")
        self.assertEqual(append_chain("", "trim:12-45"), "trim:12-45")
        self.assertEqual(append_chain("trim:12-45", "speed:2"), "trim:12-45;speed:2")


class TestChainEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        shutil.copy(FIXTURE, self.tmp / "input.mp4")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_chain_survives_trim_then_speed(self):
        r1 = edit({"tool": "trim", "input": "input.mp4", "start": 0.0, "end": 20.0}, self.tmp)
        self.assertTrue(r1["ok"], r1)
        self.assertEqual(r1["chain"], "trim:0-20")
        self.assertEqual(read_chain(r1["output"]), "trim:0-20")
        # Second edit takes the first output as input: chain grows.
        out1 = Path(r1["output"]).name
        r2 = edit({"tool": "speed", "input": out1, "factor": 2.0}, self.tmp)
        self.assertTrue(r2["ok"], r2)
        self.assertEqual(r2["chain"], "trim:0-20;speed:2")

    def test_model_output_ignored_and_bumped(self):
        # The baseline_v2 failure: model chose a .vp9 extension ffmpeg
        # cannot mux. The wrapper must ignore it and use its own name.
        r = edit({"tool": "convert", "input": "input.mp4", "codec": "vp9",
                  "container": "webm", "output": "clip_webm.vp9"}, self.tmp)
        self.assertTrue(r["ok"], r)
        self.assertTrue(Path(r["output"]).suffix == ".webm")
        self.assertFalse(Path(self.tmp / "clip_webm.vp9").exists())
        # Same edit twice: second run bumps, never overwrites the first.
        r2 = edit({"tool": "convert", "input": "input.mp4", "codec": "vp9",
                   "container": "webm"}, self.tmp)
        self.assertTrue(r2["ok"], r2)
        self.assertNotEqual(r["output"], r2["output"])
        self.assertTrue(Path(r2["output"]).name.endswith("_v2.webm"))
        self.assertTrue(Path(r["output"]).exists())

    def test_fresh_file_has_no_chain(self):
        self.assertIsNone(read_chain(self.tmp / "input.mp4"))


if __name__ == "__main__":
    unittest.main()
