import json
import re
import unittest
from pathlib import Path

from training.build_pilot import build

ROOT = Path(__file__).resolve().parents[1]


class TestPilot(unittest.TestCase):
    def test_reproducible_json_only_and_unique(self):
        rows, meta = build(ROOT)
        for name, payload in (("train_pilot_v1", rows), ("train_pilot_v1_meta", meta)):
            self.assertEqual((ROOT / "training" / f"{name}.jsonl").read_text(),
                             "".join(json.dumps(r) + "\n" for r in payload))
        prompts = [r["messages"][1]["content"] for r in rows]
        self.assertEqual(len(prompts), len(set(prompts)))
        self.assertGreater(len(rows), 2000)
        self.assertEqual({m["tool"] for m in meta}, {"trim", "cut_out", "scale", "speed", "concat",
                                                    "mute", "convert", "extract_audio", "clarify"})
        freeze = json.loads((ROOT / "eval/freeze.json").read_text())
        frozen = {json.loads(line)["prompt"] for p in (ROOT / "eval").glob("eval_*.jsonl")
                  for line in p.read_text().splitlines()}
        for row, m in zip(rows, meta, strict=True):
            p = row["messages"][1]["content"]
            c = json.loads(row["messages"][2]["content"])
            self.assertNotIn(p, frozen)
            self.assertNotIn(m["template_id"], freeze["heldout_template_ids"])
            self.assertNotEqual(m["fixture"], freeze["reserved_fixture"])
            self.assertNotEqual(m["template_id"], "train_clarify_invalid")
            if c["tool"] == "scale":
                self.assertRegex(p, rf"{c['width']}(?:x| by ){c['height']}")
            if c["tool"] == "concat" and len(c["inputs"]) == 3:
                self.assertRegex(p, r"\bthree\b")
            if c["tool"] == "convert":
                self.assertEqual(c["crf"], 23)
                self.assertNotIn([c["codec"], c["container"], c["crf"]],
                                 freeze["combo_arg_pools"]["convert"])
            if c["tool"] in ("trim", "cut_out"):
                self.assertNotRegex(p, r"half|%")
                if re.search(r"\d+:\d+", p):
                    self.assertEqual(c["start"], int(c["start"]))
                    self.assertEqual(c["end"], int(c["end"]))
