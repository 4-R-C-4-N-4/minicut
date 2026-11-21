"""Training dataset tests — train_v1.jsonl + train_v1_meta.jsonl.

Run: python -m unittest discover -s tests
Checks format, the exact system prompt, label validity, and freeze
compliance re-derived from eval/freeze.json and the frozen eval files:
no held-out template id, no reserved fixture, no combo-pool arg, and
no (prompt, label) pair that appears in any eval split.
"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minicut.llm.prompts import SYSTEM_PROMPT  # noqa: E402
from training.generate_dataset import _canon  # noqa: E402

DATA = ROOT / "training" / "train_v1.jsonl"
META = ROOT / "training" / "train_v1_meta.jsonl"
FREEZE = ROOT / "eval" / "freeze.json"
TOOLS = {"trim", "cut_out", "concat", "scale", "speed", "convert",
         "extract_audio", "mute", "clarify"}


def _load():
    rows = [json.loads(l) for l in DATA.read_text().splitlines()]
    meta = [json.loads(l) for l in META.read_text().splitlines()]
    return rows, meta


class TestDatasetFormat(unittest.TestCase):
    def test_rows_and_meta_aligned(self):
        rows, meta = _load()
        self.assertEqual(len(rows), len(meta))
        self.assertGreaterEqual(len(rows), 2000)
        for r in rows:
            self.assertEqual(list(r.keys()), ["messages"])
            self.assertEqual(len(r["messages"]), 3)
            roles = [m["role"] for m in r["messages"]]
            self.assertEqual(roles, ["system", "user", "assistant"])

    def test_system_prompt_exact(self):
        rows, _ = _load()
        for r in rows:
            self.assertEqual(r["messages"][0]["content"], SYSTEM_PROMPT)

    def test_assistant_is_json_tool_call_or_general_text(self):
        rows, meta = _load()
        for r, m in zip(rows, meta):
            content = r["messages"][2]["content"]
            if m["kind"] == "general":
                self.assertFalse(content.lstrip().startswith("{"),
                                 f"general example must not be a JSON tool call: {content[:60]}")
                continue
            call = json.loads(content)  # must parse
            self.assertIn(call["tool"], TOOLS)
            self.assertEqual(call["tool"], m["tool"])

    def test_clarify_has_nonempty_missing(self):
        rows, meta = _load()
        for r, m in zip(rows, meta):
            if m["tool"] != "clarify":
                continue
            call = json.loads(r["messages"][2]["content"])
            self.assertIsInstance(call.get("missing"), list)
            self.assertTrue(call["missing"], "clarify missing list must not be empty")
            self.assertTrue(call.get("question"), "clarify needs a question")

    def test_mix_proportions(self):
        _, meta = _load()
        n = len(meta)
        kinds = {}
        for m in meta:
            k = "general" if m["kind"] == "general" else m["tool"]
            kinds[k] = kinds.get(k, 0) + 1
        self.assertAlmostEqual(kinds.get("clarify", 0) / n, 0.117, delta=0.02)
        self.assertAlmostEqual(kinds.get("general", 0) / n, 0.12, delta=0.02)


class TestFreezeCompliance(unittest.TestCase):
    def test_no_heldout_template_or_reserved_fixture(self):
        freeze = json.loads(FREEZE.read_text())
        heldout = set(freeze["heldout_template_ids"])
        _, meta = _load()
        for m in meta:
            self.assertNotIn(m["template_id"], heldout,
                             f"held-out template {m['template_id']} in training data")
            self.assertNotEqual(m["fixture"], freeze["reserved_fixture"])

    def test_no_combo_pool_args(self):
        freeze = json.loads(FREEZE.read_text())
        cp = freeze["combo_arg_pools"]
        rows, meta = _load()
        for r, m in zip(rows, meta):
            if m["kind"] != "tool":
                continue
            c = json.loads(r["messages"][2]["content"])
            t = c["tool"]
            if t == "trim":
                self.assertNotIn([c["start"], c["end"]], cp["trim"])
            elif t == "cut_out":
                self.assertNotIn([c["start"], c["end"]], cp["cut_out"])
            elif t == "scale":
                self.assertNotIn([c["width"], c["height"]], cp["scale"])
            elif t == "speed":
                self.assertNotIn(c["factor"], cp["speed"])
            elif t == "convert":
                self.assertNotIn([c["codec"], c["container"], c["crf"]], cp["convert"])
            elif t == "extract_audio":
                self.assertNotIn(c["format"], cp["extract_audio"])

    def test_no_eval_case_sampled(self):
        eval_pairs = set()
        for path in sorted((ROOT / "eval").glob("eval_*.jsonl")):
            for line in path.read_text().splitlines():
                c = json.loads(line)
                eval_pairs.add((c["prompt"], _canon(c["expected"])))
        rows, meta = _load()
        for r, m in zip(rows, meta):
            if m["kind"] == "general":
                continue
            c = json.loads(r["messages"][2]["content"])
            self.assertNotIn((r["messages"][1]["content"], _canon(c)), eval_pairs,
                             f"eval case sampled into training: {r['messages'][1]['content']!r}")

    def test_no_heldout_prompt_wording(self):
        """No training prompt equals a held-out template rendering for
        the same args (re-rendered here from eval.generate_eval)."""
        sys.path.insert(0, str(ROOT))
        from eval.generate_eval import TEMPLATES
        rows, meta = _load()
        heldout_prompts = set()
        for tool, (_, held) in TEMPLATES.items():
            if tool == "clarify":
                for _tid, prompt, _cargs in held:
                    heldout_prompts.add(prompt)
                continue
            for _tid, render in held:
                heldout_prompts.add(render)
        for r, m in zip(rows, meta):
            if m["kind"] == "general":
                continue
            prompt = r["messages"][1]["content"]
            call = json.loads(r["messages"][2]["content"])
            tool = call["tool"]
            if tool == "clarify":
                self.assertNotIn(prompt, heldout_prompts,
                                 f"held-out clarify wording in training: {prompt!r}")
                continue
            args = {k: v for k, v in call.items()
                    if k not in ("tool", "input", "output")}
            _, held = TEMPLATES[tool]
            import re
            norm = re.sub(r"\S+\.mp4", "clip.mp4", prompt)
            for _tid, render in held:
                try:
                    rendered = render(args, "base_30s")
                except Exception:
                    continue
                self.assertNotEqual(norm, rendered,
                                    f"training prompt equals held-out render ({_tid}) "
                                    f"up to filename: {prompt!r}")


if __name__ == "__main__":
    unittest.main()
