"""Release benchmark: roster freeze, separation, and grading controls.

The full media control run is `python -m eval.release controls`; these
tests cover the roster gate and one keep/remove pair end to end.
"""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.release import bench  # noqa: E402


class TestRosterGate(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        for name in bench.FROZEN_FILES:
            shutil.copy(bench.RELEASE_DIR / name, self.tmp / name)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_pinned_roster_is_accepted(self):
        roster = bench.load_roster(bench.RELEASE_DIR)
        self.assertEqual(len(roster.cases), bench.PIN["cases"])

    def test_changed_roster_is_rejected(self):
        path = self.tmp / "cases.jsonl"
        path.write_text(path.read_text().replace("seconds", "secs", 1))
        with self.assertRaisesRegex(bench.RosterError, "digest"):
            bench.load_roster(self.tmp)

    def test_incomplete_roster_is_rejected_even_if_repinned(self):
        path = self.tmp / "cases.jsonl"
        path.write_text("".join(path.read_text().splitlines(True)[:-1]))
        pin = dict(bench.PIN, **{"cases.jsonl": bench.sha256_file(path)})
        with self.assertRaisesRegex(bench.RosterError, "cases"):
            bench.load_roster(self.tmp, pin=pin)

    def test_roster_is_separate_from_development_and_training(self):
        roster = bench.load_roster(bench.RELEASE_DIR)
        self.assertEqual(bench.separation_problems(roster), [])


class TestTimeline(unittest.TestCase):
    SOURCE = [(0, 6, "yellow"), (6, 12, "cyan"), (12, 18, "magenta")]

    def test_keep_and_remove_of_same_range_differ(self):
        keep = bench.expected_timeline({"tool": "trim", "start": 0, "end": 9},
                                       {"x": self.SOURCE}, "x")
        remove = bench.expected_timeline({"tool": "cut_out", "start": 0, "end": 9},
                                         {"x": self.SOURCE}, "x")
        self.assertEqual(keep, [(0, 6, "yellow"), (6, 9, "cyan")])
        self.assertEqual(remove, [(0, 3, "cyan"), (3, 9, "magenta")])


class TestMediaControls(unittest.TestCase):
    def test_keep_remove_pair_and_clarification(self):
        roster = bench.load_roster(bench.RELEASE_DIR)
        fixtures = bench.ensure_fixtures(roster, bench.DEFAULT_FIXTURES)
        case = next(c for c in roster.cases if c["category"] == "keep_remove")
        right = bench.run_case(case, case["expected"]["call"], fixtures)
        wrong = bench.run_case(case, case["control_wrong"], fixtures)
        self.assertTrue(right["intent_pass"], right)
        self.assertTrue(right["sources_unchanged"], right)
        self.assertEqual(len(right["published"]), 1)
        self.assertFalse(wrong["intent_pass"], wrong)
        self.assertTrue(wrong["sources_unchanged"], wrong)
        asked = bench.run_case(case, {"tool": "clarify", "question": "Which part?",
                                      "missing": ["start", "end"]}, fixtures)
        self.assertFalse(asked["intent_pass"])
        self.assertEqual(asked["published"], [])

    def test_source_change_is_detected(self):
        roster = bench.load_roster(bench.RELEASE_DIR)
        fixtures = bench.ensure_fixtures(roster, bench.DEFAULT_FIXTURES)
        case = next(c for c in roster.cases if c["category"] == "keep_remove")

        def tampering_editor(call, workdir, **_):
            source = Path(workdir) / next(iter(case["files"]))
            with source.open("ab") as f:
                f.write(b"\0")
            return {"ok": False, "error": "tampered"}

        result = bench.run_case(case, case["expected"]["call"], fixtures, editor=tampering_editor)
        self.assertFalse(result["sources_unchanged"])


if __name__ == "__main__":
    unittest.main()
