"""Run the release benchmark calibration controls (CPU only, no model).

    python -m eval.release controls [--fixtures DIR] [--report PATH]

Exits 0 only if every control behaves as expected:
  * the pinned roster loads, and a changed or incomplete roster is rejected;
  * the roster is separate from development and training data;
  * each expected edit passes both axes, and its wrong-but-executable
    control fails intent while leaving every source unchanged;
  * a clarification on a clear request fails and publishes nothing;
  * each expected clarification passes and publishes nothing, and a
    guessed edit on an ambiguous request fails;
  * an editor that modifies a source is caught.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import bench


def _roster_controls() -> list[str]:
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name in bench.FROZEN_FILES:
            shutil.copyfile(bench.RELEASE_DIR / name, tmp / name)
        cases = tmp / "cases.jsonl"
        lines = cases.read_text().splitlines(True)
        variants = {
            "changed byte": (lines[0].replace("seconds", "secs", 1), lines[1:], False),
            "dropped case (repinned)": ("", lines[:-1], True),
            "duplicated case (repinned)": ("", lines[:-1] + [lines[0]], True),
        }
        for label, (head, rest, repin) in variants.items():
            cases.write_text(head + "".join(rest))
            pin = dict(bench.PIN, **{"cases.jsonl": bench.sha256_file(cases)}) if repin else bench.PIN
            try:
                bench.load_roster(tmp, pin=pin)
                failures.append(f"roster control accepted a {label}")
            except bench.RosterError:
                pass
    return failures


def _case_controls(case: dict, fixtures: Path, roster: bench.Roster) -> tuple[list[dict], list[str]]:
    expected = case["expected"]
    attempts = []
    if expected["kind"] == "edit":
        attempts.append(("oracle", expected["call"], True))
        attempts.append(("wrong", case["control_wrong"], False))
        asked = {"tool": "clarify", "question": "Can you say more?", "missing": ["start", "end"]}
        attempts.append(("needless_clarify", asked, False))
    else:
        oracle = {"tool": "clarify", "question": "What should I use?", "missing": expected["missing"]}
        attempts.append(("oracle", oracle, True))
        attempts.append(("guessed_edit", case["control_wrong"], False))
    rows, failures = [], []
    for label, call, should_pass in attempts:
        row = dict(bench.run_case(case, call, fixtures, roster=roster), control=label)
        rows.append(row)
        if row["intent_pass"] != should_pass:
            failures.append(f"{case['id']} {label}: intent_pass={row['intent_pass']} ({row['reason']})")
        if not row["sources_unchanged"]:
            failures.append(f"{case['id']} {label}: a source changed")
        published = len(row["published"])
        if label == "oracle" and published != (1 if expected["kind"] == "edit" else 0):
            failures.append(f"{case['id']} oracle published {published} files")
        if label == "needless_clarify" and published:
            failures.append(f"{case['id']} clarification published files")
    return rows, failures


def controls(fixtures: Path, report: Path | None) -> int:
    failures = []
    try:
        roster = bench.load_roster(bench.RELEASE_DIR)
    except bench.RosterError as exc:
        print(f"FAIL pinned roster: {exc}")
        return 1
    failures += _roster_controls()
    failures += bench.separation_problems(roster)
    fixtures = bench.ensure_fixtures(roster, fixtures)

    def tamper(call, workdir):
        with (Path(workdir) / next(iter(case0["files"]))).open("ab") as f:
            f.write(b"\0")
        return {"ok": False, "error": "tampered"}

    case0 = roster.cases[0]
    if bench.run_case(case0, case0["expected"]["call"], fixtures, editor=tamper,
                      roster=roster)["sources_unchanged"]:
        failures.append("source-tampering editor was not detected")

    workers = max(1, min(8, (os.cpu_count() or 2) // 2))
    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(lambda c: _case_controls(c, fixtures, roster), roster.cases))
    rows = [row for case_rows, _ in results for row in case_rows]
    failures += [f for _, case_failures in results for f in case_failures]

    by_control: dict[str, list[int]] = {}
    for row in rows:
        tally = by_control.setdefault(row["control"], [0, 0, 0])
        tally[0] += row["intent_pass"]
        tally[1] += row["sources_unchanged"]
        tally[2] += 1
    print(f"roster: {len(roster.cases)} cases {bench.PIN['categories']}")
    for label, (intent, kept, n) in sorted(by_control.items()):
        print(f"{label:>17}: intent pass {intent}/{n}, sources unchanged {kept}/{n}")
    if report:
        report.write_text(json.dumps({"pin": bench.PIN, "rows": rows, "failures": failures},
                                     indent=2) + "\n")
    for failure in failures:
        print(f"FAIL {failure}")
    print("controls: " + ("PASS" if not failures else f"FAIL ({len(failures)})"))
    return 0 if not failures else 1


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.release", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("controls", help="run every calibration control")
    run.add_argument("--fixtures", type=Path, default=bench.DEFAULT_FIXTURES)
    run.add_argument("--report", type=Path)
    args = parser.parse_args()
    return controls(args.fixtures, args.report)


if __name__ == "__main__":
    sys.exit(main())
