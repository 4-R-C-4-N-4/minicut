#!/usr/bin/env python
"""Acceptance gate for generated Phase 4 training rows.

Checks, each printing `PASS/FAIL/SKIP name: details` to stdout:
  pilot_filters  -- no row would have been dropped by the pilot's filters
  validator      -- minicut.validate.validate(label, probe, workdir) == []
  grammar        -- (optional) every assistant target parses under a GBNF
  preflight      -- (optional) training.train_lora --preflight succeeds
  release        -- (optional) trim/cut_out release cases match a built
                    `minicut` binary's `meaning` subcommand output

Exits 1 if any check FAILs.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _rows_and_meta(rows_path: Path, meta_path: Path):
    rows = _read_jsonl(rows_path)
    meta = _read_jsonl(meta_path)
    if len(rows) != len(meta):
        raise ValueError(f"rows/meta count mismatch: {len(rows)} vs {len(meta)}")
    out = []
    for row, m in zip(rows, meta):
        messages = row["messages"]
        prompt = messages[1]["content"]
        label = json.loads(messages[2]["content"])
        out.append((prompt, label, m))
    return out


# --- (a) pilot_filters ------------------------------------------------------
def check_pilot_filters(cases: list[tuple], root: Path = ROOT) -> tuple[bool, str]:
    from training.build_pilot import pilot_drop_reason

    freeze = json.loads((root / "eval/freeze.json").read_text())
    combos = freeze["combo_arg_pools"]["convert"]
    reasons: Counter[str] = Counter()
    for prompt, label, meta in cases:
        # pilot_drop_reason needs {"kind", "template_id"}; the verified
        # meta's "kind" is "tool"/"clarify" (never "general" -- that v1
        # bucket has no generator analogue), so only template_id-driven
        # drops (held-out ids, train_clarify_invalid) can ever fire here.
        pilot_meta = {"kind": meta.get("kind", "tool"), "template_id": meta.get("template_id")}
        reason = pilot_drop_reason(prompt, label, pilot_meta, combos)
        if reason is not None:
            reasons[reason] += 1
    ok = not reasons
    detail = f"{len(cases)} rows; drops by reason: {dict(sorted(reasons.items()))}"
    return ok, detail


# --- (b) validator -----------------------------------------------------------
def check_validator(cases: list[tuple], root: Path = ROOT) -> tuple[bool, str]:
    from minicut.probe import probe_file
    from minicut.validate import validate

    probes: dict[str | None, object] = {None: None}
    errors_by_fixture: Counter[str] = Counter()
    total_errors = 0
    examples = []
    for prompt, label, meta in cases:
        fixture = meta.get("fixture")
        if fixture not in probes:
            probes[fixture] = probe_file(root / "factory/fixtures" / fixture / "input.mp4")
        errs = validate(label, probes[fixture], Path("."))
        if errs:
            total_errors += 1
            errors_by_fixture[fixture or "<none>"] += 1
            if len(examples) < 5:
                examples.append(f"{prompt!r}: {errs}")
    ok = total_errors == 0
    detail = f"{len(cases)} rows; {total_errors} failed validate()"
    if examples:
        detail += "; e.g. " + "; ".join(examples)
    return ok, detail


# --- (c) grammar --------------------------------------------------------------
def check_grammar(cases: list[tuple], validator: str, grammar: str, tmpdir: Path) -> tuple[bool, str]:
    tmpdir.mkdir(parents=True, exist_ok=True)

    def run_one(i, label):
        path = tmpdir / f"grammar_{i}.txt"
        path.write_text(json.dumps(label))
        try:
            out = subprocess.run([validator, grammar, str(path)], capture_output=True,
                                  text=True, timeout=30)
            text = out.stdout + out.stderr
        except Exception as exc:  # noqa: BLE001
            text = str(exc)
        return i, "Input string is valid" in text, text

    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(run_one, i, label) for i, (_, label, _) in enumerate(cases)]
        for fut in concurrent.futures.as_completed(futures):
            i, ok, text = fut.result()
            if not ok:
                failures.append((i, text.strip()[:200]))
    ok = not failures
    detail = f"{len(cases)} rows; {len(failures)} failed grammar validation"
    if failures:
        detail += "; e.g. " + "; ".join(f"row {i}: {t}" for i, t in failures[:5])
    return ok, detail


# --- (d) preflight -------------------------------------------------------------
def check_preflight(rows_path: Path, root: Path = ROOT) -> tuple[bool, str]:
    cmd = [sys.executable, "-m", "training.train_lora", "--data", str(rows_path), "--preflight"]
    out = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
    ok = out.returncode == 0
    detail = (out.stdout + out.stderr).strip().splitlines()
    detail = detail[-1] if detail else f"exit {out.returncode}"
    return ok, detail


# --- (e) release meaning --------------------------------------------------------
def _expected_runs(call: dict, duration: float) -> int:
    if call["tool"] == "trim":
        return 1
    start, end = float(call["start"]), float(call["end"])
    return 1 if (start == 0 or end == duration) else 2


def check_release(minicut_bin: str, root: Path = ROOT) -> tuple[bool, str]:
    from eval.release.bench import _timeline, expected_timeline

    release_dir = root / "eval/release"
    fixtures = json.loads((release_dir / "fixtures.json").read_text())
    cases = _read_jsonl(release_dir / "cases.jsonl")

    checked, failures = 0, []
    for case in cases:
        expected = case["expected"]
        if expected.get("kind") != "edit":
            continue
        call = expected["call"]
        if call["tool"] not in {"trim", "cut_out"}:
            continue
        checked += 1
        files = case["files"]
        name = call["input"]
        fixture_id = files[name]
        spec = fixtures[fixture_id]
        specs = {n: fixtures[fid] for n, fid in files.items()}
        timelines = {n: _timeline(s) for n, s in specs.items()}
        pieces = expected_timeline(call, timelines, name)
        expected_ms = round(1000 * pieces[-1][1]) if pieces else 0
        expected_runs = _expected_runs(call, spec["duration"])

        probe = {"duration_s": spec["duration"], "width": spec["width"], "height": spec["height"],
                  "has_audio": spec["audio"]}
        cmd = [minicut_bin, "meaning", "--call", json.dumps(call), "--probe", json.dumps(probe)]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        try:
            got = json.loads(out.stdout)
            kept = got["kept"]
            got_ms = sum(b - a for a, b in kept)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{case['id']}: could not parse output: {exc}; stdout={out.stdout!r}")
            continue
        if abs(got_ms - expected_ms) > 1 or len(kept) != expected_runs:
            failures.append(f"{case['id']}: got kept={kept} (total {got_ms}ms, {len(kept)} runs), "
                            f"expected total {expected_ms}ms, {expected_runs} runs")
    ok = not failures
    detail = f"{checked} trim/cut_out cases; {len(failures)} mismatched"
    if failures:
        detail += "; " + "; ".join(failures[:5])
    return ok, detail


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--gbnf-validator")
    parser.add_argument("--grammar")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--minicut-bin")
    parser.add_argument("--tmpdir", type=Path, default=Path.cwd() / "accept_dataset_tmp")
    args = parser.parse_args()

    cases = _rows_and_meta(args.rows, args.meta)
    results = []

    ok, detail = check_pilot_filters(cases)
    results.append(("PASS" if ok else "FAIL", "pilot_filters", detail))

    ok, detail = check_validator(cases)
    results.append(("PASS" if ok else "FAIL", "validator", detail))

    if args.gbnf_validator and args.grammar:
        ok, detail = check_grammar(cases, args.gbnf_validator, args.grammar, args.tmpdir)
        results.append(("PASS" if ok else "FAIL", "grammar", detail))
    else:
        results.append(("SKIP", "grammar", "no --gbnf-validator/--grammar given"))

    if args.preflight:
        ok, detail = check_preflight(args.rows)
        results.append(("PASS" if ok else "FAIL", "preflight", detail))
    else:
        results.append(("SKIP", "preflight", "no --preflight given"))

    if args.minicut_bin:
        ok, detail = check_release(args.minicut_bin)
        results.append(("PASS" if ok else "FAIL", "release", detail))
    else:
        results.append(("SKIP", "release", "no --minicut-bin given"))

    for status, name, detail in results:
        print(f"{status} {name}: {detail}")
    if any(status == "FAIL" for status, _, _ in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
