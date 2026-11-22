"""Intent-aware MiniCut development eval, using a CPU oracle or llama.cpp.

Task success requires canonical intent plus verified execution, or the
expected accepted clarification. Wrapper verification alone is diagnostic.
See eval/README.md for commands and the frozen-label overlay.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minicut.llm.client import predict  # noqa: E402
from minicut.wrapper import edit  # noqa: E402

EVAL_DIR = ROOT / "eval"
FIXTURES_DIR = ROOT / "factory" / "fixtures"
SCORER = "intent_v2"

# Args compared for exact equality, per tool. input/output are excluded:
# the wrapper derives output, and the harness fixes the input filename.
COMPARE_ARGS = {
    "trim": ["start", "end"],
    "cut_out": ["start", "end"],
    "concat": ["inputs"],
    "scale": ["width", "height"],
    "speed": ["factor"],
    "convert": ["codec", "container", "crf"],
    "extract_audio": ["format"],
    "mute": [],
}


def _args_correct(expected: dict, predicted: dict) -> bool:
    if expected.get("tool") != predicted.get("tool"):
        return False
    if expected["tool"] == "clarify":
        missing = predicted.get("missing")
        return (isinstance(missing, list) and all(isinstance(k, str) for k in missing)
                and len(missing) == len(set(missing))
                and set(expected.get("missing", [])) == set(missing))
    for key in COMPARE_ARGS.get(expected["tool"], []):
        exp_v, pred_v = expected.get(key), predicted.get(key)
        if key == "crf":
            exp_v = expected.get(key, 23)
            pred_v = predicted.get(key, 23)
        if isinstance(exp_v, float) or isinstance(pred_v, float):
            if (not isinstance(exp_v, (float, int)) or not isinstance(pred_v, (float, int))
                    or isinstance(pred_v, bool)):
                return False
            if not math.isclose(exp_v, pred_v, rel_tol=0, abs_tol=1e-6):
                return False
        elif exp_v != pred_v:
            return False
    return True


def _expected_intent(case: dict) -> dict:
    """Repair prompt/label discrepancies without rewriting frozen cases."""
    expected = dict(case["expected"])
    prompt = case["prompt"]
    if expected["tool"] == "convert" and not re.search(r"\bcrf\b", prompt, re.I):
        expected["crf"] = 23
    if expected["tool"] == "concat" and prompt in {
        "concatenate clip.mp4 with itself", "combine clip.mp4 with itself into one file",
    }:
        expected["inputs"] = ["clip.mp4", "clip.mp4"]
    return expected


def _http_predict(prompt: str, base_url: str, use_grammar: bool, timeout_s: int = 120) -> str:
    """One chat completion from a llama.cpp server, via the CLI's client."""
    return predict(prompt, base_url, use_grammar=use_grammar, timeout_s=timeout_s)


def _run_case(case: dict, predict) -> dict:
    expected = _expected_intent(case)
    tool = expected["tool"]
    t0 = time.monotonic()
    raw = predict(case["prompt"])
    latency = time.monotonic() - t0
    result: dict = {
        "id": case["id"], "split": case["split"], "prompt": case["prompt"],
        "fixture": case["fixture"], "template_id": case.get("template_id"),
        "scorer": SCORER, "expected": expected, "raw_output": raw,
        "latency_s": round(latency, 2),
    }
    if expected != case["expected"]:
        result["frozen_expected"] = case["expected"]
    try:
        predicted = json.loads(raw)
        if not isinstance(predicted, dict) or "tool" not in predicted:
            raise ValueError("not a JSON object with a tool")
    except (json.JSONDecodeError, ValueError) as e:
        result.update(parse_ok=False, tool_correct=False, args_correct=False,
                      execution_ok=False, verification_pass=False, task_pass=False,
                      error=f"parse failed: {e}", retries=0)
        return result
    result["predicted"] = predicted
    result["parse_ok"] = True
    result["tool_correct"] = predicted.get("tool") == tool
    result["args_correct"] = _args_correct(expected, predicted)
    if predicted.get("tool") == "clarify":
        with tempfile.TemporaryDirectory() as workdir:
            r = edit(dict(predicted), Path(workdir))
        result["execution_ok"] = bool(r.get("ok"))
        result["verification_pass"] = False  # No media effect to verify.
        result["task_pass"] = result["execution_ok"] and result["args_correct"]
        result["retries"] = 0
        return result
    if tool == "clarify":
        result.update(execution_ok=False, verification_pass=False, task_pass=False,
                      retries=0, error="edit proposed instead of clarification")
        return result
    workdir = Path(tempfile.mkdtemp())
    try:
        shutil.copy(FIXTURES_DIR / case["fixture"] / "input.mp4", workdir / "clip.mp4")
        call = dict(predicted)
        call.setdefault("input", "clip.mp4")
        # The fixture runner stages only this file; never probe model-supplied paths.
        inputs = call.get("inputs")
        valid_concat = isinstance(inputs, list) and len(inputs) in (2, 3) and all(
            f == "clip.mp4" for f in inputs
        )
        if call["input"] != "clip.mp4" or (call.get("tool") == "concat" and not valid_concat):
            result.update(execution_ok=False, verification_pass=False, task_pass=False,
                          retries=0, error="input does not match staged fixture")
            return result
        r = edit(call, workdir)
        result["execution_ok"] = bool(r.get("ok"))
        result["verification_pass"] = bool(r.get("ok"))
        result["task_pass"] = result["verification_pass"] and result["args_correct"]
        result["retries"] = int(r.get("retries", 0))
        if not r.get("ok"):
            result["error"] = str(r.get("error", ""))[:300]
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return result


def _summarise(rows: list[dict]) -> dict:
    n = len(rows)
    def rate(key: str) -> float:
        return round(sum(1 for r in rows if r.get(key)) / n, 4) if n else 0.0
    failures: dict[str, int] = {}
    for r in rows:
        if not r.get("parse_ok"):
            failures["parse_fail"] = failures.get("parse_fail", 0) + 1
        elif not r.get("tool_correct"):
            failures["wrong_tool"] = failures.get("wrong_tool", 0) + 1
        elif not r.get("args_correct"):
            failures["wrong_arg"] = failures.get("wrong_arg", 0) + 1
        elif not r.get("execution_ok"):
            failures["execution_fail"] = failures.get("execution_fail", 0) + 1
        elif not r.get("task_pass"):
            failures["verification_fail"] = failures.get("verification_fail", 0) + 1
    per_tool: dict[str, dict] = {}
    for tool in sorted({r["expected"]["tool"] for r in rows}):
        sub = [r for r in rows if r["expected"]["tool"] == tool]
        per_tool[tool] = {
            "n": len(sub),
            "task_pass_rate": round(sum(1 for r in sub if r["task_pass"]) / len(sub), 4),
            "tool_correct_rate": round(sum(1 for r in sub if r["tool_correct"]) / len(sub), 4),
        }
    return {
        "n": n,
        "parse_ok_rate": rate("parse_ok"),
        "tool_correct_rate": rate("tool_correct"),
        "args_correct_rate": rate("args_correct"),
        "execution_ok_rate": rate("execution_ok"),
        "verification_pass_rate": rate("verification_pass"),
        "task_pass_rate": rate("task_pass"),
        "mean_latency_s": round(sum(r["latency_s"] for r in rows) / n, 2) if n else 0.0,
        "mean_retries": round(sum(r.get("retries", 0) for r in rows) / n, 2) if n else 0.0,
        "failure_modes": failures,
        "per_tool": per_tool,
    }


def _write_summary(out_dir: Path, label: str, split: str, summary: dict, backend: str) -> Path:
    lines = [f"# Eval: {label} — {split}", "",
             f"Backend: {backend} · scorer: {SCORER} · n={summary['n']} · "
             f"task pass rate: {summary['task_pass_rate']:.1%} · "
             f"wrapper media verification (all rows): {summary['verification_pass_rate']:.1%} · "
             f"tool accuracy: {summary['tool_correct_rate']:.1%} · "
             f"arg accuracy: {summary['args_correct_rate']:.1%} · "
             f"mean latency: {summary['mean_latency_s']}s · "
             f"mean retries: {summary['mean_retries']}", "",
             "## Per-tool task pass rate", ""]
    for tool, v in summary["per_tool"].items():
        lines.append(f"- {tool}: {v['task_pass_rate']:.1%} ({v['n']} cases, tool-correct {v['tool_correct_rate']:.1%})")
    lines += ["", "## Failure modes", ""]
    if summary["failure_modes"]:
        for k, v in sorted(summary["failure_modes"].items(), key=lambda kv: -kv[1]):
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- none")
    path = out_dir / f"{label}_{split}.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", help="eval JSONL file, e.g. eval/eval_seen.jsonl")
    ap.add_argument("--all", action="store_true", help="run all four frozen splits")
    ap.add_argument("--backend", choices=["oracle", "http"], default="oracle")
    ap.add_argument("--llm-url", default="http://127.0.0.1:8080",
                    help="llama.cpp server base URL (http backend)")
    ap.add_argument("--no-grammar", action="store_true")
    ap.add_argument("--label", default="sanity_intent_v2",
                    help="label for the output files")
    ap.add_argument("--out", default="eval/results")
    ap.add_argument("--limit", type=int, default=0, help="run only the first N cases")
    args = ap.parse_args()

    if args.label.startswith("baseline") and args.backend != "http":
        print(f"refusing: {args.label} must come from the real model (http backend), not oracle",
              file=sys.stderr)
        return 2
    if args.all:
        split_files = [EVAL_DIR / f"{s}.jsonl" for s in
                       ("eval_seen", "eval_unseen_paraphrase", "eval_unseen_combo", "eval_unseen_fixture")]
    elif args.split:
        split_files = [Path(args.split)]
    else:
        print("pass --split or --all", file=sys.stderr)
        return 2

    out_dir = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_file in split_files:
        for ext in (".jsonl", ".md"):
            path = out_dir / f"{args.label}_{split_file.stem}{ext}"
            if path.exists():
                ap.error(f"refusing to overwrite {path}; choose a new label or output directory")

    if args.backend == "http":
        def predict(prompt: str) -> str:
            return _http_predict(prompt, args.llm_url, use_grammar=not args.no_grammar)

    for split_file in split_files:
        cases = [json.loads(line) for line in split_file.read_text().splitlines() if line.strip()]
        if args.limit:
            cases = cases[: args.limit]
        rows = []
        for case in cases:
            if args.backend == "oracle":
                rows.append(_run_case(case, lambda p, c=case: json.dumps(_expected_intent(c))))
            else:
                rows.append(_run_case(case, predict))
            print(f"{case['id']}: pass={rows[-1]['task_pass']} "
                  f"tool={rows[-1].get('tool_correct')} args={rows[-1].get('args_correct')}")
        summary = _summarise(rows)
        stem = split_file.stem
        (out_dir / f"{args.label}_{stem}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows))
        md = _write_summary(out_dir, args.label, stem, summary, args.backend)
        print(f"{stem}: task success {summary['task_pass_rate']:.1%} "
              f"({sum(1 for r in rows if r['task_pass'])}/{len(rows)}) -> {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
