"""MiniCut Phase 3 eval harness.

Takes a frozen eval JSONL file, sends each prompt to a model (or the
oracle, for a sanity run), feeds the returned JSON to the wrapper, and
records tool accuracy, arg accuracy, execution success, verification
pass rate, retry count and latency.

Backends:
  oracle — returns the expected JSON. Sanity mode ONLY: it proves the
           harness and the eval cases execute end to end. An oracle
           run must NEVER be recorded as a baseline (the harness
           refuses to label it baseline_v1).
  http   — POSTs to a llama.cpp server's OpenAI-compatible endpoint
           (default http://127.0.0.1:8080/v1/chat/completions) with the
           system prompt + few-shot examples from minicut/llm/prompts.py,
           temperature 0 for a reproducible baseline, and the GBNF
           grammar passed in the request body. The llama.cpp server
           honours a top-level "grammar" field; if your server version
           ignores it, pass --no-grammar and the run log will say so.
           This path is NOT tested against a live server here — no
           model weights or llama.cpp are installed in this workspace —
           so the first real baseline run is the moment to confirm the
           server accepts the grammar field.

Scoring, per case:
  parse_ok          — model output parsed as a JSON object with a tool
  tool_correct      — predicted tool == expected tool
  args_correct      — every expected arg equal (input/output ignored:
                      the wrapper derives output; clarify compares the
                      missing-arg set, not the question wording)
  execution_ok      — wrapper edit() returned ok (clarify: accepted)
  verification_pass — edit() ok and, for edits, the verifier passed
  retries / latency_s — from the wrapper result and wall clock

Usage:
    # Sanity: prove harness + cases execute (NOT a baseline)
    python -m eval.run_eval --split eval/eval_seen.jsonl --backend oracle --out eval/results
    # Real baseline (needs llama-server running with MiniCPM5-2B GGUF)
    python -m eval.run_eval --split eval/eval_seen.jsonl --backend http --label baseline_v1 --out eval/results
    # All four splits, one label
    python -m eval.run_eval --all --backend http --label baseline_v1 --out eval/results
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
import urllib.request
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minicut.llm.prompts import FEW_SHOT, SYSTEM_PROMPT  # noqa: E402
from minicut.wrapper import edit  # noqa: E402

EVAL_DIR = ROOT / "eval"
FIXTURES_DIR = ROOT / "factory" / "fixtures"
GRAMMAR_PATH = ROOT / "minicut" / "grammar" / "minicut.gbnf"

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
        return set(expected.get("missing", [])) == set(predicted.get("missing", []))
    for key in COMPARE_ARGS.get(expected["tool"], []):
        exp_v, pred_v = expected.get(key), predicted.get(key)
        if key == "crf":
            exp_v = exp_v if exp_v is not None else 23
            pred_v = pred_v if pred_v is not None else 23
        if isinstance(exp_v, float) or isinstance(pred_v, float):
            try:
                if abs(float(exp_v) - float(pred_v)) > 1e-6:
                    return False
            except (TypeError, ValueError):
                return False
        elif exp_v != pred_v:
            return False
    return True


def _http_predict(prompt: str, base_url: str, use_grammar: bool, timeout_s: int = 120) -> str:
    """One chat completion from a llama.cpp server. Returns raw content."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user, assistant in FEW_SHOT:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": prompt})
    body: dict = {
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 256,
        "stream": False,
    }
    if use_grammar:
        body["grammar"] = GRAMMAR_PATH.read_text()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"llama.cpp server HTTP {e.code}: {e.read().decode()[:300]}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"cannot reach llama.cpp server at {base_url}: {e.reason}")
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(f"unexpected server response: {json.dumps(data)[:300]}")


def _run_case(case: dict, predict) -> dict:
    expected = case["expected"]
    tool = expected["tool"]
    t0 = time.monotonic()
    raw = predict(case["prompt"])
    latency = time.monotonic() - t0
    result: dict = {
        "id": case["id"], "split": case["split"], "prompt": case["prompt"],
        "fixture": case["fixture"], "template_id": case.get("template_id"),
        "expected": expected, "raw_output": raw, "latency_s": round(latency, 2),
    }
    try:
        predicted = json.loads(raw)
        if not isinstance(predicted, dict) or "tool" not in predicted:
            raise ValueError("not a JSON object with a tool")
    except (json.JSONDecodeError, ValueError) as e:
        result.update(parse_ok=False, tool_correct=False, args_correct=False,
                      execution_ok=False, verification_pass=False,
                      error=f"parse failed: {e}", retries=0)
        return result
    result["predicted"] = predicted
    result["parse_ok"] = True
    result["tool_correct"] = predicted.get("tool") == tool
    result["args_correct"] = _args_correct(expected, predicted)
    if tool == "clarify":
        # No ffmpeg involved: correct iff tool + missing set match and
        # the wrapper accepts the clarify call.
        r = edit(dict(predicted), Path(tempfile.mkdtemp()))
        result["execution_ok"] = bool(r.get("ok"))
        result["verification_pass"] = bool(r.get("ok")) and result["args_correct"]
        result["retries"] = 0
        return result
    workdir = Path(tempfile.mkdtemp())
    try:
        shutil.copy(FIXTURES_DIR / case["fixture"] / "input.mp4", workdir / "clip.mp4")
        call = dict(predicted)
        call.setdefault("input", "clip.mp4")
        r = edit(call, workdir)
        result["execution_ok"] = bool(r.get("ok"))
        result["verification_pass"] = bool(r.get("ok"))
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
        elif not r.get("verification_pass"):
            failures["verification_fail"] = failures.get("verification_fail", 0) + 1
    per_tool: dict[str, dict] = {}
    for tool in sorted({r["expected"]["tool"] for r in rows}):
        sub = [r for r in rows if r["expected"]["tool"] == tool]
        per_tool[tool] = {
            "n": len(sub),
            "verification_pass_rate": round(sum(1 for r in sub if r["verification_pass"]) / len(sub), 4),
            "tool_correct_rate": round(sum(1 for r in sub if r["tool_correct"]) / len(sub), 4),
        }
    return {
        "n": n,
        "parse_ok_rate": rate("parse_ok"),
        "tool_correct_rate": rate("tool_correct"),
        "args_correct_rate": rate("args_correct"),
        "execution_ok_rate": rate("execution_ok"),
        "verification_pass_rate": rate("verification_pass"),
        "mean_latency_s": round(sum(r["latency_s"] for r in rows) / n, 2) if n else 0.0,
        "mean_retries": round(sum(r.get("retries", 0) for r in rows) / n, 2) if n else 0.0,
        "failure_modes": failures,
        "per_tool": per_tool,
    }


def _write_summary(out_dir: Path, label: str, split: str, summary: dict, backend: str) -> Path:
    lines = [f"# Eval: {label} — {split}", "",
             f"Backend: {backend} · n={summary['n']} · "
             f"verification pass rate: {summary['verification_pass_rate']:.1%} · "
             f"tool accuracy: {summary['tool_correct_rate']:.1%} · "
             f"arg accuracy: {summary['args_correct_rate']:.1%} · "
             f"mean latency: {summary['mean_latency_s']}s · "
             f"mean retries: {summary['mean_retries']}", "",
             "## Per-tool verification pass rate", ""]
    for tool, v in summary["per_tool"].items():
        lines.append(f"- {tool}: {v['verification_pass_rate']:.1%} ({v['n']} cases, tool-correct {v['tool_correct_rate']:.1%})")
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
    ap.add_argument("--label", default="sanity_oracle",
                    help="label for the output files")
    ap.add_argument("--out", default="eval/results")
    ap.add_argument("--limit", type=int, default=0, help="run only the first N cases")
    args = ap.parse_args()

    if args.label == "baseline_v1" and args.backend != "http":
        print("refusing: baseline_v1 must come from the real model (http backend), not oracle",
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
                rows.append(_run_case(case, lambda p, c=case: json.dumps(c["expected"])))
            else:
                rows.append(_run_case(case, predict))
            print(f"{case['id']}: pass={rows[-1]['verification_pass']} "
                  f"tool={rows[-1].get('tool_correct')} args={rows[-1].get('args_correct')}")
        summary = _summarise(rows)
        stem = split_file.stem
        (out_dir / f"{args.label}_{stem}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows))
        md = _write_summary(out_dir, args.label, stem, summary, args.backend)
        print(f"{stem}: verification {summary['verification_pass_rate']:.1%} "
              f"({sum(1 for r in rows if r['verification_pass'])}/{len(rows)}) -> {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
