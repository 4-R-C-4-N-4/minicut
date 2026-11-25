"""Compare bounded MiniCut checkpoints on a matched private llama.cpp server."""
from __future__ import annotations

import argparse
import json
import os
import random
import socket
import subprocess
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from eval.run_eval import _args_correct, _http_predict, _run_case


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def collect(args, label, adapter, cases, fit):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    cmd = [args.server, "-m", str(args.gguf), "--jinja", "--host", "127.0.0.1",
           "--port", str(port), "-ngl", "all", "-c", "2048", "-np", "1"]
    if adapter:
        cmd += ["--lora", str(adapter)]
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def get(path):
        with opener.open(url + path, timeout=2) as reply:
            return json.load(reply)

    with (args.run / f"{label}.server.log").open("x") as log:
        process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(120):
                if process.poll() is not None:
                    raise RuntimeError(f"{label} server exited: {process.returncode}")
                try:
                    if get("/health")["status"] == "ok":
                        break
                except OSError:
                    time.sleep(0.25)
            else:
                raise RuntimeError(f"{label} server did not become ready")
            loaded = get("/lora-adapters")
            if adapter:
                assert len(loaded) == 1 and Path(loaded[0]["path"]).resolve() == adapter.resolve()
                assert loaded[0]["scale"] == 1
            else:
                assert not loaded
            write_json(args.run / f"{label}.server.json", {"argv": cmd, "adapters": loaded})
            with (args.run / f"{label}.predictions.jsonl").open("x", buffering=1) as out:
                for scope, roster in (("fit", fit), ("dev", cases)):
                    for case in roster:
                        start = time.monotonic()
                        raw = _http_predict(case["prompt"], url, True, timeout_s=20)
                        out.write(json.dumps({"scope": scope, "id": case["id"], "raw": raw,
                                              "seconds": time.monotonic() - start}) + "\n")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def fit_roster(data=Path("training/train_pilot_v1.jsonl")):
    groups = defaultdict(list)
    for i, row in enumerate(read_jsonl(data), 1):
        messages = row["messages"]
        expected = json.loads(messages[-1]["content"])
        groups[expected["tool"]].append({"id": f"train-{i}", "prompt": messages[-2]["content"],
                                         "expected": expected})
    rng = random.Random(42)
    return [row for tool in sorted(groups) for row in rng.sample(groups[tool], 4)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--server", required=True)
    parser.add_argument("--data", type=Path, default=Path("training/train_pilot_v1.jsonl"),
                        help="training rows the fit probe samples from")
    args = parser.parse_args()
    assert os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID"
    devices = subprocess.check_output([args.server, "--list-devices"], text=True)
    assert devices.count("CUDA") == 1 and "RTX 3090" in devices, devices
    cases = [r for path in sorted(Path("eval").glob("eval_*.jsonl")) for r in read_jsonl(path)]
    assert len(cases) == len({r["id"] for r in cases}) == 200
    fit = fit_roster(args.data)
    write_json(args.run / "fit-roster.json", fit)
    arms = [("stock", None, cases), ("early", args.run / "early.gguf", []),
            ("final", args.run / "final.gguf", cases)]
    urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))
    for label, adapter, dev in arms:
        collect(args, label, adapter, dev, fit)
        print(f"captured {label}: {len(dev)} dev and {len(fit)} fit", flush=True)
    dev_results = {}
    fit_counts = {}
    for label, _, dev in arms:
        predictions = read_jsonl(args.run / f"{label}.predictions.jsonl")
        raw = {r["id"]: r for r in predictions}
        assert len(raw) == len(predictions) and set(raw) == {r["id"] for r in dev + fit}
        counts = Counter()
        for case in fit:
            expected = case["expected"]
            tool = expected["tool"]
            counts[(tool, "total")] += 1
            try:
                pred = json.loads(raw[case["id"]]["raw"])
            except json.JSONDecodeError:
                continue
            if not isinstance(pred, dict) or not _args_correct(expected, pred):
                continue
            if tool not in {"concat", "clarify"} and pred.get("input") != expected.get("input"):
                continue
            counts[(tool, "pass")] += 1
        fit_counts[label] = {t: [counts[(t, "pass")], counts[(t, "total")]]
                             for t in sorted({r["expected"]["tool"] for r in fit})}
        if not dev:
            continue
        graded = []
        with (args.run / f"{label}.graded.jsonl").open("x", buffering=1) as out:
            for case in dev:
                pred = raw[case["id"]]
                row = _run_case(case, lambda _, p=pred: p["raw"])
                row["latency_s"] = pred["seconds"]
                graded.append(row)
                out.write(json.dumps(row) + "\n")
        dev_results[label] = graded
    stock, final = dev_results["stock"], dev_results["final"]
    pairs = list(zip(stock, final, strict=True))
    assert all(a["id"] == b["id"] for a, b in pairs)

    def inversion(row):
        return {row["expected"]["tool"], row.get("predicted", {}).get("tool")} == {"trim", "cut_out"}

    def comparison(pairs):
        return {"n": len(pairs), "stock_pass": sum(a["task_pass"] for a, _ in pairs),
                "final_pass": sum(b["task_pass"] for _, b in pairs),
                "gains": [a["id"] for a, b in pairs if not a["task_pass"] and b["task_pass"]],
                "regressions": [a["id"] for a, b in pairs if a["task_pass"] and not b["task_pass"]]}

    report = {"all": comparison(pairs), "by_split": {s: comparison([(a, b) for a, b in pairs
               if a["split"] == s]) for s in sorted({r["split"] for r in stock})},
              "keep_remove": comparison([(a, b) for a, b in pairs if a["expected"]["tool"]
                                         in {"trim", "cut_out"}]),
              "inversions": {"stock": sum(map(inversion, stock)), "final": sum(map(inversion, final)),
                             "new": [b["id"] for a, b in pairs if not inversion(a) and inversion(b)]},
              "unnecessary_clarifications": {label: sum(r["expected"]["tool"] != "clarify"
               and r.get("predicted", {}).get("tool") == "clarify" for r in rows)
               for label, rows in dev_results.items()}, "fit": fit_counts}
    report["gate_pass"] = (report["all"]["final_pass"] > report["all"]["stock_pass"]
                           and report["keep_remove"]["final_pass"] > report["keep_remove"]["stock_pass"]
                           and not report["inversions"]["new"])
    write_json(args.run / "comparison.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
