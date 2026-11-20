"""Tests for the Phase 3 eval harness (eval/run_eval.py).

Covers: arg comparison (incl. crf default), oracle scoring on real
generated cases, the http client against a fake llama.cpp server,
and the guard that refuses to label an oracle run as the baseline.
Run: python -m unittest discover -s tests -v
"""
import json
import subprocess
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.run_eval import _args_correct, _http_predict, _run_case  # noqa: E402


class TestArgsCorrect(unittest.TestCase):
    def test_trim_exact(self):
        exp = {"tool": "trim", "input": "clip.mp4", "start": 12.0, "end": 27.0}
        self.assertTrue(_args_correct(exp, dict(exp)))
        self.assertFalse(_args_correct(exp, {**exp, "end": 26.0}))

    def test_convert_crf_defaults_to_23(self):
        exp = {"tool": "convert", "input": "clip.mp4", "codec": "libx264",
               "container": "mp4", "crf": 23}
        self.assertTrue(_args_correct(exp, {k: v for k, v in exp.items() if k != "crf"}))

    def test_clarify_compares_missing_set_not_question(self):
        exp = {"tool": "clarify", "question": "What times?", "missing": ["start", "end"]}
        pred = {"tool": "clarify", "question": "Something else entirely", "missing": ["end", "start"]}
        self.assertTrue(_args_correct(exp, pred))
        self.assertFalse(_args_correct(exp, {**pred, "missing": ["start"]}))


class TestOracleScoring(unittest.TestCase):
    def test_generated_cases_execute(self):
        rows = [json.loads(l) for l in
                (ROOT / "eval" / "eval_seen.jsonl").read_text().splitlines()]
        trim = next(r for r in rows if r["expected"]["tool"] == "trim")
        clarify = next(r for r in rows if r["expected"]["tool"] == "clarify")
        for case in (trim, clarify):
            out = _run_case(case, lambda p, c=case: json.dumps(c["expected"]))
            self.assertTrue(out["parse_ok"], out)
            self.assertTrue(out["tool_correct"], out)
            self.assertTrue(out["args_correct"], out)
            self.assertTrue(out["verification_pass"], out)


class FakeLlamaHandler(BaseHTTPRequestHandler):
    payload = {"choices": [{"message": {"content": '{"tool": "mute", "input": "clip.mp4"}'}}]}

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        self.server.last_body = json.loads(self.rfile.read(length) or b"{}")
        data = json.dumps(self.payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):  # silence test output
        pass


class TestHttpClient(unittest.TestCase):
    def test_chat_completions_round_trip(self):
        server = HTTPServer(("127.0.0.1", 0), FakeLlamaHandler)
        threading.Thread(target=server.handle_request, daemon=True).start()
        url = f"http://127.0.0.1:{server.server_address[1]}"
        out = _http_predict("mute clip.mp4", url, use_grammar=True)
        self.assertEqual(json.loads(out)["tool"], "mute")
        body = server.last_body
        self.assertEqual(body["temperature"], 0.0)
        self.assertIn("grammar", body)
        self.assertEqual(body["messages"][0]["role"], "system")
        server.server_close()


class TestBaselineGuard(unittest.TestCase):
    def test_oracle_cannot_be_labelled_baseline(self):
        r = subprocess.run(
            [sys.executable, "-m", "eval.run_eval", "--split", "eval/eval_seen.jsonl",
             "--backend", "oracle", "--label", "baseline_v1", "--out", "eval/results",
             "--limit", "1"],
            cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)
        self.assertIn("baseline_v1", r.stderr)


if __name__ == "__main__":
    unittest.main()
