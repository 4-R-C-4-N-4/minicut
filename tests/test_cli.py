"""CLI tests: request -> model call -> preview or verified derivative.

The model is either a fake llama.cpp server or a fixed --call, so these
run on CPU without weights. Requires factory fixtures (see test_wrapper).
"""
import json
import shutil
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minicut import cli  # noqa: E402
from minicut.llm.prompts import edit_messages  # noqa: E402

FIXTURE = ROOT / "factory" / "fixtures" / "base_30s" / "input.mp4"
TRIM = {"tool": "trim", "input": "clip.mp4", "start": 2.0, "end": 8.0}


class FakeModel(BaseHTTPRequestHandler):
    reply = TRIM
    seen: list = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeModel.seen.append(body)
        data = json.dumps({"choices": [{"message": {"content": json.dumps(FakeModel.reply)}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


class TestCli(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.clip = self.tmp / "clip.mp4"
        shutil.copy(FIXTURE, self.clip)
        self.digest = sha256(self.clip.read_bytes()).hexdigest()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_cli(self, *argv):
        out = StringIO()
        with redirect_stdout(out):
            code = cli.main([*argv, "--json"])
        return code, json.loads(out.getvalue())

    def files(self, folder=None):
        return {p.name for p in (folder or self.tmp).iterdir()}

    def assert_source_unchanged(self):
        self.assertEqual(sha256(self.clip.read_bytes()).hexdigest(), self.digest)

    def test_edit_writes_one_verified_derivative(self):
        code, out = self.run_cli("keep 2 to 8 seconds", str(self.clip), "--call", json.dumps(TRIM))
        self.assertEqual(code, 0, out)
        self.assertEqual(out["status"], "done")
        self.assertEqual(Path(out["output"]).name, "clip_trim.mp4")
        self.assertIn("keep 2.0-8.0s", out["plan"])
        self.assertEqual(self.files(), {"clip.mp4", "clip_trim.mp4"})
        self.assert_source_unchanged()

    def test_preview_writes_nothing(self):
        call = {"tool": "cut_out", "input": "clip.mp4", "start": 2.0, "end": 8.0}
        code, out = self.run_cli("remove 2 to 8 seconds", str(self.clip), "--preview",
                                 "--call", json.dumps(call))
        self.assertEqual(code, 0, out)
        self.assertEqual(out["status"], "preview")
        self.assertIn("remove 2.0-8.0s", out["plan"])
        self.assertEqual(self.files(), {"clip.mp4"})

    def test_output_dir(self):
        dest = self.tmp / "renders"
        code, out = self.run_cli("keep 2 to 8 seconds", str(self.clip), "--output-dir", str(dest),
                                 "--call", json.dumps(TRIM))
        self.assertEqual(code, 0, out)
        self.assertEqual(self.files(dest), {"clip_trim.mp4"})
        self.assertEqual(self.files(), {"clip.mp4", "renders"})

    def test_app_owns_single_file_selection(self):
        call = dict(TRIM, input="something_else.mp4", output="clip.mp4")
        code, out = self.run_cli("keep 2 to 8 seconds", str(self.clip), "--call", json.dumps(call))
        self.assertEqual(code, 0, out)
        self.assertEqual(out["call"]["input"], "clip.mp4")
        self.assertNotIn("output", out["call"])
        self.assert_source_unchanged()

    def test_concat_cannot_name_unlisted_files(self):
        call = {"tool": "concat", "inputs": ["clip.mp4", "other.mp4"]}
        code, out = self.run_cli("join them", str(self.clip), "--call", json.dumps(call))
        self.assertEqual(code, 2, out)
        self.assertEqual(self.files(), {"clip.mp4"})

    def test_invalid_edit_fails_without_writing(self):
        call = dict(TRIM, end=99.0)
        code, out = self.run_cli("keep 2 to 99 seconds", str(self.clip), "--call", json.dumps(call))
        self.assertEqual(code, 1, out)
        self.assertEqual(out["status"], "rejected")
        self.assertEqual(self.files(), {"clip.mp4"})

    def test_inputs_must_share_a_directory(self):
        other = self.tmp / "sub"
        other.mkdir()
        shutil.copy(FIXTURE, other / "b.mp4")
        code, out = self.run_cli("join them", str(self.clip), str(other / "b.mp4"),
                                 "--call", json.dumps({"tool": "concat", "inputs": ["clip.mp4", "b.mp4"]}))
        self.assertEqual(code, 2, out)

    def test_model_request_and_clarification(self):
        server = HTTPServer(("127.0.0.1", 0), FakeModel)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            FakeModel.seen.clear()
            FakeModel.reply = TRIM
            code, out = self.run_cli("keep 2 to 8 seconds", str(self.clip), "--llm-url", url)
            self.assertEqual(code, 0, out)
            body = FakeModel.seen[-1]
            self.assertEqual(body["messages"], edit_messages("keep 2 to 8 seconds in clip.mp4"))
            self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
            self.assertEqual(body["temperature"], 0.0)
            self.assertIn("grammar", body)

            FakeModel.reply = {"tool": "clarify", "question": "Which part?", "missing": ["start", "end"]}
            before = self.files()
            code, out = self.run_cli("shorten clip.mp4", str(self.clip), "--llm-url", url)
            self.assertEqual(code, 3, out)
            self.assertEqual(out["status"], "clarify")
            self.assertEqual(out["question"], "Which part?")
            self.assertEqual(FakeModel.seen[-1]["messages"][-1]["content"], "shorten clip.mp4")
            self.assertEqual(self.files(), before)
        finally:
            server.shutdown()
            server.server_close()

    def test_unreachable_model_is_reported(self):
        code, out = self.run_cli("keep 2 to 8 seconds", str(self.clip), "--llm-url", "http://127.0.0.1:9")
        self.assertEqual(code, 2, out)
        self.assertEqual(out["status"], "error")
        self.assertEqual(self.files(), {"clip.mp4"})


if __name__ == "__main__":
    unittest.main()


class TestHelp(unittest.TestCase):
    def test_help_shows_the_phrasebook(self):
        out = StringIO()
        with redirect_stdout(out), self.assertRaises(SystemExit) as done:
            cli.main(["--help"])
        self.assertEqual(done.exception.code, 0)
        text = out.getvalue()
        self.assertIn("How to ask (the MiniCut phrasebook)", text)
        self.assertIn("cut the first 10 seconds off clip.mp4", text)
        self.assertIn('"cut ... of" and "cut ... off" mean the same', text)
