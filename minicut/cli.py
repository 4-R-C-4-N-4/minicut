"""MiniCut command line: one plain-English edit -> one new verified file.

    minicut "keep 2 to 8 seconds" clip.mp4
    minicut "join a.mp4 then b.mp4" a.mp4 b.mp4 --output-dir renders/
    minicut "remove the first 10 seconds" clip.mp4 --preview

The local model (llama.cpp, MiniCPM5-2B) turns the request into one JSON
tool call. The application chooses the files, validates the call against
ffprobe, and the wrapper renders, verifies and publishes a new file.
Sources are never modified. --preview shows the plan and writes nothing.

Exit codes: 0 done or previewed, 1 edit rejected or failed,
2 usage, model or file-selection error, 3 the model asked a question.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .llm.client import predict
from .probe import probe_file
from .validate import validate
from .wrapper import edit

DEFAULT_LLM_URL = "http://127.0.0.1:8080"


class CliError(Exception):
    """Usage, model or file-selection problem (exit 2)."""


def _secs(v: float) -> str:
    return f"{float(v):.1f}"


def describe(call: dict, probes: dict) -> str:
    """Plain-language description of what this call does to the sources."""
    tool = call["tool"]
    if tool == "concat":
        total = sum(probes[f].duration_s for f in call["inputs"])
        return f"join {' then '.join(call['inputs'])} into one {total:.1f}s video"
    name = call["input"]
    p = probes[name]
    if tool == "trim":
        s, e = call["start"], call["end"]
        return f"keep {_secs(s)}-{_secs(e)}s of {name} ({e - s:.1f}s of {p.duration_s:.1f}s) and drop the rest"
    if tool == "cut_out":
        s, e = call["start"], call["end"]
        return (f"remove {_secs(s)}-{_secs(e)}s from {name} and keep the other "
                f"{p.duration_s - (e - s):.1f}s")
    if tool == "scale":
        return f"resize {name} from {p.width}x{p.height} to {call['width']}x{call['height']}"
    if tool == "speed":
        f = float(call["factor"])
        return f"play {name} at {f:g}x speed ({p.duration_s:.1f}s -> {p.duration_s / f:.1f}s)"
    if tool == "convert":
        return f"re-encode {name} as {call['codec']} in .{call['container']} (crf {call.get('crf', 23)})"
    if tool == "extract_audio":
        return f"save the audio of {name} as {call['format']}"
    if tool == "mute":
        return f"remove the audio track from {name} and keep the picture"
    return tool


def compose_prompt(request: str, names: list[str]) -> str:
    """Name the files for the model when the request does not."""
    if any(n in request for n in names):
        return request
    if len(names) == 1:
        return f"{request} in {names[0]}"
    return f"{request} (files: {', '.join(names)})"


def select_files(paths: list[str]) -> tuple[Path, list[str]]:
    resolved = [Path(p).expanduser().resolve() for p in paths]
    for p in resolved:
        if not p.is_file():
            raise CliError(f"not a file: {p}")
    folders = {p.parent for p in resolved}
    if len(folders) != 1:
        raise CliError("all input files must be in the same folder")
    names = [p.name for p in resolved]
    return folders.pop(), list(dict.fromkeys(names))


def bind_files(call: dict, names: list[str]) -> dict:
    """Apply the application's file choice; the model never picks paths."""
    call = {k: v for k, v in call.items() if k != "output"}
    tool = call.get("tool")
    if tool == "clarify":
        return call
    if tool == "concat":
        inputs = call.get("inputs")
        if not isinstance(inputs, list) or any(f not in names for f in inputs):
            raise CliError(f"model joined files that were not given: {inputs!r}")
        return call
    if len(names) != 1:
        raise CliError(f"'{tool}' edits one file; give exactly one input")
    return dict(call, input=names[0])


def run(args) -> tuple[int, dict]:
    workdir, names = select_files(args.files)
    out: dict = {"request": args.request, "files": names}
    if args.call:
        raw = args.call
    else:
        prompt = compose_prompt(args.request, names)
        out["prompt"] = prompt
        try:
            raw = predict(prompt, args.llm_url, use_grammar=not args.no_grammar)
        except RuntimeError as e:
            raise CliError(str(e)) from e
    out["raw"] = raw
    try:
        call = json.loads(raw)
    except json.JSONDecodeError as e:
        raise CliError(f"model output is not JSON: {raw[:200]!r}") from e
    if not isinstance(call, dict):
        raise CliError("model output is not a JSON object")
    call = bind_files(call, names)
    out["call"] = call
    errors = validate(call, None, workdir)
    if errors:
        return 1, dict(out, status="rejected", error="; ".join(errors))
    if call["tool"] == "clarify":
        return 3, dict(out, status="clarify", question=call["question"],
                       missing=call.get("missing", []))
    inputs = call["inputs"] if call["tool"] == "concat" else [call["input"]]
    probes = {name: probe_file(workdir / name) for name in set(inputs)}
    errors = validate(call, probes[inputs[0]], workdir)
    if errors:
        return 1, dict(out, status="rejected", error="; ".join(errors))
    destination = Path(args.output_dir).expanduser().resolve() if args.output_dir else workdir
    out.update(plan=describe(call, probes), destination=str(destination))
    if args.preview:
        return 0, dict(out, status="preview")
    result = edit(call, workdir, output_dir=destination)
    if not result.get("ok"):
        return 1, dict(out, status="failed", error=result.get("error", "unknown error"))
    return 0, dict(out, status="done", output=result["output"],
                   verification=result.get("verification"), chain=result.get("chain"))


def _print_human(code: int, out: dict) -> None:
    status = out.get("status")
    if status == "error":
        print(f"error: {out['error']}", file=sys.stderr)
    elif status == "clarify":
        print(f"MiniCut needs more detail: {out['question']}")
    elif status in ("rejected", "failed"):
        if out.get("plan"):
            print(f"plan: {out['plan']}")
        print(f"not done: {out['error']}", file=sys.stderr)
    elif status == "preview":
        print(f"plan: {out['plan']}")
        print(f"would write a new file in {out['destination']}; sources are not modified")
    else:
        print(f"done: {out['plan']}")
        print(f"wrote {out['output']}")
        print(f"verified: {out['verification']}")
        print("sources were not modified")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="minicut", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("request", help="what to do, in plain English")
    ap.add_argument("files", nargs="+", help="input video file(s), all in one folder")
    ap.add_argument("--preview", action="store_true", help="show the plan; write nothing")
    ap.add_argument("--output-dir", help="where to write the new file (default: next to the input)")
    ap.add_argument("--llm-url", default=os.environ.get("MINICUT_LLM_URL", DEFAULT_LLM_URL),
                    help=f"llama.cpp server (default: $MINICUT_LLM_URL or {DEFAULT_LLM_URL})")
    ap.add_argument("--no-grammar", action="store_true", help="do not send the GBNF grammar")
    ap.add_argument("--call", help="use this JSON tool call instead of asking the model")
    ap.add_argument("--json", action="store_true", help="print a JSON result")
    args = ap.parse_args(argv)
    try:
        code, out = run(args)
    except CliError as e:
        code, out = 2, {"request": args.request, "status": "error", "error": str(e)}
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        _print_human(code, out)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
