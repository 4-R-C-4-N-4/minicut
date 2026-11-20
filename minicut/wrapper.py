"""Orchestration: probe -> validate -> build -> run -> verify -> retry.

The model never edits a command. On failure the wrapper feeds the
probe + stderr back to the model (via `llm` callable) for a corrected
JSON call, max 2 retries, then reports the error plainly.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Callable

from . import build as build_mod
from .probe import probe_file
from .run import run
from .validate import validate
from .verify import verify_output

MAX_RETRIES = 2


def _resolve_output(tool_call: dict, workdir: Path) -> Path:
    name = tool_call.get("output") or build_mod.default_output(tool_call)
    return workdir / name


def edit(tool_call: dict, workdir: str | Path,
         llm: Callable[[dict], dict] | None = None) -> dict:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    tool = tool_call.get("tool")

    if tool == "clarify":
        errors = validate(tool_call, None, workdir)
        if errors:
            return {"ok": False, "error": "; ".join(errors)}
        return {"ok": True, "clarify": tool_call.get("question"),
                "missing": tool_call.get("missing", [])}

    # Probe the primary input (concat probes all inputs).
    try:
        if tool == "concat":
            probes = {f: probe_file(workdir / f) for f in tool_call.get("inputs", [])}
            probe = next(iter(probes.values())) if probes else None
        else:
            probe = probe_file(workdir / tool_call["input"])
            probes = {tool_call["input"]: probe}
    except (FileNotFoundError, RuntimeError) as e:
        return {"ok": False, "error": str(e)}

    errors = validate(tool_call, probe, workdir)
    if errors:
        return {"ok": False, "error": "; ".join(errors)}

    call = dict(tool_call)
    attempt = 0
    while True:
        # Concat needs a demuxer list file.
        list_file = None
        if call["tool"] == "concat":
            with tempfile.NamedTemporaryFile("w", suffix=".txt", dir=workdir,
                                            delete=False) as fh:
                for f in call["inputs"]:
                    fh.write(f"file '{workdir / f}'\n")
                list_file = fh.name
            call = dict(call, _list_file=list_file)
        try:
            argv = build_mod.build(call, probe, workdir)
            result = run(argv)
        finally:
            if list_file:
                Path(list_file).unlink(missing_ok=True)
        out_path = _resolve_output(call, workdir)
        if result.exit_code != 0:
            err = result.stderr.strip()[-800:]
            if llm and attempt < MAX_RETRIES:
                attempt += 1
                call = llm({"failed_call": call, "stderr": err,
                            "probe": {"duration_s": probe.duration_s,
                                      "width": probe.width, "height": probe.height,
                                      "has_audio": probe.has_audio}})
                continue
            return {"ok": False, "error": f"ffmpeg failed (exit {result.exit_code}): {err}",
                    "argv": argv}
        passed, reason = verify_output(call, out_path, probe, probes)
        if not passed:
            if llm and attempt < MAX_RETRIES:
                attempt += 1
                call = llm({"failed_call": call, "verification": reason,
                            "probe": {"duration_s": probe.duration_s,
                                      "width": probe.width, "height": probe.height,
                                      "has_audio": probe.has_audio}})
                continue
            return {"ok": False, "error": f"verification failed: {reason}",
                    "output": str(out_path)}
        return {"ok": True, "output": str(out_path), "verification": reason,
                "retries": attempt}
