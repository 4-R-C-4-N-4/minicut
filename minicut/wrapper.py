"""Orchestration: probe -> validate -> build -> run -> verify -> retry.

The model never edits a command. On failure the wrapper feeds the
probe + stderr back to the model (via `llm` callable) for a corrected
JSON call, max 2 retries, then reports the error plainly.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from subprocess import TimeoutExpired
from typing import Callable

from . import build as build_mod
from .probe import probe_file
from .run import run
from .validate import validate
from .verify import verify_output

MAX_RETRIES = 2


def _publish(staged: Path, dest: Path) -> Path:
    """Publish on the same filesystem without copying or replacing any entry."""
    out = dest / staged.name
    n = 2
    while True:
        try:
            os.link(staged, out)
        except FileExistsError:
            out = dest / f"{staged.stem}_v{n}{staged.suffix}"
            n += 1
        else:
            return out


def edit(tool_call: dict, workdir: str | Path,
         llm: Callable[[dict], dict] | None = None, *,
         output_dir: str | Path | None = None) -> dict:
    """Render a new derivative; only the application may choose output_dir."""
    workdir = Path(workdir)
    dest = Path(output_dir) if output_dir is not None else workdir
    call = tool_call
    for attempt in range(MAX_RETRIES + 1):
        if not isinstance(call, dict):
            return {"ok": False, "error": "tool call must be a JSON object"}
        call = {k: v for k, v in call.items() if k != "output"}
        errors = validate(call, None, workdir)
        if errors:
            return {"ok": False, "error": "; ".join(errors)}
        tool = call["tool"]
        if tool == "clarify":
            return {"ok": True, "clarify": call["question"], "missing": call.get("missing", [])}
        files = call["inputs"] if tool == "concat" else [call["input"]]
        try:
            root = workdir.resolve()
            paths = {f: (root / f).resolve(strict=True) for f in files}
            if any(not p.is_relative_to(root) or not p.is_file() for p in paths.values()):
                return {"ok": False, "error": "inputs must be regular files inside the workspace"}
            probes = {f: probe_file(p) for f, p in paths.items()}
            probe = probes[files[0]]
            errors = validate(call, probe, workdir)
            if errors:
                return {"ok": False, "error": "; ".join(errors)}
            dest.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".minicut-", dir=dest) as staging:
                staged = Path(staging).absolute() / build_mod.default_output(call)
                render_call = dict(call, output=str(staged))
                if tool != "concat":
                    render_call["input"] = str(paths[files[0]])
                if tool == "concat":
                    listing = staged.parent / "inputs.ffconcat"
                    entries = [str(paths[f]) for f in files]
                    if any("\n" in p or "\r" in p for p in entries):
                        return {"ok": False, "error": "concat paths must not contain line breaks"}
                    quoted = [p.replace("'", "'\\''") for p in entries]
                    listing.write_text("".join(f"file '{p}'\n" for p in quoted))
                    render_call["_list_file"] = str(listing)
                argv = build_mod.build(render_call, probe, workdir)
                result = run(argv)
                if result.exit_code != 0:
                    detail = {"stderr": result.stderr.strip()[-800:]}
                    error = f"ffmpeg failed (exit {result.exit_code}): {detail['stderr']}"
                else:
                    passed, reason = verify_output(call, staged, probe, probes)
                    if passed:
                        chain = probe_file(staged).chain
                        output = _publish(staged, dest)
                        return {"ok": True, "output": str(output), "verification": reason,
                                "chain": chain, "retries": attempt}
                    detail = {"verification": reason}
                    error = f"verification failed: {reason}"
        except TimeoutExpired:
            return {"ok": False, "error": "media command timed out"}
        except (OSError, RuntimeError, ValueError) as exc:
            return {"ok": False, "error": str(exc)}
        if llm is None or attempt == MAX_RETRIES:
            return {"ok": False, "error": error}
        call = llm({"failed_call": call, **detail,
                    "probe": {"duration_s": probe.duration_s, "width": probe.width,
                              "height": probe.height, "has_audio": probe.has_audio}})
    raise AssertionError("retry loop exhausted without a result")
