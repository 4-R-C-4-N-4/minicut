"""Run ffmpeg argv. No shell, capture output, timeout."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass


@dataclass
class RunResult:
    exit_code: int
    stdout: str
    stderr: str


def run(argv: list[str], timeout_s: int = 120) -> RunResult:
    r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s)
    return RunResult(exit_code=r.returncode, stdout=r.stdout, stderr=r.stderr)
