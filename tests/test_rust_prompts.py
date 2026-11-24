"""Rust `minicut system-prompt` must byte-equal the Python SYSTEM_PROMPT."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from minicut.llm.prompts import SYSTEM_PROMPT  # noqa: E402

BIN = ROOT / "target/debug/minicut"


def test_system_prompt_matches_python():
    if not BIN.exists():
        pytest.skip(f"binary not built: {BIN}")
    out = subprocess.run([str(BIN), "system-prompt"], capture_output=True, text=True, check=True)
    assert out.stdout == SYSTEM_PROMPT
