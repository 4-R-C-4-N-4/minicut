"""Fail if a `pub fn` in the verified core has a `requires` clause.

Verus checks `requires` only for verified callers. The `minicut` app crate
is ordinary Rust, so a precondition on a function it can call is silently
unchecked there. Every core function the app can reach must therefore be
total: no `requires`, with failure expressed as `Option` in its `ensures`.
Functions that need a precondition stay `pub(crate)` (or private) and are
reached through a checked wrapper such as `try_label`.

    python scripts/check_core_api.py [crates/minicut-core/src/lib.rs]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

DEFAULT = Path(__file__).resolve().parents[1] / "crates/minicut-core/src/lib.rs"
# `pub fn` / `pub const fn` exactly: pub(crate), proof fn and spec fn are fine.
PUB_EXEC_FN = re.compile(r"^\s*pub\s+(?:const\s+)?fn\s+(\w+)", re.M)


def offenders(src: str) -> list[str]:
    bad = []
    for m in PUB_EXEC_FN.finditer(src):
        body_start = src.find("{", m.end())
        signature = src[m.end():body_start]
        if re.search(r"\brequires\b", signature):
            line = src.count("\n", 0, m.start()) + 1
            bad.append(f"line {line}: pub fn {m.group(1)} has `requires`")
    return bad


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else DEFAULT
    src = path.read_text().split("} // verus!")[0]
    bad = offenders(src)
    for b in bad:
        print(f"{path}: {b}")
    if not bad:
        print(f"{path}: every pub exec fn is total ({len(PUB_EXEC_FN.findall(src))} checked)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
