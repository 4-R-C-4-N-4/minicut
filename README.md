# MiniCut

Local ffmpeg editor, driven by MiniCPM5-2B + LoRA.

The model outputs exactly one JSON tool call (GBNF-constrained in
llama.cpp). A wrapper validates it against `ffprobe`, builds a
deterministic ffmpeg command (no shell), runs it, and verifies the
output. See `spec.md` and `IMPLEMENTATION_GUIDE.md`.

## v1 tools (8)

trim, cut_out, concat, scale, speed, convert, extract_audio, mute —
plus `clarify` for missing/ambiguous args.

## Quickstart

```bash
pip install -e ".[dev]"

# 1. Factory — generate verifiable fixtures + manifests
python -m factory.generate --out factory/fixtures --all
python -m factory.verify_fixtures --root factory/fixtures

# 2. Wrapper tests — hand-written JSON, no model needed
python -m pytest tests/ -q

# 3. One edit by hand (from a workdir containing clip.mp4)
python - <<'PY'
from pathlib import Path
from minicut.wrapper import edit
print(edit({"tool": "trim", "input": "clip.mp4", "start": 12.0, "end": 27.0}, Path("workdir")))
PY
```

## Eval (Phase 3)

```bash
# Generate the frozen splits (seed 20261120, deterministic)
python -m eval.generate_eval --out eval

# Sanity: prove harness + cases execute (NOT a baseline)
python -m eval.run_eval --all --backend oracle --label sanity_oracle --out eval/results

# Real baseline (needs llama-server + MiniCPM5-2B GGUF)
python -m eval.run_eval --all --backend http --label baseline_v1 --out eval/results
```

## Layout

- `minicut/` — probe, validate, build, run, verify, wrapper, grammar, prompts
- `factory/` — fixture generator + fixture verifier
- `config/tools.yaml` — tool registry (source of truth)
- `tests/` — wrapper tests, 8 tools + negative checks
- `formal/dafny/` — Dafny dictionary for v1 (`minicut.dfy`, 32 verified): the tools as types, validity + expected-output postconditions, chain and never-overwrite invariants, clarify exactness. See `formal/dafny/README.md`. No change to runtime code.
