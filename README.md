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

## Output policy

MiniCut leaves source clips unchanged and gives each successful edit a new
file. The wrapper ignores model-proposed output names, renders privately, then
publishes the verified result under an unused name without copying it again.
Handled failures and timeouts do not publish a derivative. Verification checks
the selected operation, not whether the model understood the request.

By default, derivatives go in the workdir. Pass `output_dir=Path(...)` to
`edit()` to put them on another drive. Publication requires a destination
filesystem that supports hard links. There is no in-place or automatic
deletion mode.

## Development eval and training

See [progress](PROGRESS.md), the [pre-GPU plan](docs/pre-gpu-plan.md), and
the [training runbook](training/README.md). The existing eval inputs are frozen.

```bash
# Sanity: prove harness + cases execute (NOT a baseline)
python -m eval.run_eval --all --backend oracle --out "$(mktemp -d)"

# Real baseline (needs llama-server + MiniCPM5-2B GGUF)
python -m eval.run_eval --all --backend http --label baseline_intent_v2 --out eval/results
```

## Layout

- `minicut/` — probe, validate, build, run, verify, wrapper, grammar, prompts
- `factory/` — fixture generator + fixture verifier
- `config/tools.yaml` — tool registry (source of truth)
- `tests/` — wrapper tests, 8 tools + negative checks
