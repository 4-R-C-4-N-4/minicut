# MiniCut eval — Phase 3 (frozen)

The baseline number lives here. These four splits were generated
2026-11-20 (seed 20261120, `generate_eval.py`) **before any training
data exists**, so training cannot leak into them.

## Files

- `eval_seen.jsonl` — train-pool paraphrases, new arg values (50)
- `eval_unseen_paraphrase.jsonl` — held-out paraphrases, never in training (50)
- `eval_unseen_combo.jsonl` — train-pool paraphrases, reserved arg combos (50)
- `eval_unseen_fixture.jsonl` — `long_60s_1080p` only, a fixture never in training (50)
- `freeze.json` — seed, per-split counts, held-out template ids, reserved
  fixture, combo pools, and the freeze rules
- `generate_eval.py` — the generator (deterministic: same seed, same files)
- `run_eval.py` — the harness
- `results/` — per-run JSONL + summary markdown. `baseline_v1.md` is
  written only by a real model run (`--backend http --label baseline_v1`).
  `sanity_oracle_*` files are harness checks, NOT a baseline.

## Freeze rules (Phase 4 must obey these)

1. Held-out template ids (`*_hN` in `freeze.json`) never appear in training data.
2. The `long_60s_1080p` fixture is never used in training data.
3. The combo arg pools in `freeze.json` are never used in training data.
4. Eval cases are never sampled into the training pool.
5. Do not edit the frozen JSONL files after the baseline run. New runs
   get new files; the baseline file is never overwritten.

## How to run the baseline

Needs a llama.cpp server with the MiniCPM5-2B GGUF loaded, thinking off:

```bash
# terminal 1: start the server (model path is yours to choose)
llama-server -m <path-to-minicpm5-2b.gguf> --port 8080

# terminal 2: baseline, all four splits
python -m eval.run_eval --all --backend http --label baseline_v1 --out eval/results
```

The harness sends the system prompt + few-shot examples from
`minicut/llm/prompts.py`, temperature 0, and the GBNF grammar from
`minicut/grammar/minicut.gbnf`. If your llama.cpp version ignores the
`grammar` field in chat completions, pass `--no-grammar` — the run log
will record that the grammar was not enforced.

## What the numbers mean

Per case: parse_ok (valid JSON tool call), tool_correct, args_correct
(input/output ignored — the wrapper derives output), execution_ok,
verification_pass (ffprobe/colour/tone checks, not just exit code 0).
The headline number is **verification pass rate on the unseen splits**.
Per the guide: if the base model is already ≥80% on simple edits, LoRA
is for tightening format and edge cases; if below, LoRA is the main lever.
Either way the decision gets recorded in `results/baseline_v1_*.md`,
not assumed.

## Sanity check (no model needed)

```bash
python -m eval.run_eval --all --backend oracle --label sanity_oracle --out eval/results
```

Oracle feeds each case's expected JSON straight to the wrapper. It
proves the harness and every eval case execute end to end. See
`results/sanity_oracle_*.md` for the latest run — it must read 100%
on all four splits before the baseline means anything. It is not a
baseline and the harness refuses to label it one.

## Things found while building this

- **WebM convert was broken.** `minicut/build.py` wrote AAC audio into
  WebM, which only allows Vorbis/Opus — `convert vp9/webm` failed at
  the mux. Fixed: WebM now uses `libopus`; mkv/mp4 keep AAC. Verified
  `vp9/webm` passes on audio and silent fixtures; wrapper tests still pass.
- **AAC extract had a wrong duration.** Extracting audio as raw `.aac`
  (ADTS) under-reports duration by ~0.35s (29.67 vs 30.02), which
  failed verification for no real reason. Fixed: `extract_audio` with
  format `aac` now writes `.m4a` — same AAC codec, proper container,
  duration 30.03. Verified directly; the format arg is still `aac`.
- **A cut_out case removed the whole fixture.** The first oracle run
  caught it: `cut_out 0–10` on the 10s fixture leaves nothing for
  ffmpeg to concat. Fixed in the generator: a cut_out must leave
  ≥1s of footage, and the combo pool gained a (2, 8) entry valid on
  the 10s fixture.
- **Filename convention.** Every eval prompt says `clip.mp4`, matching
  the few-shot examples in `minicut/llm/prompts.py`, and the harness
  copies the fixture to `clip.mp4` per case — so the model is never
  punished for the filename the few-shot examples taught it.
