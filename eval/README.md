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

## Run the intent-aware development eval

```bash
# Use a matching MiniCPM5-2B GGUF on a private llama.cpp server with --jinja.
python -m eval.run_eval --all --backend http --label baseline_intent_v2 --out eval/results
# Full CPU control: no model or server involved.
python -m eval.run_eval --all --backend oracle --out "$(mktemp -d)"
```

The HTTP evaluator sends the shared system and few-shot messages, greedy decoding,
thinking off, and the GBNF grammar by default. Keep those settings fixed between
stock and adapter. Use a separate label per run; the runner refuses to overwrite
existing reports. The four frozen splits are development evidence, not a release set.
The independent release suite lives in [`release/`](release/README.md).

`intent_v2` records parse/tool/argument agreement and wrapper media verification
separately. `task_pass` requires the intended action and canonical arguments plus
verified media, or an accepted clarification with exactly the expected missing
fields. The grader does not judge the wording of clarification questions.

The frozen files were not edited. In v2, unspecified CRF means the documented
wrapper default (23), and the two bare “with itself” concat phrasings mean two
copies. Reports retain `frozen_expected` wherever the overlay changed a label.
The oracle runs the normalized intent through the wrapper; its score is a fixture
and harness check, not a model score. `baseline_v1`–`baseline_v3` are historical
and their legacy verification rates are not comparable to `task_pass`.

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
