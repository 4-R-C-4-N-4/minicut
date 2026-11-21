# MiniCut Phase 4 — Training dataset v1

`train_v1.jsonl` — 3000 examples, one `{"messages": [...]}` per line
(system / user / assistant). The assistant content is the ground-truth
JSON tool call only: no markdown fence, no explanation, thinking off.
The system prompt is the exact string from `minicut/llm/prompts.py`,
identical to inference. Labels include the canonical `output` field
from `minicut/build.py:default_output`, matching the few-shot format
the model sees at inference; the wrapper ignores it at runtime.

`train_v1_meta.jsonl` is line-aligned provenance for auditing (not
training data): `{"kind": "tool"|"clarify"|"general", "tool", "fixture",
"template_id"}`.

## Generation

```bash
python -m training.generate_dataset --out training --seed 20261121
```

Seed 20261121 (deliberately not the eval seed 20261120). Generator:
`training/generate_dataset.py`. It reuses the eval template pools and
arg-sampling logic from `eval/generate_eval.py` (train templates,
`_fmt_secs`/`_mmss`, `_silent_ok`, fixture durations) and adds fresh
paraphrases where the eval pools are exhausted (see below).

## Counts per tool

| tool | count | share |
|---|---|---|
| trim | 360 | 12.0% |
| cut_out | 400 | 13.3% |
| concat | 270 | 9.0% |
| scale | 270 | 9.0% |
| speed | 300 | 10.0% |
| convert | 300 | 10.0% |
| extract_audio | 270 | 9.0% |
| mute | 120 | 4.0% |
| clarify | 350 | 11.7% |
| general | 360 | 12.0% |
| **total** | **3000** | |

Weak spots from baseline_v2 (94.5%) / v3 (95.5%) are over-represented:
cut_out (start/end inversion on "remove this range" phrasings),
speed / convert / extract_audio (arg names), clarify (exact missing
lists). Contrast pairs: 38 trim + 38 cut_out with the same times
("cut the first 15 seconds off" vs "trim to the first 15 seconds",
"remove 5 to 15 seconds" vs "keep only 5 to 15 seconds").

## Mix

- Tool edits (incl. clarify): 2640 (88.0%)
- Clarify within that: 350 (11.7% of total) — 310 ambiguous tool-only
  prompts with the exact minimal `missing` list, 40 invalid-against-
  manifest prompts (end > duration, out-of-range speed, odd scale
  dimensions) -> clarify. Limitation, recorded honestly: the model
  does not see the probe at inference, so invalid-against-manifest
  clarify only teaches "impossible numbers -> ask"; kept small.
- General instruction mix: 360 (12.0%) — hand-written short Q&A by
  the dataset author (60 topics, question + answer variants),
  plain-text assistant replies, no tool call. Source: this repo,
  `GENERAL_BASES` in the generator. Purpose: limit catastrophic
  forgetting of general language (guide §7.1.5).
- Unit variety: seconds, mm:ss, short "12s" units, percentages and
  halves resolved against the fixture duration at generation time
  (81 percentage/half prompts, 115 mm:ss prompts); resolutions as
  WxH, named 540p/270p, and relative "half size"/"quarter size";
  speeds as factors and percentages ("25% faster" = 1.25x). Labels
  are always canonical. Fixtures cap at 30s, so minute-style times
  ("1m30s") cannot occur in v1.
- Filenames: prompts mostly say `clip.mp4` (the eval convention,
  559/3000), but 30 filenames are used. This is deliberate: the
  low-entropy tools (concat, mute, extract, clarify) have almost no
  distinct (prompt, label) pairs otherwise, and it teaches the model
  to copy the filename from the prompt into the `input` field.

## Freeze-rule compliance

Read against `eval/freeze.json` and the frozen eval files:

- Only train template ids are used. Mute / concat / clarify train
  templates are **not** reused: every mute_tN and clarify_tN prompt
  appears in eval, so those tools get fresh paraphrases written for
  this dataset (ids `train_*_xN`), each checked against the eval
  prompt set.
- The reserved fixture `long_60s_1080p` is never used. Training
  fixtures: base_30s (1262), portrait_30s (454), short_10s (338),
  silent_30s (276). Extract/mute stay off silent_30s (no audio).
- Combo arg pools are never used (0 hits, verified independently in
  `tests/test_dataset.py`). Arg pools are additionally disjoint from
  the eval *seen* pools where a disjoint pool exists: times sampled
  on a 0.5s grid excluding all eval pairs, fresh scale pairs, fresh
  speed factors, fresh convert triples. Extract is mp3-only in
  training: the combo pool is {aac, wav} and wav is also in the seen
  pool, so no other format is available without breaking the freeze.
- No eval case is sampled into training: every candidate (prompt,
  label) and (template_id, label) pair is checked against all 200
  eval cases before acceptance. 272 candidates were dropped as eval
  collisions during generation; the final file has 0 (prompt, label)
  collisions, verified independently by the test suite. The cut_out
  "remove this range" phrasings avoid the held-out cut_h3 wording
  ("without the part from ...") entirely.
- No frozen eval file, baseline result, or prompt was edited to make
  this dataset.

## Validation

- Every label was validated with `minicut.validate.validate()`
  against its fixture's probe before writing (semantic check, no
  ffmpeg): **0 labels failed validation**.
- Duplicates dropped during generation: 1747 (mostly low-entropy
  tools re-sampling the same (prompt, label) pair).

## Wrapper sample check

```bash
python -m training.check_sample --n 200 --seed 20261122
```

Result: **200/200 passed (100.0%)** — every sampled tool label
executed through `wrapper.edit()` and verified, zero failures. No
label fixes and no wrapper fixes were needed; the only issues found
during dataset construction were generator-side (see "Validation"
above) and were fixed in the generator before the final file was
written. Check script: `training/check_sample.py` (samples tool
examples only; clarify and general involve no ffmpeg).

## LoRA config

`training/lora_config.yaml` — r 16, alpha 32, lr 1e-4, epochs 2,
max_seq_len 1024, general_mix 0.12, thinking off. Training runs on
Ivy's local GPU, not on the CPU-only VM where this dataset was built.
Framework: plain transformers Trainer + PEFT (`training/train_lora.py`,
no TRL) — completion-only loss, tokenizer's own chat template.

## Train on the local GPU (runbook for Ivy's machine)

This repo was built on a CPU-only VM, so training was never run here.
On the GPU machine, from the repo root:

```bash
pip install -r training/requirements-gpu.txt
# torch: install the CUDA build matching the machine, e.g.
#   pip install torch --index-url https://download.pytorch.org/whl/cu121
python -m training.train_lora --data training/train_v1.jsonl --out training/output/lora_v1
```

The script refuses to run without CUDA, trains on the assistant
completion only (prompt tokens masked), and saves the adapter +
tokenizer to `training/output/lora_v1`. Expect roughly: 3000 short
examples x 2 epochs, batch 8 x grad-accum 2 — a few tens of minutes
on a mid-range GPU; lower `--batch` if VRAM is tight (the base model
is 2B params in bf16, ~4 GB, plus LoRA + optimizer + activations).

Then convert the adapter to GGUF and eval it against the frozen
baseline (baseline_v3: 95.5% verification, the number to beat):

```bash
python llama.cpp/convert_lora_to_gguf.py --base models/MiniCPM5-2B-F16.gguf \
    --out training/output/lora_v1.gguf training/output/lora_v1
llama-server -m models/MiniCPM5-2B-F16.gguf --lora training/output/lora_v1.gguf \
    --jinja -c 8192 --port 8080
python -m eval.run_eval --all --backend http --label lora_v1 --out eval/results
```

Acceptance (guide §8.3): verification on unseen paraphrase + unseen
combo beats baseline_v3, no regression on seen, 8/8 wrapper tests
still pass. Ship only then. If it does not beat baseline, inspect
failures first — the usual suspects are dataset label errors, a
template mismatch between train and inference, or too few contrast
pairs. Change one thing per run and record it. Adapter weights are
gitignored; commit the adapter config, the conversion command, and
the `lora_v1` eval results — not the weights.

## Tests

```bash
python -m unittest tests.test_dataset -v   # 9 tests
```

Checks format, exact system prompt, JSON labels, clarify missing
lists, mix proportions, and freeze compliance re-derived from
`eval/freeze.json` (no held-out template id, no reserved fixture, no
combo-pool arg, no eval case sampled, no held-out prompt wording).
