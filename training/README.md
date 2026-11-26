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

## Current pilot and runbook

`train_v1.jsonl` and the counts above are historical. The trainer now defaults to
`train_pilot_v1.jsonl`. Build it with `python -m training.build_pilot`; metadata
records the one-based source row. The pilot contains 2,319 JSON-only examples.
It drops plain-text Q&A, hidden-fixture tasks, subsecond mm:ss labels, and
ambiguous three-copy prompts. Unspecified conversion CRF is 23; any resulting
reserved combo is discarded. Keep contrast pairs where the wording distinguishes
keeping a range from removing it.

### CPU gate (no model weights loaded)

From the repo root, with the cached HF snapshot and Transformers available:

```bash
BASE="$HOME/.cache/huggingface/hub/models--openbmb--MiniCPM5-2B/snapshots/42433e90dad95123ab87729fd7011e7110629556"
CUDA_VISIBLE_DEVICES='' python -m training.train_lora --base "$BASE" --preflight
python -m unittest discover -s tests -v
python -m eval.run_eval --all --backend oracle --out "$(mktemp -d)"
```

Preflight checks every row against the deployed no-thinking prefix with the
shared few-shot messages. Only JSON and `<|im_end|>` are supervised. Mismatched
prefixes and overlong rows fail instead of clipping. This checks the HF
rendering, not yet GGUF server rendering.

### GPU smoke (only once the GPU is available and authorized)

Install the matching CUDA PyTorch build and `training/requirements-gpu.txt` in a
separate env. Record `pip freeze`, the selected device, model and GGUF hashes, and
llama.cpp commit. Expose exactly the confirmed BF16 device before running:

```bash
python -m training.train_lora --base "$BASE" --max-steps 3 --batch 1 --grad-accum 1 \
    --out training/output/smoke_v1
python "$LLAMA_CPP/convert_lora_to_gguf.py" --base "$BASE" \
    --outfile training/output/smoke_v1.gguf training/output/smoke_v1
```

The converter's `--base` is the HF config directory, not the GGUF. The trainer
requires finite loss and a changed LoRA B weight in the bounded run. The smoke
adapter is not a quality candidate. Check its load and a known request using the
exact matching stock GGUF, loopback `llama-server`, and `--jinja`. Confirm rendered
prompt parity before the pilot. Use the same server build, flags, base GGUF,
few-shot context, and grammar for stock vs. tuned; the only serving difference
is `--lora path/to/adapter.gguf`. Save each run under a new label with
`python -m eval.run_eval --all --backend http --label <label> --out eval/results`.

The initial recipe remains rank 16, alpha 32, learning rate 1e-4 and at most two
epochs; pick microbatch from measured headroom. The historical 95.5% legacy
wrapper score is not the acceptance gate. See [the pre-GPU plan](../docs/pre-gpu-plan.md)
for the matched development comparison and independent release gate.

### Repeat a bounded development pilot

With `BASE` as above, `LLAMA_CPP` pointing at the checked-out llama.cpp,
`GGUF` the matching Q4_K_M base, and `GPU_UUID` the intended RTX 3090 UUID:

```bash
systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=1G \
  python -m training.factory pilot --run training/output/pilot-new \
  --base "$BASE" --gguf "$GGUF" --llama "$LLAMA_CPP" --gpu "$GPU_UUID"
python -m training.factory board --run training/output/pilot-new
```

The factory locks the selected GPU against other factory runs in this checkout,
trains for up to 100 steps or 15 minutes, checks the completion manifest, and
converts step 25 and the final adapter. A timeout preserves saved artifacts but
does not score a partial run as a completed pilot. The matched development
comparison uses the same base and server for stock and final; the early adapter
only runs the fixed training-corpus probe. `board` checks the saved 200-case
rosters and paired totals without touching CUDA. The release set is not used.

## Verified dataset v1 (`train_verified_v1.jsonl`)

Same size and per-tool mix as `train_pilot_v1.jsonl`, so a matched pilot run isolates label
quality. Labels come only from the Verus-verified core (`crates/minicut-core`). See
`docs/plans/2026-11-24-rust-port-phase3.md` and the reviewed template table in `docs/templates.md`.

```bash
cargo build --release
target/release/minicut label --seed 20261124 --oversample 100 \
    --out "$TMPDIR/candidates.jsonl" --meta "$TMPDIR/candidates_meta.jsonl"
python -m training.build_verified --rows "$TMPDIR/candidates.jsonl" \
    --meta "$TMPDIR/candidates_meta.jsonl" --out training --targets training/verified_targets.json
mv training/stats.json training/train_verified_v1_stats.json
```

- SHA-256 `4a787fa231e1df65a9fa517bf606fb0e165b89274ac9d49f75a67ceafb328ca1` (rows), `b4d5738b0b7c42045dd415bf11148b4ce7a6336a06bbfd8af67eb76adaa7f178` (meta).
  The same bytes come from a debug and a release build.
- **Counts:** trim 306, cut_out 368, concat 223, scale 255, speed 300, convert 193, extract_audio 244,
  mute 120, clarify 310. Clarify splits into 250 argument-missing rows (the `missing` list is computed by
  the core; the question offers our phrasings) and 60 reading-ambiguous rows (`missing: []`; the question
  offers both readings in the user's own terms).
- **Freeze gate** (drop-only; `training/train_verified_v1_stats.json`): 4,454 candidates → 360
  (template id, label) eval collisions, 75 combo-pool args, 21 (prompt, label) collisions, 4 eval prompts,
  1,675 over target → 2,319 kept.
- **Labels never need hidden clip facts.** Last-N, percentage and relative-size wordings need the clip's
  duration or size, which the model never sees. Those templates are marked `probe` in `docs/templates.md`
  and are not used for tool rows. A test checks across 3 full datasets that one prompt never maps to
  two targets.
- **Acceptance** (`scripts/accept_dataset.py`, all PASS):
  - pilot filters: 0 drops; 48 rows exempt only by a per-row proof that the clock text round-trips to the label.
    Named sizes ("540p") are not exempt: they assume a 16:9 source (P3-D12);
  - `validate.py`: 2,319 / 2,319;
  - llama.cpp `test-gbnf-validator` against `minicut grammar`: 2,319 / 2,319;
  - tokenizer preflight: max 649 tokens, min supervised 26;
  - release-grader meaning agreement: 16 / 16 trim/cut_out cases.

## Tests

```bash
python -m unittest tests.test_dataset -v   # 9 tests
```

Checks format, exact system prompt, JSON labels, clarify missing
lists, mix proportions, and freeze compliance re-derived from
`eval/freeze.json` (no held-out template id, no reserved fixture, no
combo-pool arg, no eval case sampled, no held-out prompt wording).

## Verified dataset v4 (phrasebook)

`train_verified_v4` applies the phrasebook (`docs/plans/2026-11-25-phrasebook.md`, `minicut --help`):
- keep vs remove follows one rule, so there are no keep-or-remove clarify rows;
- clarify is 280 missing-argument + 30 unsupported-request rows.

Reproduce:

    cargo build --release
    target/release/minicut label --seed 20261124 --oversample 100 --out $TMPDIR/c.jsonl --meta $TMPDIR/cm.jsonl
    python -m training.build_verified --rows $TMPDIR/c.jsonl --meta $TMPDIR/cm.jsonl --out training \
        --targets training/verified_targets.json --name train_verified_v4
    mv training/stats.json training/train_verified_v4_stats.json

SHA-256 `6e55c920011f17f353d1811ff148c1e1b6bdbfb5fced73cbe66dc27b95bb92fe`. All five `scripts/accept_dataset.py` checks PASS.
