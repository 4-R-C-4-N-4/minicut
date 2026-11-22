# MiniCut release benchmark

A frozen, independent suite for release decisions. It is separate from the 200 development cases in `eval/eval_*.jsonl` and from all training data. Do not tune prompts, data or recipes against it; score only a selected candidate, alongside stock in the same runtime.

## Run the calibration controls (CPU, no model)

```bash
python -m eval.release controls --report /tmp/release-controls.json
```

Exits 0 only when every control behaves as expected:

- The pinned roster loads; a changed byte, a dropped case or a duplicated case is rejected, even if someone re-pins the digest after the change.
- No release prompt template (filenames and numbers normalised), filename or fixture appears in the development splits, `train_v1`, or `train_pilot_v1`.
- For every edit case, the expected edit passes; a wrong-but-executable control fails; a clarification instead of the edit fails and publishes nothing.
- For every ambiguous case, a clarification naming exactly the missing fields passes and publishes nothing; a guessed edit fails.
- Sources stay byte-identical in every attempt, and an editor that modifies a source is detected.

Missing fixtures are generated into `eval/release/fixtures/` (git-ignored) from `fixtures.json`, then checked against their specs before any case runs. Expect roughly 15 seconds on 24 cores.

## What is frozen

- `cases.jsonl` — 38 cases: 16 keep/remove (8 pairs over the same ranges, including same-duration pairs), 8 ambiguous requests, 6 distinct-file concat orders, and 8 other-tool cases (scale, speed, mute, extract audio, convert).
- `fixtures.json` — three solid-colour fixtures with new colours, durations, an 18s and a 12s clip with audio, and a silent portrait clip. Workspace filenames (`interview.mp4`, `broll.mp4`, `phone.mp4`) are not used in training.

Both files are pinned by SHA-256, with case and category counts, in `bench.PIN`. Changing either requires a new versioned suite, not an edit in place.

## How a case is scored

Each attempt runs a candidate tool call through `minicut.wrapper.edit` in a fresh workspace holding copies of the case's fixtures. Two results are recorded independently:

- `intent_pass`: the single published derivative has the expected effect. The grader derives the expected colour timeline, duration, dimensions and streams from the expected edit and the fixture specs, then samples the output. It never trusts the candidate call or the wrapper's own verification. For ambiguous cases, a pass requires a clarification with exactly the expected missing fields and nothing published.
- `sources_unchanged`: every staged source has the same SHA-256 after the attempt.

`bench.run_case(case, call, fixtures_root)` accepts any candidate call, so a model run can reuse it. Model scoring, thresholds and the stock comparison are a separate step and are not implemented here.

Limits: solid-colour synthetic clips with sine tones; timing is checked at segment quarter points with 0.25s duration tolerance (0.35s for concat). Prompt separation is a normalised exact-template check, not a paraphrase detector. Named-resolution requests such as "720p" are excluded until the aspect-ratio policy is decided.
