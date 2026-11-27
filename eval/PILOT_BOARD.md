# MiniCut development benchmark board

Matched `intent_v2` evaluation on the four frozen development splits; not an independent release evaluation. Historical verification-only baselines are not comparable.

| Run (seed) | Stock | Tuned | Paired gains / regressions | Keep/remove | New inversions | Gate |
|---|---:|---:|---:|---:|---:|---|
| `pilot-20261121-191106` (42) | 173/200 | 197/200 | 27 / 3 | 54/64 → 63/64 | 1 | Fail |
| `pilot-20261121-191106` (42), re-graded with today's grader | 167/200 | 191/200 | 27 / 3 | 52/64 → 61/64 | 1 | Fail |
| `pilot-verified-20261124-201746` (42), `train_verified_v1` | 167/200 | 191/200 | 26 / 2 | 52/64 → 61/64 | 0 | **Pass** |

**Grader drift:** `minicut/validate.py` started rejecting no-op edits (whole-clip trim, same-size scale) on 2026-11-22 (`1e2e009`), after the first pilot was graded. Six stock predictions are identical across the two runs but now fail. Compare runs only on the same grader: the re-graded row replays the saved raw predictions through today's `eval.run_eval._run_case`. That harness reproduces the verified run's own 400 verdicts exactly.

## First pilot

MiniCPM5-2B, BF16 LoRA rank 16 / alpha 32, 100 updates, microbatch 2 × accumulation 2. Training took 112 seconds on the RTX 3090; PyTorch peak allocated/reserved CUDA memory: 7.12/16.84 GiB. Stock and tuned served the same Q4_K_M base with llama.cpp `01818e495`, with the adapter applied only in the tuned arm. Median request time was 146 ms stock and 267 ms tuned for these runs; this is not end-to-end video editing latency. Unnecessary clarifications fell from 12 to 0. On a fixed 36-row training-corpus check, stock/step 25/step 100 scored 26/34/36. Step 25 was not scored on the full development suite.

| Split | Stock | Tuned |
|---|---:|---:|
| Seen | 44/50 | 49/50 |
| Unseen combo | 42/50 | 50/50 |
| Unseen fixture | 44/50 | 49/50 |
| Unseen paraphrase | 43/50 | 49/50 |

The tuned model fixed all three stock keep/remove inversions but introduced one new inversion: `eval_unseen_paraphrase-008`, “isolate seconds 3 through 9 in clip.mp4,” was interpreted as removing that range. Both scale regressions interpreted `720p` as 720×1280 rather than the expected 1280×720. Keep stock as the default. Diagnose the directional failure before increasing training exposure; named resolution defaults are a separate interface decision.

Local evidence: `training/output/pilot-20261121-191106/` (`adapter/run.json`, `comparison.json`, `comparison.sha256`, per-arm predictions and graded rows). The CPU board command validates the saved comparison against the full case roster and recomputes the gate. See [the factory runbook](../training/README.md).


## Verified-label pilot (`train_verified_v1`)

Same recipe, seed, step cap, base, server (`01818e495`) and package versions as the first pilot. Only the data differs: `training/train_verified_v1.jsonl` (SHA `4a787fa2…`), the same size and per-tool mix, with labels from the Verus-verified core (see `training/README.md`). Training: 100 updates in 128 s. Unnecessary clarifications: 12 → 1 (first pilot: 12 → 0). Training-corpus fit check: 36/36 at step 100.

| Split | Stock | Tuned |
|---|---:|---:|
| Seen | 43/50 | 48/50 |
| Unseen combo | 39/50 | 47/50 |
| Unseen fixture | 44/50 | 49/50 |
| Unseen paraphrase | 41/50 | 47/50 |

On the same grader the two adapters **tie at 191/200** and agree on 198 cases. They differ on exactly two held-out wordings:

- `eval_unseen_paraphrase-008` "isolate seconds 3 through 9 in clip.mp4": the first pilot inverted it (cut_out), which failed its gate. The verified pilot keeps 3–9 correctly. Keep/remove inversions: 3 → 0.
- `eval_unseen_paraphrase-015` "clip.mp4 without the part from 0 to 5 seconds": the first pilot is correct. The verified pilot asks a clarify whose question invents "5 to 12 seconds" (the generator's fixed example values) and lists start/end as missing although both were given.

**Reading:** with one seed and one case each way, label quality did not move the score. It removed the keep/remove inversion and introduced an over-clarify with parroted example numbers. That second effect is a generator defect: fixed example values in clarify suggestions. Before more training, fix the suggestion examples and measure seed variance (both datasets, more than one seed) before claiming a difference.

Local evidence: `training/output/pilot-verified-20261124-201746/` (factory inputs, `comparison.json`, per-arm predictions and graded rows).


## Seed sweep: v1 pilot data vs `train_verified_v2` (seeds 1–3)

`train_verified_v2` is `train_verified_v1` with one change: clarify suggestions show slots ("keep only START to END seconds of X") instead of the fixed example "5 to 12 seconds" that the first verified pilot copied into eval_unseen_paraphrase-015. Same recipe and step cap; only the data and seed vary. All runs on today's grader; every board re-verified on CPU. The seed 42 rows are from above (v1 re-graded; verified *v1* data).

| Data | Seed | Tuned | Gains / regressions | Keep/remove (of 64) | New inversions | Unnecessary clarifies | Gate |
|---|---:|---:|---:|---:|---|---:|---|
| pilot v1 | 42 | 191 | 27 / 3 | 61 | paraphrase-008 | 0 | Fail |
| pilot v1 | 1 | 191 | 27 / 3 | 61 | paraphrase-016 | 0 | Fail |
| pilot v1 | 2 | 188 | 25 / 4 | 60 | paraphrase-006, -008 | 0 | Fail |
| pilot v1 | 3 | 189 | 24 / 2 | 61 | paraphrase-008 | 0 | Fail |
| verified v1 | 42 | 191 | 26 / 2 | 61 | — | 1 | Pass |
| verified v2 | 1 | **175** | 18 / 10 | 49 | — | **18** | Fail |
| verified v2 | 2 | 190 | 25 / 2 | 62 | — | 1 | Pass |
| verified v2 | 3 | 190 | 25 / 2 | 62 | — | 2 | Pass |

Stock is 167/200 in every run.

**Reading.**
- **Inversions:** v1 data introduces a new keep/remove inversion at every seed (4/4; paraphrase-008 "isolate seconds 3 through 9" in 3 of them). Verified data introduces none (0/4). This is the consistent, seed-stable effect of the verified labels.
- **Score:** no reliable difference. v1 data spans 188–191. Verified data scores 190–191 except one collapse.
- **The collapse (v2 seed 1):** the model over-clarifies plain keep/remove requests, for example "keep only 2 to 8 seconds of clip.mp4" → "Do you want to keep only part of clip.mp4, or cut part of it out?". 13 of its 18 unnecessary clarifies are that question. The ambiguous-reading rows (60 rows, 2.6%) taught "keep vs remove?" as a reflex that, at this seed, spread to unambiguous wordings. Likely contributor (unverified): the ambiguous template `v3_amb_cut_first` ("cut the first N seconds **of** X", 18 rows) sits one word away from the dev case "cut the first 10 seconds **off** clip.mp4" (expected cut_out).
- So the verified labels trade one failure mode (silent inversions, every seed) for another (over-asking, 1 seed in 3). A silent inversion destroys the wrong part of a video; an unnecessary question costs a turn. The safer failure mode favours verified data, but the collapse must be fixed before preferring it.

**Next.** Stabilise the ambiguity rows rather than add data:
- pair every ambiguous row with near-miss unambiguous rows (for example "cut the first N seconds off X" → cut_out next to "… of X" → clarify), so the decision is learned from wording rather than from topic;
- re-run seeds 1–3 and accept verified data only if no seed collapses.

Local evidence: `training/output/sweep-{pilot_v1,verified_v2}-s{1,2,3}/`; `scripts/seed_sweep.sh` reproduces them (it skips finished runs).


## Near-miss pairs (`train_verified_v3`, seeds 1–3)

`train_verified_v3` emits every ambiguous clarify row together with unambiguous partners on the same file, and with the same seconds where both use seconds. For example, "cut the first 7 seconds **of** X" → clarify sits next to "cut the first 7 seconds **off** X" → cut_out and "keep only the first 7 seconds of X" → trim. The dataset has 60 groups and 114 partners, with per-tool counts unchanged. All runs were graded by today's grader, and every board was re-verified on CPU.

| Data | Seed 1 | Seed 2 | Seed 3 | New inversions | Unnecessary clarifies (s1/s2/s3) | Gate |
|---|---:|---:|---:|---|---|---|
| pilot v1 | 191 | 188 | 189 | every seed | 0 / 0 / 0 | 0/3 |
| verified v2 | 175 | 190 | 190 | none | 18 / 1 / 2 | 2/3 |
| verified v3 (near-miss) | 182 | 191 | 180 | none | 7 / 1 / 13 | 3/3 |

**Reading: the pairs did not stabilise it.**
- v3 has two weak seeds out of three; v2 had one.
- Means: v1 data 189.3, v2 185.0, v3 184.3.
- The verified data still introduces no keep/remove inversion in any of its 7 runs.

The failure is concentrated on one wording:
- The dev prompt "cut the first 10 seconds off clip.mp4" (expected cut_out) appears 4 times in the dev suite.
- It flips to a clarify in every weak verified run (v2 s1, v3 s1, v3 s3), costing 4 points each time.
- It is one letter from the ambiguous template `v3_amb_cut_first` ("… of X"). Pairing added more "… off X" rows (38 vs 27 in v2) but 100 updates do not learn the of/off distinction reliably.

At v3 seed 3 the reflex also spread to other tools: "remove the audio from clip.mp4" became "Do you want to keep only the audio, or remove the audio?" (3 cases), and "scale … to 720p" became a size question.

**Next (decision needed):**
- Whether "cut the first N seconds of X" belongs in the ambiguous table at all. A wording one letter from a canonical cut is a poor teaching signal at this size.
- An ablation without reading-ambiguous rows: verified tool labels plus missing-argument clarifies only. It would show whether the inversion fix survives without the over-clarify risk.

Local evidence: `training/output/sweep-verified_v3-s{1,2,3}/`.

## Phrasebook (`train_verified_v4`, seeds 1–3)

`train_verified_v4` applies the phrasebook: one keep/remove rule, no keep-or-remove clarify rows, and 30 unsupported-request rows ("make X quiet" and three volume wordings, each offering "mute X"). All runs were graded by today's grader, and every board was re-verified on CPU.

| Data | Seed 1 | Seed 2 | Seed 3 | Mean | New inversions (seeds) | Unnecessary clarifies (s1/s2/s3) |
|---|---:|---:|---:|---:|---|---|
| pilot v1 | 191 | 188 | 189 | 189.3 | 3 of 3 | 0 / 0 / 0 |
| verified v2 | 175 | 190 | 190 | 185.0 | 0 of 3 | 18 / 1 / 2 |
| verified v3 (near-miss) | 182 | 191 | 180 | 184.3 | 0 of 3 | 7 / 1 / 13 |
| verified v4 (phrasebook) | 180 | 186 | 186 | 184.0 | 2 of 3 | 8 / 1 / 7 |

**Reading.**
- The phrasebook did not raise the score, and v4 lost the zero-inversion record. At seed 1, "clip.mp4 without the part from 5 to 10 seconds" became trim. At seed 3, "take the 0 to 5 second bit out of clip.mp4" became trim.
- A new failure that I introduced: the unsupported-request rows all answer 'Do you mean "mute X"?', and that answer spread to real mute requests.
  - "remove the audio from clip.mp4" and "make a silent version of clip.mp4" became 'Do you mean "mute clip.mp4"? I can't …' (7 cases at seed 3).
  - "turn the sound **down** in X" (unsupported) also sits next to the mute template "turn **off** the sound in X", another near-pair.
- **Cross-version pattern:** v1 data never over-clarifies (0/0/0), and every verified dataset does.
  - The consistent difference is the clarify targets. v1 questions are short (median 42 characters, "What audio format should I extract to?").
  - Ours are twice as long (median 94–95) and always end with 'For example: "<a full tool request>"'.
  - The model is trained to emit real request phrasings inside clarify answers, with twice as many supervised clarify tokens.
  - Hypothesis, untested: this is what tips plain requests into clarifies.

**Next (decision needed; the example suggestions implement the "suggest MiniCut's own phrasings" rule):** v5 = v4 with short v1-style questions (the phrasebook is now in `--help`, so it doesn't need to be in every answer) and only the v1 "make X quiet" unsupported row. If that removes the over-clarify, the suggestion text was the cause.

Local evidence: `training/output/sweep-verified_v4-s{1,2,3}/`.
