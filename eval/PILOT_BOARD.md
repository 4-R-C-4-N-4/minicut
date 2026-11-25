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
