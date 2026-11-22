# MiniCut development benchmark board

Matched `intent_v2` evaluation on the four frozen development splits; not an independent release evaluation. Historical verification-only baselines are not comparable.

| Run (seed) | Stock | Tuned | Paired gains / regressions | Keep/remove | New inversions | Gate |
|---|---:|---:|---:|---:|---:|---|
| `pilot-20261121-191106` (42) | 173/200 | 197/200 | 27 / 3 | 54/64 → 63/64 | 1 | Fail |

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
