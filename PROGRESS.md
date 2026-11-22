# MiniCut progress

## CLI with stock model

- `minicut REQUEST FILE...` (`minicut/cli.py`, packaged as a console script): asks local llama.cpp for one tool call, binds it to the given files, validates against ffprobe, then renders one verified derivative, or previews with `--preview`. `--output-dir`, `--json`, exit codes in README. The model client is shared with the dev eval (`minicut/llm/client.py`).
- `validate()` now rejects edits that would change nothing (trim covering the whole clip, speed 1x, scale to current size). No dev, release or pilot label is a no-op.
- Fresh wheel install outside the source tree, then real stock MiniCPM5-2B Q4_K_M on llama.cpp `01818e495` (RTX 3090), base and short fixtures: remove 5–12s, scale 640x360, a.mp4-then-b.mp4 concat and preview were correct. Impossible 20–45s on 30s was rejected. "make clip.mp4 shorter" came back as a whole-clip trim and was rejected as a no-op instead of rendering. "keep only 5 to 12 seconds" came back as keep 0–12s, with or without the filename in the request; the plan line shows it. Sources were byte-identical in all 8 runs; no files were written for rejections or preview.
- 55 tests pass; release controls still pass. `nvidia-smi` currently fails with a driver/library version mismatch (CUDA itself works); a reboot should clear it.

## Release benchmark (frozen, controls only)

- `eval/release/`: 38 pinned cases separate from the 200 development cases and training data — 16 keep/remove, 8 ambiguous, 6 distinct-file concat orders, 8 other tools — on three new synthetic fixtures. Each attempt is graded separately on intent (the output's measured effect versus the expected edit) and source preservation (SHA-256).
- `python -m eval.release controls` passed: 38/38 expected outcomes pass both axes; 30/30 wrong-but-executable edits, 30/30 needless clarifications and 8/8 guessed edits fail intent; sources unchanged in all 114 attempts. Changed, truncated and duplicated rosters are rejected, as is a source-modifying editor. No model has been scored on it.

## Derivative-only output

- `wrapper.edit()` stages and verifies renders before atomic no-replace
  publication; handled failures clean staging. `output_dir` is a trusted
  application setting, not a tool argument. Sources are never render targets.
- CPU suite: 38 passing, including a real source checksum, a publication
  collision, a dangling destination symlink, and failed-render cleanup.

## Pre-GPU readiness

- Base: `40023e6` on `main`; historical v1 corpus, frozen development inputs, and baseline artifacts are preserved.
- Before changes: 22 CPU tests pass after fixture generation. Baseline v3's 191/200 measures legacy wrapper success, not fulfillment of the requested edit. The 160/200 strict-label diagnostic still inherits imperfect frozen labels.
- CPU gate passed: 28 tests, 40/40 sampled pilot labels through FFmpeg, and 200/200 oracle cases under `intent_v2` (50 per split). Oracle results validate the harness, not model quality.
- The reproducible pilot has 2,319 unique JSON-only requests. Offline preflight with cached MiniCPM5-2B revision `42433e90dad95123ab87729fd7011e7110629556` checked every row (max sequence 625 tokens, min supervised suffix 26 tokens). Data SHA-256: `5f7d9b1d1325ebfa6f848c473dda1f3f2f6b9bf342409cdf2e22db648fecca02`.
- GPU smoke passed on the RTX 3090: the real MiniCPM5-2B checkpoint completed three BF16 optimizer steps with finite losses and changed LoRA B weights. The adapter converted to GGUF and loaded at scale 1 with the local Q4_K_M base on llama.cpp `01818e495`. For one trim request, all 569 prompt tokens matched HF and the server returned the correct JSON call.
- Smoke artifacts: `training/output/gpu-smoke-20261121-135102/` (`train.log`, `adapter/run.json`, `convert.log`, `gguf.sha256`, `serve.json`). The smoke checked training and serving, not model quality or full-run VRAM.
- Pilot launched on the RTX 3090 in `training/output/pilot-20261121-191106/` with microbatch 2, accumulation 2, seed 42, up to 100 updates and a 15-minute wall cap. Intermediate saves are every 25 updates. The matched comparison uses the 200 development cases, not a release suite. Gate: positive net task gain, better trim/cut-out results, and no new keep/remove inversion.
- Pilot training completed all 100 updates in 112 seconds. Matched development score: 173/200 stock, 197/200 tuned (27 gains, 3 regressions); keep/remove improved from 54/64 to 63/64. The new isolate-to-cut inversion failed the predeclared gate. Keep stock as default. See [the benchmark board](eval/PILOT_BOARD.md) for the regression IDs and evidence.
- `training.factory` now has a bounded `pilot` command and a CPU-only `board` command. The staged train/export/compare workflow ran on hardware; `board` verified the real artifacts. No duplicate GPU run or release eval was started.

See [the pre-GPU plan](docs/pre-gpu-plan.md) for gates and next steps.
