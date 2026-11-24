# MiniCut progress

## Rust port, Phase 3: verified training labels (planned)

- Plan: [docs/plans/2026-11-24-rust-port-phase3.md](docs/plans/2026-11-24-rust-port-phase3.md). Meaning first: a sampled intent is rendered through a reviewed template table, and the label comes only from the proved `try_label`. A Python freeze gate drops but never edits. Rows keep the pilot format.
- Verified core spiked (not committed): `try_label`, `pct_of`, `scale_half`/`scale_quarter`, speed percentages, `to_secs` and `kept_intervals`. 35 verified, 0 errors under 10 SMT seeds; 10/10 data-generation mutants killed; passes on stock Rust 1.99. A core rebuilt only from the plan's code blocks gives the same result.
- Found while spiking: (1) Verus does not enforce `requires` on unverified callers, so the core's public API must be total. `scripts/check_core_api.py` flags `label` and `atempo_stages` on this branch (latent; the app calls neither); this is Task 3.1. (2) A variable-divisor scale proof passed seeds 0, 2 and 3 and failed 1 and 7, so the stability check widens to seeds 0–9. (3) Existing wording traps: `trim_t4` "cut X down to" means keep; "the middle" is false when start = 0; v1 labelled "trim the first 25%" as keep-first.

## Rust port, Phases 0–1: verified core + `grammar`/`validate` (branch `rust-port`)

- State: implemented and independently reviewed (spec PASS; quality APPROVED; proof-soundness review found unconstrained `kept_ms` arms and a position-blind label spec, both fixed below). Not merged. The Python tool is still the shipped one; no Python file changed.
- `crates/minicut-core`: entirely inside `verus!`, integer units (ms, per-mille), no I/O. Proved: `validate == accepted` (well-formed + no-op rule), trim/cut_out partition, intent→label keeps meaning and the keep/remove contract, lossless mm:ss clock, atempo stages in range and within 0.001×, clarify `missing` = required − provided, speed keeps ⌊duration/factor⌋ (faster is shorter), and the other single-input edits keep the full duration. Labels are proved point-wise: every source instant survives exactly when the request says so. `scripts/verus.sh` → 21 verified, 0 errors (Verus `0.2026.11.22.1ce6a59`, check-time only).
- `scripts/mutate_core.sh`: the core verifies standalone under SMT seeds 0 1 2 3 7, and 12/12 planted bugs are rejected: keep/remove inversion, remove-last labelled as remove-first, mm:ss fraction loss, clarify listing provided args, off-by-one label, zero-length trim, odd dims, speed-1x no-op, speed meaning inverted, mute duration, and two atempo faults. The script aborts if the baseline does not verify, so a broken toolchain cannot fake kills.
- Found in review: `kept_ms`'s speed and duration-preserving arms were unconstrained (a speed inversion verified 16/0), and the label spec checked only amount and tool family, so "remove the last 5 s" mislabelled as "remove the first 5 s" verified. Both are now proved and guarded by mutants. Concat's meaning is explicitly not proved yet (Phase 2).
- Found while landing: the first atempo bound, which had division in its `ensures`, verified only when the clarify section followed it. Replaced with a division-free bound; the seed/standalone check now guards this class.
- `crates/minicut`: `minicut grammar` rule lines equal the golden from closed PR #3 (`b460cfd`), which llama.cpp's validator accepted for 1262/1262 corpus and eval targets. `minicut validate` is a strict wire layer and the single float→integer boundary. 10 Rust tests on stock stable (1 core, 1 grammar golden, 8 validate boundary, including duplicate-key rejection).
- `scripts/parity_validate.py`: 20,000 sampled calls, 0 unexpected mismatches against `validate.py`. 75 hits of the one named divergence (an extra `inputs` key on single-input tools; Rust rejects it). A planted 10 ms tolerance drift is caught (20 mismatches, exit 1), while Verus still passes: proofs check code = spec, parity checks spec = shipped behaviour.
- CI (`.github/workflows/ci.yml`): python (pytest + release controls), rust (fmt, tests, parity), verus (verify + mutation). First run is on the PR.
- Next: Phase 2, the binary (probe, argv, run, verify, publish, CLI), gated on release controls through the Rust editor. See [docs/rust-port.md](docs/rust-port.md).

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
