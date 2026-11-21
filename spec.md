# MiniCut — spec

**Project name:** MiniCut
**Status:** Draft v0.1 — 2026-11-19
**Model card:** https://huggingface.co/openbmb/MiniCPM5-2B

## 1. Goal

Turn MiniCPM5-2B into a local ffmpeg editor for average editing tasks.

Not a general video chatbot. Not a raw-bash generator. A narrow, reliable
tool: user says what they want in plain language, the model picks one
structured edit, a wrapper builds and runs the ffmpeg command, and the
result is verified.

Average editing = the everyday set: trim, cut out a section, concat,
resize/scale, crop, rotate, change speed, convert format/codec, extract
audio, mute/remove audio, fade in/out. Multi-step edits come later, built
out of single steps.

## 2. Base model

- **Model:** `openbmb/MiniCPM5-2B`
- **Type:** Dense causal LM, standard `LlamaForCausalLM`
- **Params:** 2,516,756,480 total / 1,981,982,720 non-embedding
- **Layers:** 42, GQA 16 Q / 2 KV heads
- **Context:** 131,072 tokens
- **License:** Apache-2.0
- **Post-training:** SFT + RL + OPD (On-Policy Distillation). Use the final
  RL+OPD checkpoint as the fine-tune base, not the SFT-only / Base /
  Midtrain checkpoints.
- **Sampling (per card):** `temperature=1.0, top_p=0.95, min_p=0.0`
  (llama.cpp defaults `min_p=0.05` — set it to 0.0, it can cause repetition)
- **Thinking:** `enable_thinking` exists. Default OFF for single-step
  edits — faster, cleaner output. ON only for multi-step planning later.
- **Tool calling:** XML-style natively; SGLang has a built-in `minicpm5`
  parser. MiniCut does not rely on that — it uses llama.cpp + GBNF
  (see §4), so the output format is enforced by grammar, not by hope.
- **Why this model fits:** small enough to run locally and LoRA cheaply,
  strong for its size on code reasoning and tool use (card reports
  53.9 avg across 40+ benchmarks, BFCL v4 66.6, LiveCodeBench 69.1),
  131k context leaves room to feed `ffprobe` output back in on retry.

## 3. Architecture

```
User prompt (plain language)
  → MiniCPM5-2B + LoRA adapter (llama.cpp)
  → GBNF grammar → exactly one JSON tool call
  → Wrapper: validate against ffprobe + schema
  → Wrapper: build ffmpeg argv (deterministic template, no shell)
  → ffmpeg executes
  → Wrapper: verify output (ffprobe + frame/audio checks)
  → On failure: feed stderr + probe back to model, max 2 retries
  → Output file, or a clarification question if args are missing/ambiguous
```

Rules that do not change:

1. The model never emits raw ffmpeg / bash. It emits JSON only.
2. The wrapper owns ffmpeg syntax, flag order, quoting, and codecs.
3. No shell. `subprocess` argv list only.
4. No execution without validation. Every arg is checked against the
   actual input file (duration, resolution, streams) before running.
5. Verification is part of the pipeline, not a manual step.

## 4. Runtime: llama.cpp + LoRA + GBNF grammar

- **Runtime:** llama.cpp (`llama-server` / `llama-cli`).
- **Adapter:** LoRA fine-tune, loaded as a GGUF adapter in llama.cpp.
  Training itself can be done in TRL/PEFT, LLaMA-Factory, ms-swift, or
  Unsloth (all have official MiniCPM5 cookbooks on the model card),
  then converted for llama.cpp inference. The adapter is the
  deliverable; the base weights stay untouched.
- **Grammar:** GBNF grammar fixed for the wrapper. It constrains output
  to the tool-call JSON schema in §5 — tool name from an enum, typed
  fields only, no free text, no extra keys. This is what makes a 2B
  model reliable enough for a deterministic wrapper: the grammar
  removes the whole class of "almost valid JSON / invented flag" errors.
- **Decoding:** greedy or near-greedy for tool calls. Thinking off
  for v1 single-step edits.

## 5. Tool schema (v1)

One tool call per turn. JSON only, e.g.:

```json
{"tool": "trim", "input": "clip.mp4", "start": 12.0, "end": 45.0, "output": "clip_trim.mp4"}
```

v1 intents (average editing only) — DECISION 2026-11-19: v1 is 8 tools, not 12.
Cut to: trim, cut_out, concat, scale, speed, convert, extract_audio, mute.
Crop, rotate, fade_in, fade_out move to v2. Clarification is a tool call
(`clarify`), not plain text — see §5.

| Tool | Args | Notes |
|---|---|---|
| `trim` | start, end | seconds; `end <= duration` |
| `cut_out` | start, end | remove a section, keep the rest |
| `concat` | inputs[], | same codec/res preferred; re-encode fallback |
| `scale` | width, height | dims must be even for yuv420p |
| `crop` | x, y, width, height | within frame bounds |
| `rotate` | degrees | 90 / 180 / 270 |
| `speed` | factor | 0.25–4.0; audio pitch handling via atempo chain |
| `convert` | codec, container, crf | allowlist only (see §7) |
| `extract_audio` | format | mp3 / wav / aac |
| `mute` | — | drop audio stream |
| `fade_in` | duration | video and/or audio |
| `fade_out` | duration | video and/or audio |

Out of scope for v1: subtitles/overlay text, filters beyond the list,
scene detection, silence removal, multi-step plans, batch jobs.
Those are v2, composed from v1 tools.

Ambiguity handling is a first-class output, not a failure:
- Missing arg (e.g. "trim it" with no times) → clarification tool call,
  not a guessed time.
- Unit normalisation is the model's job: `mm:ss`, `1m30s`, seconds,
  percentages ("cut the first 10%") all map to seconds in the JSON.
- "cut the first 10 seconds" vs "trim to 10 seconds" are different
  tools (`cut_out` 0–10 vs `trim` 0–10) — dataset must contrast these.

## 6. Wrapper

Responsibilities, in order:

1. **Probe:** `ffprobe -show_format -show_streams` → duration,
   resolution, fps, codecs, audio streams, audio sample rate.
2. **Validate:** schema check (grammar already did most of this),
   then semantic check: times in range, crop in bounds, scale dims
   even, output path inside the sandbox/workdir, no overwrite of the
   input unless explicitly allowed.
3. **Build:** deterministic template per tool. One template per tool,
   tested. No string concatenation of user text into a command.
4. **Run:** argv list, capture stdout/stderr, timeout per job.
5. **Verify:** same checks as the test factory (§8) — duration,
   dimensions, codec, stream presence. A zero exit code is not enough.
6. **Retry:** on failure, feed the probe + stderr back to the model
   as a tool result and ask for a corrected JSON call. Max 2 retries,
   then report the error plainly.
7. **Report:** output path + what changed (duration, size, codec),
   or the clarification question.

## 7. Safety

- Allowlist for codecs/containers/filters. No user-supplied filter
  strings reach ffmpeg.
- Output sandbox: outputs only under the project workdir.
- No network, no shell metacharacters, no `rm` / move / delete tools.
- Input filenames are treated as data, never interpolated into a
  shell string.
- Size/duration caps per job (to be set in v1 config).

## 8. Test factory — verifiable videos

This is what makes the fine-tune trainable and evaluable. The factory
generates its own fixtures with ffmpeg, so every label is ground truth.

**Fixture generation (ffmpeg `lavfi` only, no real footage needed):**
- Video: `testsrc2` — frame counter / timestamp burned in, so a trim
  is checkable by sampling the first frame.
- Colour segments: e.g. 0–10s red, 10–20s green, 20–30s blue, so
  cut/concat order is checkable by pixel colour at a timestamp.
- Audio: `sine` tones per segment (440 Hz, then 880 Hz), so audio
  cuts / mute / extract are checkable by frequency, not just
  "has audio".
- Fixture matrix: durations 10s / 30s / 60s; 720p / 1080p;
  24 / 30 / 60 fps; with-audio / silent; portrait + landscape.

**Manifest:** every fixture ships a JSON manifest — duration,
resolution, fps, codecs, audio freq, colour-at-time, etc. The
manifest is the expected output for verification.

**Verification per edit (programmatic, no eyeballing):**
- trim / cut_out: `ffprobe` duration ±0.1s + first-frame timecode
- scale / crop / rotate: `ffprobe` width/height (+ swap on rotate)
- speed: duration scales by 1/factor
- convert: codec / container from `ffprobe`
- mute / extract_audio: audio stream present/absent, tone frequency
- concat: duration = sum, colour at 5s / 15s / 25s in the right order
- fade_in / fade_out: frame brightness near start/end

**Dataset generation from the factory:**
1. Factory generates video + manifest.
2. Template generates the ground-truth JSON tool call.
3. Prompt is paraphrased 5–10 ways per intent (hand-written + model
   paraphrase, human-checked).
4. Hold out unseen paraphrases AND unseen arg combos for eval, so
   the test measures generalisation, not memorisation.
5. Include negatives: ambiguous prompts (→ clarification), invalid
   times (end > duration), out-of-bounds crop, missing args.

Target dataset size for v1: 2k–10k examples across the v1 intents,
heavily weighted to the most common ones (trim, convert, scale,
extract_audio).

## 9. Fine-tuning

- **Base:** final RL+OPD `openbmb/MiniCPM5-2B` checkpoint.
- **Method:** LoRA. Starting point: r=16–32, alpha 2×r,
  lr 1e-4–2e-4, 2–3 epochs. Tune from eval, don't guess twice.
- **Format:** chat template, thinking off, assistant output = the
  JSON tool call only. No explanation, no bash, no preamble.
- **Mix:** ~10–20% general instruction data mixed in, to limit
  catastrophic forgetting / overfitting to the narrow JSON format.
- **What is being learned:** slot-filling, unit normalisation,
  intent disambiguation, and when to clarify — not ffmpeg syntax.
  The wrapper owns syntax.
- **Baseline first:** before training, prompt the base model with a
  system prompt + few-shot examples + a small cheat sheet, and run
  it through the factory eval on ~50 real cases. If it's already
  ~80%+ on simple edits, LoRA is for tightening format and edge
  cases. If it's below that, LoRA is the main lever. Record the
  baseline number — it's the number the fine-tune has to beat.

## 10. Evaluation

Metrics, on the held-out factory set:
- Tool accuracy (right intent)
- Arg accuracy (right values, within tolerance for times)
- Execution success (ffmpeg exit 0)
- Verification pass rate (the real metric — output matches manifest)
- Retry count (should fall after fine-tuning)
- Latency per edit (thinking off, local)

Splits: seen paraphrase / unseen paraphrase / unseen arg combo /
unseen fixture. Report all four, not just the easy one.

## 11. Milestones

1. **Factory v1** — fixture generator + manifest + verifier for the
   v1 intents. No model involved yet.
2. **Wrapper v1** — probe → validate → build → run → verify, with
   hand-written JSON inputs. Proves the templates work.
3. **Baseline eval** — base MiniCPM5-2B + prompt + grammar, through
   the factory. Record the number.
4. **Dataset v1** — factory-generated prompts + ground-truth JSON,
   with negatives and clarifications.
5. **LoRA v1** — train, convert to GGUF, load in llama.cpp, eval
   against baseline on the same held-out set.
6. **v2 (later)** — multi-step plans, subtitles/overlay, silence
   removal, batch.

## 12. Open questions

- Exact v1 intent list — is the §5 table the right 12, or cut to 8?
- Thinking on or off for multi-step (v2) — measure, don't assume.
- LoRA rank / LR — set from a small sweep, not from this spec.
- Where outputs live by default, and overwrite policy.
- Whether clarification is a separate tool call or a plain-text
  reply (grammar currently assumes tool call; decide in wrapper v1).

## 13. Sources

- Model card: https://huggingface.co/openbmb/MiniCPM5-2B
- GitHub: https://github.com/OpenBMB/MiniCPM
- Fine-tune cookbooks on the card: TRL+PEFT, LLaMA-Factory,
  ms-swift, Unsloth
