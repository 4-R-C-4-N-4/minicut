# MiniCut — Implementation Guide

**Version:** v0.1 — 2026-11-19
**Source spec:** `spec.md` (v0.1)
**Base model:** `openbmb/MiniCPM5-2B` — https://huggingface.co/openbmb/MiniCPM5-2B
**Runtime:** llama.cpp + LoRA (GGUF adapter) + GBNF grammar
**Status:** Guide only. No code in this guide is final until it passes the
factory eval in Phase 1–2.

This guide turns the spec into build order. Follow the phases in order.
Do not start Phase 4 (training) until Phase 1 (factory) and Phase 2
(wrapper) pass their acceptance criteria — the factory is the only thing
that can tell you whether the training worked.

## 0. Principles (non-negotiable)

1. **Model outputs JSON only.** Never raw ffmpeg / bash.
2. **Grammar enforces format.** GBNF in llama.cpp, not a system-prompt plea.
3. **Wrapper owns ffmpeg.** One deterministic template per tool. No string
   concatenation of user text into a command. No shell — argv list only.
4. **Probe before running.** Every arg is validated against `ffprobe`
   output for the actual input file.
5. **Verification is in the pipeline.** Exit code 0 is not success.
   Duration / dimensions / codec / streams must match the expected values.
6. **Ground truth is generated, not scraped.** The test factory creates
   the videos and the labels. No unverified Stack Overflow commands in
   training data.
7. **Baseline first.** Record the base model's factory score before any
   LoRA training. The fine-tune has to beat a number, not a feeling.

## 1. Repository layout

```
minicut/
  spec.md
  IMPLEMENTATION_GUIDE.md
  README.md                  # user-facing: install, run one edit, examples
  pyproject.toml             # Python 3.11+, dependencies: see §2
  config/
    tools.yaml               # tool registry: templates, arg ranges, allowlists
    v1_intents.yaml          # the 8 v1 tools, defaults, examples
  minicut/
    __init__.py
    probe.py                 # ffprobe wrapper -> Probe dataclass
    validate.py              # schema + semantic validation
    build.py                 # tool name + args -> ffmpeg argv list
    run.py                   # subprocess runner, timeout, stderr capture
    verify.py                # ffprobe + frame/audio checks vs expected
    wrapper.py               # orchestration: probe -> validate -> build
                             #   -> run -> verify -> retry (max 2)
    grammar/
      minicut.gbnf           # GBNF grammar for the JSON tool call
    llm/
      client.py              # llama.cpp server client (OpenAI-compatible)
      prompts.py             # system prompt, few-shot, retry prompt
  factory/
    generate.py              # fixture generator (lavfi only)
    manifest.py              # manifest JSON writer/reader
    verify_fixtures.py       # checks a generated fixture matches its manifest
    fixtures/                # generated, gitignored (large)
      <fixture_id>/
        input.mp4
        manifest.json
  data/
    prompts/                 # paraphrase templates per intent
    dataset_v1.jsonl         # generated training data (see §6)
    eval/
      eval_seen.jsonl
      eval_unseen_paraphrase.jsonl
      eval_unseen_combo.jsonl
      eval_unseen_fixture.jsonl
  eval/
    run_eval.py              # runs a model + wrapper over an eval set
    report.py                # metrics: tool/arg/exec/verify/retry/latency
  training/
    lora_config.yaml         # r, alpha, lr, epochs, mix (see §7)
    README.md                # exact commands used for the run, with hashes
  tests/
    test_wrapper.py          # hand-written JSON -> argv, no model
    test_factory.py
    test_verify.py
```

Generated fixtures, model weights, and LoRA adapters are gitignored.
Only code, configs, prompts, manifests (small JSON), and eval sets are
committed.

## 2. Prerequisites

- Python 3.11+
- ffmpeg 8.1.2+ and ffprobe on PATH (verified present in this environment:
  `ffmpeg -version` → 8.1.2)
- llama.cpp build with `llama-server` supporting `--grammar` / GBNF and
  LoRA adapter loading (`--lora`)
- Base weights: `openbmb/MiniCPM5-2B` in GGUF (for llama.cpp) or BF16
  (for training frameworks). Record the exact file + SHA256 in
  `training/README.md` for every run.
- GPU for training (LoRA on 2B is small — a single consumer GPU is
  enough; record VRAM used). Inference runs locally on CPU/GPU via
  llama.cpp.
- `pip install` from `pyproject.toml`: no heavy deps in the wrapper.
  Keep the wrapper dependency-free apart from stdlib + a JSON library;
  the factory/eval may use Pillow for pixel sampling.

Environment check (run first, record output):

```bash
python3 --version
ffmpeg -version | head -1
ffprobe -version | head -1
llama-server --version
```

## 3. Phase 0 — Baseline (no training, no factory yet)

**Goal:** prove the runtime path works end to end with the base model.

1. Serve the base model:
   ```bash
   llama-server -m MiniCPM5-2B-Q4_K_M.gguf \
     --jinja -c 8192 --port 8080 \
     --grammar-file minicut/grammar/minicut.gbnf
   ```
   Notes: set `min_p=0.0` (llama.cpp default 0.05 can cause repetition
   on this model), `temperature` low / greedy for tool calls, thinking
   off for v1.
2. Send one hand-written prompt, e.g. "trim clip.mp4 from 12 seconds
   to 45 seconds", and confirm the response parses as the §5 JSON.
3. Run the wrapper by hand with that JSON (Phase 2 code may not exist
   yet — a shell one-off is fine here, it is not the product path).
4. Record: does the base model, with grammar only, produce valid JSON
   on 10 hand-written prompts? This is a smoke test, not the baseline
   eval — the real baseline is Phase 3, after the factory exists.

**Acceptance:** server runs, grammar loads, 10/10 hand prompts return
parseable JSON. If grammar loading fails, stop — do not work around it
by dropping the grammar.

## 4. Phase 1 — Test factory (build this first)

**Goal:** a generator that produces verifiable videos + manifests, and
a verifier that can check any edit output against a manifest.

### 4.1 Fixture generation

Use ffmpeg `lavfi` only. No real footage, no downloads, deterministic
inputs. Every fixture is defined by a small parameter set:

```yaml
fixture_id: base_30s_1080p30
duration_s: 30
width: 1280
height: 720
fps: 30
audio: sine_440_880   # 440 Hz 0-15s, 880 Hz 15-30s; or none / sine_440
segments:
  - {start: 0,  end: 10, color: red,   tone_hz: 440}
  - {start: 10, end: 20, color: green, tone_hz: 880}
  - {start: 20, end: 30, color: blue,  tone_hz: 440}
```

Generation approach (one command per fixture, recorded in
`factory/generate.py`, not hand-run):

- Video source: `testsrc2=size=WxH:rate=FPS:duration=D` — burned-in
  frame counter / timestamp. This is what makes trim verification
  possible from a single sampled frame.
- Colour segments: build each segment as a `color` source, then
  `concat` them, or use `drawbox` overlays on testsrc2. Either is
  acceptable — pick one, document it, and use it for every fixture.
  Do not mix methods between fixtures.
- Audio: `sine=frequency=F:sample_rate=48000:duration=D` per segment,
  concatenated to match the video segments.
- Encode: `libx264`, `yuv420p`, `crf=18`, `preset=veryfast`, AAC audio
  where present. Fixed encode settings for every fixture — encode
  settings are not a variable in v1.
- Output: `factory/fixtures/<fixture_id>/input.mp4` + `manifest.json`.

Fixture matrix (v1):

| Dimension | Values |
|---|---|
| Duration | 10s, 30s, 60s |
| Resolution | 1280x720, 1920x1080 |
| FPS | 24, 30, 60 |
| Audio | none, sine 440, sine 440/880 segmented |
| Orientation | landscape, portrait (720x1280) |

Do not generate the full cross-product. Generate a curated set of
~12–20 fixtures covering each value at least twice, with the 30s /
720p / 30fps / segmented-audio fixture as the default workhorse.

### 4.2 Manifest schema

```json
{
  "fixture_id": "base_30s_1080p30",
  "duration_s": 30.0,
  "width": 1280,
  "height": 720,
  "fps": 30,
  "video_codec": "h264",
  "audio_codec": "aac",
  "has_audio": true,
  "segments": [
    {"start": 0, "end": 10, "color": "red", "tone_hz": 440},
    {"start": 10, "end": 20, "color": "green", "tone_hz": 880},
    {"start": 20, "end": 30, "color": "blue", "tone_hz": 440}
  ]
}
```

The manifest is the contract. The verifier (§4.3) and the dataset
generator (§6) both read it. If a fixture does not match its manifest,
the fixture is wrong — fix the generator, never edit the manifest by
hand to match a bad encode.

### 4.3 Verification functions

Implement in `minicut/verify.py`. Each function takes
`(output_path, expected)` and returns `pass/fail + reason`.
Tolerances are part of the spec — do not tighten them silently to make
a run look better.

| Check | Method | Tolerance |
|---|---|---|
| Duration | `ffprobe` format duration | ±0.1s (trim/cut/concat/speed) |
| Dimensions | `ffprobe` stream width/height | exact |
| Codec / container | `ffprobe` codec_name / format | exact, allowlist only |
| Audio present/absent | `ffprobe` stream count | exact |
| First-frame timecode | extract frame at t=0, read testsrc2 counter | ±1 frame |
| Colour at time t | extract frame at t, sample centre pixel (Pillow), nearest segment colour | nearest of red/green/blue, distance threshold recorded in code |
| Tone frequency | extract audio segment, FFT peak (or `astats` + FFT via stdlib/numpy in eval only) | ±20 Hz |
| Fade brightness | extract frame near start/end, mean luma | start < 10% of mid for fade-in, end < 10% for fade-out |

Per-tool expected values (what "verify" means for each v1 tool):

- `trim(start, end)` → duration = end−start ±0.1s; first-frame
  timecode ≈ start.
- `cut_out(start, end)` → duration = orig − (end−start) ±0.1s;
  colour at the cut point = colour of the segment after `end`.
- `concat` → duration = sum ±0.2s; colours in segment order.
- `scale` / `crop` / `rotate` → dimensions exact (rotate swaps w/h).
- `speed(f)` → duration = orig/f ±0.1s.
- `convert` → codec/container exact.
- `extract_audio` → audio file, duration ±0.1s, tone present.
- `mute` → no audio stream.
- `fade_in` / `fade_out` → brightness check above.

**Acceptance (Phase 1):** generator + verifier run on all fixtures with
hand-written expected outputs, 100% pass. A verifier that passes a
wrong output is worse than no verifier — deliberately feed it one
wrong output per tool (e.g. trim to the wrong time) and confirm it
fails. Record those negative checks in `tests/test_verify.py`.

## 5. Phase 2 — Wrapper + grammar (build this second)

**Goal:** hand-written JSON in → verified ffmpeg output out, with no
model involved. This proves the templates before the model is trusted
with them.

### 5.1 Tool-call JSON schema

One tool call per turn. Exact shape:

```json
{"tool": "trim", "input": "clip.mp4", "start": 12.0, "end": 45.0, "output": "clip_trim.mp4"}
```

Field rules:

- `tool`: string, one of the v1 enum (see §5.2). No other values.
- `input` / `output`: relative filenames under the workdir. No absolute
  paths, no `..`, no shell characters. The wrapper resolves them.
- Numeric args: numbers, not strings. Times in seconds (float).
  Dimensions in pixels (int). `speed.factor` float 0.25–4.0.
- Unknown keys are rejected. Missing required keys are rejected.
- `output` may be omitted — the wrapper then derives it
  (`clip_trim.mp4`). Record the derivation rule in `config/tools.yaml`
  and use it everywhere.

v1 tools and required args (full table in `spec.md` §5; this is the
implementation copy — keep the two in sync):

| Tool | Required args |
|---|---|
| `trim` | start, end |
| `cut_out` | start, end |
| `concat` | inputs[] |
| `scale` | width, height |
| `crop` | x, y, width, height |
| `rotate` | degrees (90/180/270) |
| `speed` | factor |
| `convert` | codec, container, crf |
| `extract_audio` | format (mp3/wav/aac) |
| `mute` | — |
| `fade_in` | duration |
| `fade_out` | duration |

Clarification is a separate output shape, decided in this phase (spec
§12 open question — resolve it here, before training data is written):

- **Option A (recommended):** a `clarify` tool call,
  `{"tool": "clarify", "question": "...", "missing": ["end"]}`.
  Keeps everything inside the grammar.
- **Option B:** plain-text reply outside the grammar. Simpler, but
  splits the output contract in two.

Pick A unless there is a concrete reason not to. Record the decision
in `config/tools.yaml` and in the spec.

### 5.2 GBNF grammar

File: `minicut/grammar/minicut.gbnf`. It must enforce:

- Top level is exactly one JSON object.
- `tool` is an enum of the v1 tools (+ `clarify` if Option A).
- Each tool's args are typed (number / integer / string-enum / array).
- No trailing text, no markdown fences, no explanation.

Write the grammar by hand, test it with `llama-server` on 20 prompts
(10 valid, 10 that should be forced into valid shape or clarified).
A grammar that is too loose is a bug — if the model can emit an
invented tool name, the grammar is wrong.

Keep the grammar generated from `config/tools.yaml` if practical
(a small script in `minicut/grammar/build_grammar.py`); otherwise keep
them in sync by hand and add a test that asserts the grammar's tool
enum equals the config's tool list.

### 5.3 Wrapper modules

**`probe.py`** — runs
`ffprobe -v quiet -print_format json -show_format -show_streams <file>`
and returns a dataclass: duration, width, height, fps, video codec,
audio codec, has_audio, sample rate. Raises a typed error on missing
file / unreadable file. No other module calls ffprobe directly.

**`validate.py`** — two stages:
1. Schema: JSON shape, tool enum, arg types, no unknown keys.
2. Semantic (needs the probe): `0 <= start < end <= duration`,
   crop rectangle inside the frame, scale dims even (yuv420p) and
   > 0, speed in range, codec/container in the allowlist,
   output inside the workdir, input != output unless explicitly
   allowed. Return a list of human-readable errors, not a boolean.

**`build.py`** — one function per tool, e.g. `build_trim(args) -> list[str]`.
Each returns an argv list, e.g.:

```python
["ffmpeg", "-y", "-ss", "12.0", "-to", "45.0",
 "-i", "clip.mp4", "-c", "copy", "clip_trim.mp4"]
```

Rules: templates live in `config/tools.yaml` where possible (command
skeleton + arg mapping); `build.py` fills them in. No f-string
concatenation of user text into a shell string — argv elements are
separate list items, always. Re-encode vs stream-copy per tool is a
template decision, recorded in the config with a reason
(e.g. trim with `-c copy` is fast but keyframe-bound; decide per tool
and note it — do not leave it implicit).

**`run.py`** — `subprocess.run(argv, capture_output=True, timeout=...)`.
Timeout per job from config (default 120s for a 60s fixture).
Returns exit code, stdout, stderr (truncated in logs, full in debug).

**`verify.py`** — the Phase 1 functions, reused here. Called on every
run, not just in tests.

**`wrapper.py`** — orchestration:

```python
def edit(tool_call: dict, workdir: Path, llm=None) -> EditResult:
    probe = probe_file(tool_call["input"])
    errors = validate(tool_call, probe)
    if errors: return EditResult(error=errors)   # or clarify
    argv = build(tool_call)
    result = run(argv)
    if result.exit_code != 0:
        return retry_or_fail(tool_call, result, llm)
    check = verify(tool_call, output_path)
    if not check.passed:
        return retry_or_fail(tool_call, check, llm)
    return EditResult(output=output_path, report=...)
```

Retry: max 2. On retry, the model gets the probe + stderr + the
failed tool call as a tool result, and must return a corrected JSON
call. The wrapper never edits the command itself. After 2 failures,
return the error plainly — do not loop, do not guess.

**Acceptance (Phase 2):** all 8 v1 tools run from hand-written JSON
on the default fixture, verified pass, 8/8. Plus: one invalid JSON
per tool is rejected before ffmpeg runs (validate, not ffmpeg, catches
it). Tests in `tests/test_wrapper.py`.

## 6. Phase 3 — Baseline eval (no training yet)

**Goal:** a number the fine-tune has to beat.

1. Write the eval harness first (`eval/run_eval.py`): takes an eval
   JSONL file, sends each prompt to the llama.cpp server (base model +
   grammar, thinking off), feeds the returned JSON to the wrapper,
   records tool accuracy, arg accuracy, execution success,
   verification pass rate, retry count, latency.
2. Write the eval sets (see §7.3 for the split — build the sets now,
   before training data, so training cannot leak into them).
3. Run the base model with a system prompt + few-shot examples
   (in `minicut/llm/prompts.py`) through the factory eval on ~50 cases.
4. Record the result in `eval/baseline_v1.md`: per-tool verification
   pass rate, overall pass rate, most common failure mode
   (wrong tool? wrong arg? invalid time? grammar escape?).
   This file is the benchmark. Do not overwrite it with later runs —
   new runs get new files.

**Acceptance:** harness runs end to end, baseline number recorded.
If the base model is already ≥80% verification pass on simple edits,
LoRA is for tightening format and edge cases. If below, LoRA is the
main lever. Either way, the decision is recorded, not assumed.

## 7. Phase 4 — Dataset (generated from the factory)

**Goal:** 2k–10k training examples where every label is ground truth.

### 7.1 Generation pipeline

```
factory fixture + manifest
  -> template: ground-truth JSON tool call (from manifest + chosen args)
  -> prompt paraphrases (5-10 per intent)
  -> JSONL example: {prompt, completion}
  -> negatives + clarifications
  -> split into train / eval (eval sets frozen in Phase 3)
```

1. **Ground truth first.** Pick an intent + args, compute the expected
   output from the manifest (e.g. trim 12–45 on a 30s fixture is
   invalid — the generator must only sample valid args). The JSON is
   the label; the prompt is derived from it, never the other way round.
2. **Paraphrases.** Hand-write 5–10 prompt templates per intent, then
   paraphrase with a model and human-check a sample. Every paraphrase
   must map to the same JSON — if a paraphrase changes the meaning
   (e.g. "cut the first 10 seconds" vs "trim to 10 seconds"), it is a
   different example with a different label, not a paraphrase of the
   first. These contrast pairs are the most valuable examples in the
   dataset — include them deliberately.
3. **Unit variety.** Times as seconds (`45`), `mm:ss` (`1:30`),
   words (`1m30s`, `90 seconds`), percentages (`first 10%` — resolved
   against the manifest duration at generation time, so the label is
   still absolute seconds). Resolutions as `720p`, `1280x720`, `half
   size`. The model learns normalisation; the label is always canonical.
4. **Negatives / clarifications (10–15% of data):**
   - Ambiguous: "trim it" (no times) → `clarify`, missing `["start", "end"]`.
   - Invalid: end > duration, crop outside frame → `clarify` or a
     corrected call? Decide per case at generation time and be
     consistent: invalid-against-probe → clarify, because the model
     does not see the probe in v1 prompts. (If the probe is later fed
     in as context, revisit this.)
   - Wrong-tool traps: the contrast pairs in (2).
5. **General mix.** ~10–20% general instruction data (short, varied),
   to limit catastrophic forgetting of the narrow JSON format. Record
   the source and proportion in `training/lora_config.yaml`.

### 7.2 Training example format

JSONL, one example per line. Chat template, thinking off, assistant
output = the JSON tool call only — no explanation, no preamble, no
markdown fence (the grammar enforces this at inference; training must
match it):

```json
{"messages": [
  {"role": "system", "content": "You are MiniCut. Output exactly one JSON tool call."},
  {"role": "user", "content": "trim the first 30 seconds off clip.mp4"},
  {"role": "assistant", "content": "{\"tool\": \"trim\", \"input\": \"clip.mp4\", \"start\": 0.0, \"end\": 30.0, \"output\": \"clip_trim.mp4\"}"}
]}
```

Exact system prompt text lives in `minicut/llm/prompts.py` and is the
same string used at training and at inference. If they differ, the
eval is measuring the wrong thing.

### 7.3 Eval splits

Build these in Phase 3, freeze them before training:

- `eval_seen.jsonl` — paraphrases seen in training, new arg values.
- `eval_unseen_paraphrase.jsonl` — paraphrases never in training.
- `eval_unseen_combo.jsonl` — arg combinations never in training
  (e.g. a trim range + a fixture never paired in training).
- `eval_unseen_fixture.jsonl` — fixtures never in training.

Report all four separately. A model that only passes `eval_seen` has
memorised, not learned.

Target size: ~50 cases per split for a quick eval, ~200 per split for
a full eval. Eval cases are generated by the same factory pipeline,
but from a held-out seed — never sample eval cases from the training
pool.

## 8. Phase 5 — LoRA training + eval

**Goal:** an adapter that beats the Phase 3 baseline on verification
pass rate, on the same held-out sets.

### 8.1 Training config (starting point, not gospel)

File: `training/lora_config.yaml`.

```yaml
base: openbmb/MiniCPM5-2B        # final RL+OPD checkpoint
method: LoRA
r: 16                             # sweep 16, 32 if eval says so
alpha: 32                         # 2x r
lr: 0.0001                        # sweep 1e-4, 2e-4
epochs: 2                         # sweep 2, 3
batch: as fits VRAM; record it
max_seq_len: 1024                 # prompts are short; don't waste context
general_mix: 0.15
thinking: off
```

Frameworks (all have official MiniCPM5 cookbooks linked from the model
card — use the cookbook, do not improvise the chat template):
TRL + PEFT, LLaMA-Factory, ms-swift, or Unsloth. Pick one, record it,
and use it for every run in v1 so runs are comparable. Convert the
resulting adapter to GGUF for llama.cpp inference and record the
conversion command.

What is being learned: slot-filling, unit normalisation, intent
disambiguation, and when to clarify. Not ffmpeg syntax — the wrapper
owns that. If training loss falls but verification pass rate does not
rise, the dataset is wrong (probably paraphrases leaking meaning, or
labels that do not match the manifests). Fix the data, not the LR.

### 8.2 Eval loop

1. Load adapter in llama.cpp (`--lora adapter.gguf`), same grammar,
   same prompts, same decoding as the baseline run.
2. Run `eval/run_eval.py` on all four splits.
3. Write `eval/lora_v1.md`: per-split verification pass rate, tool
   accuracy, arg accuracy, retry count, latency, and the three most
   common failure modes with example prompts.
4. Compare against `eval/baseline_v1.md`. Ship the adapter only if it
   beats baseline on verification pass rate on the unseen splits —
   not just on `eval_seen`.
5. If it does not beat baseline: inspect failures first. The usual
   suspects, in order: (a) dataset label errors, (b) grammar/config
   mismatch between train and inference, (c) too few contrast pairs,
   (d) LR/epochs. Change one thing per run and record it.

### 8.3 Acceptance (Phase 5 — v1 done)

- Verification pass rate on unseen paraphrase + unseen combo ≥ baseline
  + a margin recorded in `eval/lora_v1.md` (set the margin before the
  run, e.g. +10 points, not after seeing the result).
- Retry count lower than baseline.
- No regression on `eval_seen`.
- 8/8 tools still pass the hand-written wrapper tests (Phase 2).
- Adapter + conversion command + training config committed; weights
  gitignored, with a download/checksum note in `training/README.md`.

## 9. Milestones and order

| # | Milestone | Depends on | Done when |
|---|---|---|---|
| 0 | Baseline smoke test | — | §3 acceptance |
| 1 | Factory v1 | — | §4.3 acceptance |
| 2 | Wrapper v1 | 1 | §5.3 acceptance |
| 3 | Baseline eval | 1, 2 | §6 acceptance |
| 4 | Dataset v1 | 1 | §7, 2k+ examples, eval splits frozen |
| 5 | LoRA v1 | 3, 4 | §8.3 acceptance |
| 6 | v2 (later) | 5 | multi-step plans, subtitles/overlay, silence removal, batch — each as its own spec addendum, not scope creep in v1 |

Do not parallelise 1 and 2 by skipping the verifier's negative checks,
and do not start 5 before 3 has a recorded number.

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Model hallucinates a plausible-but-wrong command | Grammar + wrapper templates + verification. The model never writes ffmpeg syntax. |
| Paraphrase changes meaning, label is wrong | Contrast pairs are hand-checked; dataset generator asserts the label against the manifest. |
| Eval leaks into training | Eval sets frozen in Phase 3 from a held-out seed; never sample eval from the training pool. |
| Keyframe-bound trim with `-c copy` gives wrong duration | Template decision per tool, recorded in config; verifier catches it (duration ±0.1s). If it fails, switch that tool to re-encode and re-run Phase 2. |
| LoRA overfits to JSON format, forgets clarification | General mix 10–20%; clarification examples 10–15%; eval includes ambiguous prompts. |
| Grammar and training format drift apart | Same system prompt + same JSON shape in training and inference; test asserts grammar enum == config tool list. |
| Scope creep into v2 (multi-step, filters) | v1 is 8 single-step tools. New intents go in a v2 addendum with their own factory fixtures. |

## 11. Decisions log

Record decisions here as they are made, with date. Do not leave them
in chat only.

- 2026-11-19: Project named MiniCut. Base model MiniCPM5-2B.
  Runtime llama.cpp + LoRA + GBNF grammar. Model outputs JSON only.
- 2026-11-19: Test factory first, wrapper second, baseline before
  training. (This guide.)
- 2026-11-19: Factory + wrapper v1 built and tested. 5 fixtures pass
  manifest checks (colour + tone). Wrapper: 8/8 tools + clarify pass,
  6/6 invalid calls rejected before ffmpeg. Negative verify check
  passes (wrong trim duration correctly fails). Factory uses solid
  colour segments, not testsrc2 — centre-pixel colour check is
  unambiguous; testsrc2 timecode is a v2 refinement.
- 2026-11-19: No git repo yet — remote to be decided after this build.
- 2026-11-20: Baseline v1 done (F16, llama.cpp commit 5f83714, grammar
  enforced, thinking off). 200/200 cases: verification 55.0%, tool
  86.0%, arg 48.0%, mean latency 15.4s. Per split: seen 52%, unseen
  paraphrase 54%, unseen combo 58%, unseen fixture 56%. The gap is
  schema, not intent: speed always uses "speed" not "factor" (24/24),
  concat uses "input" not "inputs", convert nests codec in
  options/params, extract_audio omits "format". Scale 100%, cut_out
  96%, trim 94% — the tools with few-shot examples work. Whenever the
  model produced the exact expected call, the wrapper verified it
  (0 failures). Harness patched for the run: sends
  chat_template_kwargs enable_thinking=false (thinking on puts the
  JSON in reasoning_content and leaves content empty) + falls back
  to reasoning_content if content is empty. Results in
  eval/results/baseline_v1_*.md. A runtime restart killed the first
  run mid-split-3; splits 1–2 were already complete and were not
  re-run, splits 3–4 were re-run after a server warmup.
- 2026-11-19: v1 intent list cut to 8 tools — trim, cut_out, concat,
  scale, speed, convert, extract_audio, mute. Crop, rotate, fade_in,
  fade_out move to v2.
- 2026-11-19: clarification is a tool call (Option A):
  `{"tool": "clarify", "question": "...", "missing": ["end"]}`.
  Decided before dataset generation.
- 2026-11-20: Phase 3 eval prep built before any training data:
  `eval/generate_eval.py` (seed 20261120, deterministic) + 4 frozen
  splits x 50 cases (`eval_seen`, `eval_unseen_paraphrase`,
  `eval_unseen_combo`, `eval_unseen_fixture`) + `eval/run_eval.py`
  harness + `eval/freeze.json` with the exclusion rules Phase 4 must
  obey (held-out `*_hN` templates, `long_60s_1080p` fixture, combo arg
  pools). Oracle sanity run: 200/200 ground-truth calls execute and
  verify — that is a harness check, NOT the baseline.
- 2026-11-20: Bug found while building eval — `convert` to WebM failed
  because the wrapper wrote AAC audio into a WebM container (WebM
  allows only Vorbis/Opus). Fixed in `minicut/build.py`: WebM uses
  `libopus`, mkv/mp4 keep AAC. Verified vp9/webm passes on audio and
  silent fixtures; wrapper tests still pass.
- 2026-11-20: Eval filename convention — every prompt says `clip.mp4`,
  matching the few-shot examples in `minicut/llm/prompts.py`; the
  harness copies the fixture to `clip.mp4` per case.
- 2026-11-20: Two more bugs found by the oracle sanity run. (a) AAC
  extract wrote raw `.aac` (ADTS), whose ffprobe duration reads ~0.35s
  short and failed verification — `extract_audio` with format `aac`
  now writes `.m4a` (same codec, proper container, duration correct).
  (b) A generated `cut_out 0–10` on the 10s fixture removed the whole
  video and ffmpeg could not concat nothing — the generator now
  requires a cut_out to leave ≥1s, and the combo pool gained a (2, 8)
  entry valid on the 10s fixture. Eval sets regenerated and the oracle
  sanity re-run certifies the new frozen files.
- _Open:_ baseline run itself — needs a MiniCPM5-2B GGUF + llama-server
  (not installed here). Harness `--backend http` is written but
  untested against a live server; confirm the server accepts the
  `grammar` field on the first real run, or pass `--no-grammar`.
- _Open:_ LoRA rank / LR — from a small sweep in Phase 5, not from
  this guide.
- _Open:_ LoRA rank / LR — from a small sweep in Phase 5, not from
  this guide.
- 2026-11-20: Baseline v2 done — same model/server/grammar, with 4
  few-shot examples added for concat, speed, convert, extract_audio
  (exact field names: inputs, factor, codec/container/crf, format).
  Verification 55.0% -> 94.5% (189/200), tool 86% -> 91%, arg
  48% -> 80%, mean latency 15.4s -> 21.5s (longer prompts).
  Per split: seen 52% -> 98%, paraphrase 54% -> 92%, combo
  58% -> 92%, fixture 56% -> 96%. Targeted tools: speed 0% -> 100%,
  concat 12% -> 100%, convert 12% -> 96%, extract_audio 0% -> 95%.
  Remaining 11 failures: clarify's exact `missing` list (4),
  cut_out start/end inversion on paraphrased prompts (4), one
  trim/cut_out confusion, and 2 execution failures from model-chosen
  output filenames bypassing wrapper defaults (.vp9 extension, raw
  .aac duration). Results in eval/results/baseline_v2_*.md. A runtime
  restart killed the run mid-split-2; split 1 was already saved and
  splits 2–4 were resumed after a server restart + warmup.
- 2026-11-20: Saving methodology decided — edit history lives in file
  metadata, NOT the filename (filename-as-history rejected) and NO
  sidecar JSON. Scheme: `minicut_chain` tag via ffmpeg -metadata,
  compact tokens e.g. `trim:12-45;speed:2.0`; probe reads it, wrapper
  appends and writes it to the new file; filename stays short (stem +
  one op, _v2 bump on collision, never overwrite, never modify input
  in place). Caveats: metadata is invisible without an inspect
  command and can be stripped by other tools/uploads — then the chain
  restarts from that op. Built 2026-11-20: `minicut/chain.py`
  (tokens + append + read), `Probe.chain` from ffprobe tags
  (case-insensitive), `build()` stamps `-metadata minicut_chain=...`
  on every template (+ `-movflags use_metadata_tags` for .mp4/.m4a,
  without which the mov muxer silently drops custom keys — found by
  a failing test, not by reading docs), wrapper derives the output
  name and ignores the model's `output` field entirely, `_v2` bump
  on collision, never overwrites, never modifies an input in place.
  Tests in `tests/test_chain.py`: tokens, trim->speed chain
  `trim:0-20;speed:2`, model-chosen `.vp9` output ignored, second
  identical edit bumps to `_v2`. WAV is the one format that drops
  the tag, so a WAV extract restarts the chain. Baseline_v3 re-run
  follows, since the wrapper now owns output names.

## 12. Sources

- Model card: https://huggingface.co/openbmb/MiniCPM5-2B
- GitHub: https://github.com/OpenBMB/MiniCPM
- Fine-tune cookbooks on the card: TRL+PEFT, LLaMA-Factory,
  ms-swift, Unsloth
- Spec: `spec.md`
