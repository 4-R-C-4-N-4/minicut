# MiniCut in Rust, with a Verus-verified core

Status: Phases 0–1 implemented on branch `rust-port` (2026-11-24). Phase 3 is planned in
[plans/2026-11-24-rust-port-phase3.md](plans/2026-11-24-rust-port-phase3.md), with its verified core
spiked. Phases 2 and 4 are not started. Phase 1 has a
task plan in [plans/2026-11-24-rust-port-phase1.md](plans/2026-11-24-rust-port-phase1.md).

## Goal

Ship the `minicut` tool as one Rust binary. Its decision-making core
(what a valid edit is, what an edit means, how a request intent becomes
a label) is proved with Verus/Z3. The same compiled core produces the
grammar, validates model calls at run time, and labels training data.
Training (torch/peft) and eval grading stay in Python.

## Why

- Today the tool contract is written three times by hand:
  `config/tools.yaml`, `minicut/grammar/minicut.gbnf` and `validate.py`.
  The Dafny dictionary (closed PR #3) added a fourth copy that nothing executes.
  Here a single Rust enum defines the contract.
- The bugs that cost the most were label/request mismatches in training
  data (`training/build_pilot.py` filters mm:ss fraction loss,
  mismatched scale dims and unclear concat counts with regexes) and
  keep/remove inversions in model output. The verified intent→label
  function rules out that class of bug in generated data.
- Users get one binary with no Python. Runtime needs are ffmpeg/ffprobe
  and a llama.cpp server, as now.

## Feasibility (checked 2026-11-24 on this machine)

- Verus release `0.2026.11.22.1ce6a59` (x86-linux zip, bundles z3)
  needs rustup toolchain `1.100.0`. It runs only at check time.
- A spike core (`Edit`, `Clip`, validate ⇔ spec, kept-duration
  meaning, intent→label, mm:ss clock, atempo stages, clarify missing
  as bitmask) gives **16 verified, 0 errors** in under a second.
- **8 of 8 planted bugs were rejected**: validate missing the speed-1x
  no-op rule, a wrong atempo stage, an out-of-range atempo stage,
  clarify listing provided args, a keep/remove inversion, an
  off-by-one label, the clock dropping the fraction, and validate
  allowing odd dims.
- The same source, with `vstd = "=0.0.0-2026-11-22-0344"` from
  crates.io, builds and unit-tests with **stock stable Rust 1.99**.
  Verification is erased at build time, so shipping needs no Verus.
- `cargo verus verify` works on a normal cargo crate with
  `[package.metadata.verus] verify = true`.
- A prototype of the Phase 1 workspace (core + wire layer +
  `grammar`/`validate` subcommands) passed: the grammar rules equal the
  golden file, and the differential test against `validate.py` gave
  **20,000 sampled calls, 0 mismatches**, with both verdicts well
  represented for every tool.
- **Proofs and parity check different things.** Changing the no-op
  tolerance from 50 to 40 ms still verified, because spec and code
  share the constant. The differential test caught it (16
  mismatches). Verus proves code = spec; the parity test checks
  spec = shipped behaviour. Both are required.
- **Proofs can depend on their context.** A division in an `ensures` clause
  (atempo bound) verified only when an unrelated section followed it in the
  file. The bound is now division-free, and `scripts/mutate_core.sh` first
  checks that the unmutated core verifies standalone under five SMT seeds.
- Constraint found: code inside `verus! {}` cannot do I/O
  (`println!` fails verification). Verified code and I/O live in
  separate crates.

## What exists and is kept

| Area | Path | Fate |
|---|---|---|
| Runtime tool | `minicut/` (~860 lines) | Ported to Rust, retired at cutover (Phase 4) |
| Model client prompts | `minicut/llm/prompts.py` | Stays the training/inference source of truth until cutover; the Rust copy is checked byte-equal to it |
| Training | `training/` | Stays Python; reads Rust-generated JSONL from Phase 3 |
| Dev eval | `eval/run_eval.py`, `eval/eval_*.jsonl`, `eval/freeze.json` | Stays Python; frozen inputs untouched |
| Release benchmark | `eval/release/` | Stays Python; grades the Rust binary through `run_case(editor=...)` |
| Fixture factory | `factory/` | Stays Python (uses `colour_at` / `tone_at`) |
| Python tests | `tests/` | Kept; CLI/wrapper tests are pointed at the binary as black-box tests |

`minicut/verify.py` `verify_output` uses only ffprobe. `colour_at`,
`tone_at` and `mean_luma` are used only by `eval/release/bench.py` and
`factory/verify_fixtures.py`, so the Rust verifier needs no image or
FFT code. Those helpers move into a small Python module at cutover.

## Architecture

```
Cargo.toml                     workspace
crates/minicut-core/           #![no I/O] verus! { types, specs, exec, proofs }
crates/minicut/                bin `minicut`: wire JSON, file binding, ffprobe,
                               argv build, run, verify, publish, llama client,
                               CLI, `grammar` / `validate` / `label` subcommands
scripts/verus.sh               fetch the pinned Verus, run `cargo verus verify`
.github/workflows/ci.yml       python + rust + verus jobs
```

Data flow at run time: model JSON → `minicut::wire` (serde, unknown
fields rejected, seconds → ms) → bind files (the app picks paths; the
model never does) → `minicut_core::validate(Edit, Clip)` → argv →
ffmpeg → ffprobe verify → hard-link publish.

## Decided

1. **Integer units in the core.** Times are `u64` ms; speed is `u64`
   per-mille (0.25..4.0 → 250..4000). This removes float tolerances
   from the proofs and makes mm:ss rendering exact. Conversion
   happens once, in `wire`, by rounding to the nearest unit.
2. **Two crates.** `minicut-core` is entirely inside `verus!` and has
   no I/O, no strings and no floats. Everything else is ordinary Rust
   in `minicut`. The unverified surface is listed under "Trusted base".
3. **Wire types are separate from core types.** Serde derives stay
   outside `verus!`. `wire::to_edit` is the only conversion and is
   covered by differential tests against `validate.py`.
4. **The binary's grammar is PR #3's strict grammar.** It was
   validated with llama.cpp `test-gbnf-validator`: 1262/1262 training
   and eval targets accepted, 6/6 known-bad calls rejected (run on the
   author's machine; CI has no llama.cpp, so CI checks rule equality
   with the golden file, not parsing). Every rule
   line the Rust emitter writes must equal that file. Only the header
   comment differs.
5. **No-op policy is a separate predicate.** `well_formed` matches
   the Dafny/spec meaning and `changes_something` is the
   `validate.py` no-op rule. `accepted = both`. The disagreement PR
   #3 recorded becomes two named predicates, not an exception.
6. **Clarify `missing` is computed, never written by hand.**
   `missing(kind, provided)` is proved to equal required − provided.
   In the CLI, `input`/`inputs` always count as provided, because the
   app binds files. That matches the frozen labels (e.g. "convert it"
   → `["codec", "container"]`).
7. **The retry loop is dropped.** `wrapper.edit(llm=...)` is never
   called with `llm` anywhere in the repo (CLI, `eval/run_eval.py`,
   `training/check_sample.py`, `eval/release/bench.py`).
8. **Verus is pinned** to release `0.2026.11.22.1ce6a59` and `vstd`
   to `=0.0.0-2026-11-22-0344`. Upgrades are deliberate, one commit
   each.
9. **Shipping builds use stock stable Rust.** Verus runs in CI and in
   `scripts/verus.sh`, never in `cargo build`.
10. **The core's public API is total.** Verus checks `requires` only for
    verified callers, and the `minicut` app crate is ordinary Rust. So a
    precondition on a function the app can call is silently unchecked.
    Every `pub fn` in `minicut-core` therefore has no `requires` and
    reports failure as `Option`. Functions that need a precondition are
    `pub(crate)` behind a checked wrapper (e.g. `try_label`). Enforced by
    `scripts/check_core_api.py` (Phase 3, Task 3.1). Today it flags
    `label` and `atempo_stages`; the app calls neither yet.
11. **Training-data policy stays in Python.** The freeze rules (held-out
    templates, reserved fixture, eval collisions) live where they are
    already implemented and reviewed. Rust produces meaning; a Python
    gate may only drop rows, never change them.
12. **Templates are the review surface.** No proof can say that a phrase
    means an intent. Each template is reviewed once, with its intent kind
    and applicability rule, in a generated `docs/templates.md`, instead
    of auditing thousands of labels.

## What is proved, and what is trusted

Proved in Phase 1 (`scripts/verus.sh`: 21 verified, 0 errors; each
claim is backed by at least one killed mutant in `scripts/mutate_core.sh`):
- `validate(e, c) == accepted(e, c)` for every edit and clip.
- trim and cut_out on the same range partition the clip.
- Meaning of each single-input edit: speed keeps exactly
  ⌊duration / factor⌋, so faster is shorter and slower is never shorter.
  scale, convert, extract_audio and mute keep the full duration.
- `label` on its own does not promise an *accepted* edit: an intent
  outside the clip still gets a correctly-meaning label that `validate`
  rejects. Phase 3 makes `label` crate-private and adds `try_label`,
  whose `Some` result is proved accepted (plan Tasks 3.1–3.3).
- For every intent, `label(i)` keeps the right *amount*
  (`kept_ms == intent_kept`) and the right *instants*: at every source
  time t inside the clip, t survives the edit exactly when the request,
  stated in its own terms ("remove the last 5 s" keeps t < dur − 5 s),
  says it should. Keep intents label as `Trim` and remove intents as
  `CutOut`: the keep/remove contract.
- `to_clock(ms)` loses nothing: `m*60000 + s*1000 + f == ms`.
- atempo stages are each within 0.5–2.0×, and their product is within
  0.001× of the requested factor.
- clarify `missing` is exactly required − provided.

Not proved yet: concat's meaning (the sum of every input's duration)
needs all probed clips. `kept_ms(Concat)` is a placeholder that no lemma
uses; Phase 2 adds it with a `seq<Clip>`.

Phase 2 targets: argv structure per tool (e.g. trim emits `-ss start
-to end` from the same ms values), the publish name sequence being
injective and never equal to a source name, and concat ordering.

Trusted base (not proved; tested instead): template wording → intent
(human-reviewed); digit printing in `render.rs` (exhaustive round-trip
over 0–3,600,000 ms); the SplitMix64 RNG; ffmpeg/ffprobe behaviour;
parsing ffprobe JSON; the OS `link()` refusing to replace an existing
name; serde_json; the f64→ms/per-mille rounding in `wire`; path safety
checks on strings; the llama.cpp client; Verus and Z3 themselves.

## Phases

**Phase 0: groundwork (½ day).** Take the strict grammar's rule lines
from closed PR #3 (commit `b460cfd`) as a golden test file; nothing else
from that PR is carried over. Add the CI workflow, including the
existing Python job, which also fixes the missing CI that let PR #2
break main. Add `scripts/verus.sh` and gitignore `.verus/` and `target/`.

**Phase 1: verified core + contract surfaces (2–3 days).**
`minicut-core` with the proofs above. In `minicut`: `wire`, a
`minicut grammar` subcommand (rule-equal to the golden grammar), and a
`minicut validate --call JSON --probe JSON` subcommand. Differential
test: a Python script samples thousands of (call, probe) pairs on the
ms / per-mille grid, records `validate.py`'s verdict, and the Rust
integration test must agree on every one except the listed
divergences. Exit: `cargo verus verify` clean, mutation script kills
every mutant, differential parity 100%.

**Phase 2: the binary (3–4 days).** Probe, argv build (ported
template by template from `minicut/build.py`), run with timeout,
verify (ffprobe only), staging + hard-link publish, llama client,
CLI with the same flags and exit codes as `minicut/cli.py`, plus an
`apply --call JSON --workdir DIR` subcommand for graders. The CLI and
`apply` must drop the model's `output` field *before* validating, as
`cli.bind_files` and `wrapper.edit` do: the application owns output
names. `minicut validate` keeps checking `output`, because it mirrors
`validate.py`, which does (parity now samples `output`).
Parity gate: `python -m eval.release controls` passes when the editor
is the Rust binary (38/38 expected outcomes, all wrong-but-executable
and tampering controls caught), 200/200 dev oracle cases, and
`tests/test_cli.py` / `tests/test_wrapper.py` pass against the binary.

**Phase 3: verified label generation (3–5 days).** Planned in
[plans/2026-11-24-rust-port-phase3.md](plans/2026-11-24-rust-port-phase3.md).
Can land before Phase 2; from Phase 2's scope it needs only
`default_output`.
- An intent sampler plus a reviewed template table renders the request.
  The label comes only from the proved `try_label` / `missing` /
  `pct_of` / `scale_half` / `scale_quarter` / speed functions.
- A drop-only Python freeze gate applies the eval policy.
- Rows keep the pilot's exact format, so `training/` consumes them
  unchanged.
- Acceptance: zero `build_pilot` filter drops; 100% `validate.py`;
  100% llama.cpp grammar parse; tokenizer preflight; the Rust
  `kept_intervals` agrees with the release grader's
  `expected_timeline` on every trim/cut_out case.
- Exit experiment: a matched-size pilot on the verified set, against
  pilot v1 on the 200 development cases.

**Phase 4: cutover (1 day).** Re-baseline stock and pilot adapter
under the new grammar with the Rust binary. Move `colour_at`/`tone_at`
into `eval/media.py`. Retire `minicut/` runtime Python and the Python
console script, and update the README. Point the Python client at the
grammar the binary emits.

Total: about 2–3 weeks part-time. Phases 1 and 2 are the commitment.
Phase 3 is the payoff for training.

## Correctness rules (non-negotiable)

- Sources are never modified, and every successful edit is a new
  verified derivative (existing project rule).
- No shell, ever. argv vectors only, as now.
- Nothing in `minicut-core` is `#[verifier::external_body]` or
  `assume(...)` without a line in the trusted-base list above.
- A divergence from `validate.py` is either fixed or listed by name
  in the differential test; it is never silently skipped.
- Frozen eval inputs, `eval/freeze.json` and the release roster
  pins are not edited.

## Explicitly deferred

Multi-step edits / chains of tools in one request; new tools; relative
edits ("a bit faster"); a Rust port of `eval/` or `factory/`;
PyO3 bindings (the CLI and JSONL are enough); proofs about strings
(path safety, chain token text); Kani (Verus covers the core;
revisit only for bounded checks of the unverified layer).

## Open questions (default in brackets)

1. ~~PR #3 grammar half: merge or take as golden?~~ **Resolved:** PR #3
   is closed. Its strict grammar's rule lines are kept only as
   `crates/minicut/tests/golden/minicut.gbnf` (source commit `b460cfd`).
   The Python runtime keeps its current loose grammar until cutover.
2. Rounding at the wire boundary: seconds → nearest ms, factor →
   nearest 0.001. [Yes. The differential test samples on that grid,
   and sub-ms/sub-0.001 inputs are listed as a divergence.]
3. ffprobe duration → ms: floor or nearest? [Floor. Then
   `end_ms <= duration_ms` never accepts an end past the real clip.]
4. `extract_audio` on a silent input: keep parity (accepted, then
   verification fails) or reject up front? [Parity in Phases 1–2;
   tighten after cutover as its own change with its own test.]
5. Python `validate.py` lets non-concat tools carry an `inputs` key,
   and `json.loads` takes the last value for duplicate keys. The Rust
   wire layer rejects both. [Reject. Both are named divergences: the
   first in `scripts/parity_validate.py`, the second pinned by
   `crates/minicut/tests/validate.rs`. The grammar already forbids
   both shapes.]
6. Binary name during the transition: the Rust bin is `minicut`,
   which clashes with the Python console script if both are installed.
   [Keep `minicut`; tests call the cargo target path directly; the
   Python script is removed at cutover.]
