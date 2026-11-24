# Rust port, Phase 3: verified training-label generation

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Generate MiniCut training rows whose labels come only from the Verus-verified core. A sampled
*intent* ("remove the last 5 s") is rendered to text through a reviewed template table and labelled by the
proved `try_label`. Python keeps the freeze policy and training unchanged.

**Architecture:** Meaning first, label computed, policy in Python.

```
intent sampler (Rust, seeded) ──► template table (reviewed) ──► user text
        │
        └──► minicut_core::try_label / missing / pct_of / scale_* / speed_*  (proved)
                   │
                   └──► canonical JSON (grammar field order, + output name) ──► assistant text
rows.jsonl + meta.jsonl ──► training/build_verified.py (freeze gate, drop-only) ──► train_verified_v1.jsonl
                         ──► scripts/accept_dataset.py (5 acceptance checks)    ──► training/train_lora.py (unchanged)
```

**Tech stack:** Rust stable (build), Verus `0.2026.11.22.1ce6a59` (check only), serde_json, no new crates
(the RNG is SplitMix64, ~10 lines). Python 3.12 for the freeze gate and acceptance checks.

**Provenance:** the core code in Tasks 3.1–3.10 was spiked on 2026-11-24 against the PR #4 core.
- **Verified:** 35 verified, 0 errors under **10 SMT seeds** (0, 1, 2, 3, 5, 7, 11, 13, 17, 19).
- **Mutants:** 10/10 new data-generation mutants killed.
- **Stock Rust:** a smoke test on 1.99 passes.
- **Boundary checker:** `scripts/check_core_api.py` was run. On the current core it flags `label` and
  `atempo_stages` (see Task 3.1); on the spiked core it passes, with 12 pub fns checked.
- **Not yet run:** the app-side tasks (3.11–3.21) have interfaces, tests and acceptance commands but were
  not executed. Treat their code as a specification.

---

## Why (grounded in the current pipeline)

- `training/generate_dataset.py` samples the *label* first (`_sample_args`, `_sample_times`), then picks a
  phrasing (`_render_times`). Nothing checks that the phrasing means the label.
- That is how v1 shipped mismatches, which `training/build_pilot.py` now filters by regex: percent and "half"
  rows, mm:ss rows with fractions, scale rows whose prompt lacks the dimensions, and 3-input concats not
  saying "three".
- The mm:ss bug has an exact source: `eval/generate_eval.py:67` `_mmss` does `divmod(int(seconds), 60)`,
  which drops the fraction.
- The tables also carry wording traps:
  - `trim_t4` "cut X down to 5-12 seconds" means *keep*.
  - `cut_t4` "delete the middle 5 to 12 seconds" is false when start = 0.
  - v1's "trim the first 25%" was labelled keep-first.
  - `CLARIFY_AMBIG` answers "cut clip.mp4" with the *trim* question.

## Decisions (P3-D1 … P3-D10; they extend docs/rust-port.md "Decided")

1. **P3-D1 Meaning first.** Every tool label is `minicut_core::try_label(intent, clip)` or one of the proved
   value constructors (`pct_of`, `scale_half`, `scale_quarter`, `speed_percent_faster`, `speed_at_percent`).
   No app code builds an `Edit` for training by hand.
2. **P3-D2 The core API is total.** Verus checks `requires` only for verified callers, and the app crate is
   ordinary Rust. So every `pub fn` in `minicut-core` has no `requires` and reports failure as `Option`.
   Functions that need a precondition are `pub(crate)` behind a checked wrapper. This is enforced by
   `scripts/check_core_api.py` in CI.
3. **P3-D3 Freeze policy stays in Python, drop-only.** `training/build_verified.py` reuses
   `generate_dataset._load_eval_collisions` / `_canon` and `eval/freeze.json`:
   - held-out template IDs, the reserved fixture `long_60s_1080p`, combo arg pools, and
     (prompt, label) / (template_id, label) collisions with the frozen eval;
   - it may drop a row, never change one (byte comparison, asserted).
4. **P3-D4 The template table is the human-review surface.** It lives in `crates/minicut/src/templates.rs`:
   - Each row has an id, exactly one intent kind, the wording, an applicability predicate
     (e.g. "middle" needs 0 < start and end < duration), and a reviewer note.
   - `docs/templates.md` is generated from it, and a test keeps them equal.
   - Ambiguous phrasings are *not* in the table; they are listed in the doc's "excluded" section.
   - Existing train ids are reused only with their exact v1 wording. New ids start with `v3_`. Held-out ids
     never appear.
5. **P3-D5 Numbers render only through proved functions.**
   - seconds via `to_secs`;
   - mm:ss via `to_clock`, with `.fff` when the fraction is non-zero, which fixes `_mmss`;
   - percent only when `pct_of` is exact; relative scale only via `scale_half` / `scale_quarter`;
   - speed percent via the two speed functions.
   Printing digits is not proved; an exhaustive render→parse test covers 0–3,600,000 ms.
6. **P3-D6 Same row format as the pilot.** `{"messages": [system, user, assistant]}`.
   - `system` is byte-equal to `minicut/llm/prompts.py` `SYSTEM_PROMPT` (tested). `train_lora.py` rejects
     anything else and adds the few-shot prefix itself.
   - The assistant JSON uses the grammar's canonical field order and keeps `output` from a Rust port of
     `minicut/build.py:default_output` (parity-tested), matching the pilot targets and the few-shot examples.
7. **P3-D7 Reproducible bytes.** A SplitMix64 RNG in the app crate and a fixed iteration order make
   seed → identical bytes on any machine. The dataset SHA-256 goes into `training/README.md`, as for the pilot.
8. **P3-D8 Scope.** Tool rows for all 8 tools plus ambiguous-request clarify rows. General Q&A and
   invalid-against-manifest clarify stay out (the pilot excluded both). Concat labels carry order and count
   only (its duration meaning is not proved yet).
9. **P3-D9 Proof stability over 10 seeds.** `scripts/mutate_core.sh` checks seeds 0–9.
   In the spike, a version of `scale_by_div` passed seeds 0, 2 and 3 and failed 1 and 7.
10. **P3-D10 Phase order.** Phase 3 can land before Phase 2. From Phase 2's scope it needs only
    `default_output` (Task 3.13).
11. **P3-D11 Ambiguous requests are rejected with a clarify that suggests our own vocabulary** (Ivy,
    2026-11-24; this replaces "exclude ambiguous phrasings"). Never a guessed label. Two classes:
    - **Reading-ambiguous:** every argument is present but the operation is not, e.g. "trim the first 25%
      of clip.mp4". The label is `clarify` with `missing: []`, which is what the proved `missing` gives
      when both readings are fully specified. The question offers both readings as canonical phrasings
      with the user's own numbers: *"Do you want to keep only the first 7.5 seconds of clip.mp4, or cut
      the first 7.5 seconds off?"* The second reading comes from the proved `flip`, and
      `readings_disagree` proves the two readings disagree at every instant of every clip.
    - **Argument-missing:** the existing class, e.g. "cut clip.mp4". `missing` comes from the proved
      `missing()`, and the question suggests a canonical phrasing for each candidate tool: *"What should
      I do with clip.mp4: keep only a part (e.g. "keep only 5 to 12 seconds") or remove a part (e.g. "cut
      out 5 to 12 seconds")? Tell me the start and end."*
    - Suggestions come from a **vocabulary table**: one canonical phrasing per intent kind, each of which
      must itself render through an unambiguous template of that kind (test). Graders compare only
      `missing` (`eval/run_eval.py:_args_correct`, `eval/release/bench.py`), so the question text is free
      to carry suggestions without changing any score.

12. **P3-D12 Labels must follow from the text alone** (found while landing). The model sees only the
    request (`cli.compose_prompt`), never the probe. Last-N, percentage and relative-size labels are
    *correct* but need the clip's duration or size, so the same text would carry different labels on
    different clips. `Template::needs_probe()` marks them; tool rows never use them. Ambiguity questions
    repeat the user's terms (a percentage stays a percentage). Their proofs stay, for a future
    intent-level output that the app resolves against the probe.
13. **P3-D13 Matched size via oversample + targets.** The gate applies every freeze drop, then keeps the
    first N rows per tool (`training/verified_targets.json`) and fails if any tool is short. Freeze rules
    stay in Python. The combo-pool rule (`eval/freeze.json`) was missing from the original gate brief and
    is now enforced.
14. **P3-D14 v1 heuristics are not waived, they are re-proved per row.** `build_pilot`'s `mmss_fraction`
    and `scale_dims_not_in_prompt` exist because v1's renderer lost information. The acceptance check
    exempts a row only when a per-row check shows the prompt states the label (clock text round-trips to
    start/end; "540p" == 960x540), and it names every exemption.

## Open questions (default in brackets)

1. ~~"trim the first 25%": exclude?~~ **Resolved (P3-D11):** reading-ambiguous wording becomes a clarify
   row offering both readings in our vocabulary.
2. Row counts. [Match the pilot's tool mix and size: trim 306, cut_out 368, concat 223, scale 255, speed
   300, convert 193, extract_audio 244, mute 120, clarify 310 (2,319 total). The clarify 310 splits into 250
   argument-missing and 60 reading-ambiguous (P3-D11).]
3. Eval-pool disjointness in the sampler or only in the gate? [Gate only, per P3-D3. If the gate drops more
   than 20% of any tool, export the pools from `eval/freeze.json` as a sampler input file.]
4. Keep `output` in targets? [Yes, as P3-D6. The grammar allows it, the few-shot examples show it, and the
   pilot has it.]
5. Verb-trap templates such as `trim_t4` "cut X down to …". [Keep them, reviewed as KeepRange. Real users
   say this, and the frozen eval contains it.]

## Patterns to copy

- Collision and freeze checks: `training/generate_dataset.py` `_canon`, `_load_eval_collisions`, `add()`.
- Pilot filters to turn into a predicate: `training/build_pilot.py` `build()`, the inline `continue` branches.
- Row shape and preflight: `training/train_lora.py` `prepare_rows` (line 46) and `--preflight`.
- Release grader meaning: `eval/release/bench.py:236` `expected_timeline`.
- Verified-function style, mutants, boundary rule: `crates/minicut-core/src/lib.rs`, `scripts/mutate_core.sh`.

## Done when

- `scripts/verus.sh` → ≥ 35 verified, 0 errors.
- `scripts/check_core_api.py` → exit 0.
- `scripts/mutate_core.sh` → stable under seeds 0–9, all mutants (24) killed.
- `cargo test --workspace`, `cargo fmt --check` → pass; the exhaustive render round-trip passes.
- `minicut label --seed 20261124 …` → rows; `training/build_verified.py` → drop counts recorded,
  no row modified.
- `scripts/accept_dataset.py` → all five checks pass. The `pilot_drop_reason` refactor reproduces
  `train_pilot_v1.jsonl` byte-identically (`5f7d9b1d…fecca02`).
- `python -m training.train_lora --data training/train_verified_v1.jsonl --preflight` → every row prepared.

---

## Verified core (Tasks 3.1–3.10)

### Task 3.1: Make the core API total

**Objective:** No `pub fn` in the core has a `requires`; the rule is checked in CI.

**Files:** Create `scripts/check_core_api.py`. Modify `crates/minicut-core/src/lib.rs` (`label` and
`atempo_stages` become `pub(crate)`) and `.github/workflows/ci.yml` (verus job: add
`python3 scripts/check_core_api.py`).

**Step 1 (red):** add the script, then run it. Expected on the current core:
`line 205: pub fn label has \`requires\``, `line 242: pub fn atempo_stages has \`requires\``, exit 1.

```python
"""Fail if a `pub fn` in the verified core has a `requires` clause.

Verus checks `requires` only for verified callers. The `minicut` app crate
is ordinary Rust, so a precondition on a function it can call is silently
unchecked there. Every core function the app can reach must therefore be
total: no `requires`, with failure expressed as `Option` in its `ensures`.
Functions that need a precondition stay `pub(crate)` (or private) and are
reached through a checked wrapper such as `try_label`.

    python scripts/check_core_api.py [crates/minicut-core/src/lib.rs]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

DEFAULT = Path(__file__).resolve().parents[1] / "crates/minicut-core/src/lib.rs"
# `pub fn` / `pub const fn` exactly: pub(crate), proof fn and spec fn are fine.
PUB_EXEC_FN = re.compile(r"^\s*pub\s+(?:const\s+)?fn\s+(\w+)", re.M)


def offenders(src: str) -> list[str]:
    bad = []
    for m in PUB_EXEC_FN.finditer(src):
        body_start = src.find("{", m.end())
        signature = src[m.end():body_start]
        if re.search(r"\brequires\b", signature):
            line = src.count("\n", 0, m.start()) + 1
            bad.append(f"line {line}: pub fn {m.group(1)} has `requires`")
    return bad


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else DEFAULT
    src = path.read_text().split("} // verus!")[0]
    bad = offenders(src)
    for b in bad:
        print(f"{path}: {b}")
    if not bad:
        print(f"{path}: every pub exec fn is total ({len(PUB_EXEC_FN.findall(src))} checked)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
```

**Step 2:** change `pub fn label(` → `pub(crate) fn label(` and `pub fn atempo_stages(` →
`pub(crate) fn atempo_stages(`. Phase 2 gets a total `try_atempo` when it needs one; nothing outside the
core calls `atempo_stages` today. Run the script again. Expected: `every pub exec fn is total`, exit 0.

**Step 3:** `scripts/verus.sh` → 0 errors. `cargo test --workspace` → pass (core tests are in-crate).

**Commit:** `core: total public API; label and atempo_stages are crate-private (check_core_api.py)`

### Task 3.2: `label` names its result

**Objective:** expose the exact label as a spec function so `try_label` can explain a `None`.

Add `labelable` and `label_spec` (the first two items of Task 3.3's block) and one ensures line on `label`:

```rust
    ensures
        e == label_spec(i, c),
        kept_ms(e, c) == intent_kept(i, c),
```

`scripts/verus.sh` → 0 errors.
**Commit:** `core: label_spec names label's result`

### Task 3.3: `try_label`, the only labelling entry point

**Objective:** total and checked. `Some` carries every label guarantee plus `accepted`. `None` happens
exactly when the intent is out of range or its label would be rejected.

```rust
// Total, checked entry point for unverified callers. Verus does not enforce
// `requires` across the verus! boundary, so app code must never call `label`
// directly. None exactly when the request cannot be labelled, or when its
// label would be rejected by the runtime validator.
pub open spec fn labelable(i: Intent, c: Clip) -> bool {
    match i {
        Intent::KeepLast { ms } => ms <= c.duration_ms,
        Intent::RemoveLast { ms } => ms <= c.duration_ms,
        _ => true,
    }
}

// Spec-level label, so a None result can be explained without exec code.
pub open spec fn label_spec(i: Intent, c: Clip) -> Edit {
    match i {
        Intent::KeepRange { start_ms, end_ms } => Edit::Trim { start_ms, end_ms },
        Intent::RemoveRange { start_ms, end_ms } => Edit::CutOut { start_ms, end_ms },
        Intent::KeepFirst { ms } => Edit::Trim { start_ms: 0, end_ms: ms },
        Intent::RemoveFirst { ms } => Edit::CutOut { start_ms: 0, end_ms: ms },
        Intent::KeepLast { ms } => Edit::Trim { start_ms: (c.duration_ms - ms) as u64, end_ms: c.duration_ms },
        Intent::RemoveLast { ms } => Edit::CutOut { start_ms: (c.duration_ms - ms) as u64, end_ms: c.duration_ms },
    }
}

pub fn try_label(i: Intent, c: Clip) -> (r: Option<Edit>)
    ensures
        match r {
            Some(e) => labelable(i, c) && e == label_spec(i, c) && accepted(e, c)
                && kept_ms(e, c) == intent_kept(i, c)
                && (forall|t: int| 0 <= t < c.duration_ms ==> #[trigger] survives(e, t) == intent_survives(i, c, t))
                && (is_keep(i) <==> (e is Trim)),
            None => !labelable(i, c) || !accepted(label_spec(i, c), c),
        },
{
    let ok = match i {
        Intent::KeepLast { ms } => ms <= c.duration_ms,
        Intent::RemoveLast { ms } => ms <= c.duration_ms,
        _ => true,
    };
    if !ok {
        return None;
    }
    let e = label(i, c);
    if validate(e, c) { Some(e) } else { None }
}
```

**Red first:** write it with body `{ None }`. Verus fails the `None =>` branch, because `None` is not
explained for labelable, accepted intents. Then use the real body. → 0 errors.
**Commit:** `core: try_label, total checked labelling entry point`

### Task 3.4: exact percentages

```rust
// "the first 25%": a percentage renders only when it is an exact ms count.
pub fn pct_of(duration_ms: u64, pct: u64) -> (r: Option<u64>)
    ensures
        match r {
            Some(m) => pct <= 100 && m * 100 == duration_ms * pct,
            None => pct > 100 || (duration_ms * pct) % 100 != 0,
        },
{
    if pct > 100 {
        return None;
    }
    proof {
        assert((duration_ms as int) * (pct as int) <= (duration_ms as int) * 100) by (nonlinear_arith)
            requires pct <= 100, duration_ms >= 0;
    }
    let p: u128 = duration_ms as u128 * pct as u128;
    if p % 100 == 0 {
        let m = p / 100;
        proof {
            assert(m * 100 == p);
            assert(m <= duration_ms) by (nonlinear_arith)
                requires m * 100 == p, p <= (duration_ms as int) * 100;
        }
        Some(m as u64)
    } else {
        None
    }
}
```

The `u128` product and the two `nonlinear_arith` hints are required. Without them Verus reports a possible
overflow and an unproved postcondition (seen in the spike). → 0 errors.
**Commit:** `core: pct_of renders a percentage only when it is an exact ms count`

### Task 3.5: relative scale

```rust
// "half size" / "a quarter of its size": exact, and still even (yuv420p).
// Constant factors with bit-vector proofs: a variable divisor, and then
// plain nonlinear reasoning, were seed-dependent (rlimit under some seeds).
pub fn scale_half(w: u32, h: u32) -> (r: Option<(u32, u32)>)
    ensures
        match r {
            Some((rw, rh)) => rw * 2 == w && rh * 2 == h && rw % 2 == 0 && rh % 2 == 0 && rw > 0 && rh > 0,
            None => !(w > 0 && h > 0 && w % 4 == 0 && h % 4 == 0),
        },
{
    if w > 0 && h > 0 && w % 4 == 0 && h % 4 == 0 {
        let (rw, rh) = (w / 2, h / 2);
        proof {
            assert(rw * 2 == w && rw % 2 == 0 && rw > 0) by (bit_vector)
                requires rw == w / 2, w % 4 == 0, w > 0;
            assert(rh * 2 == h && rh % 2 == 0 && rh > 0) by (bit_vector)
                requires rh == h / 2, h % 4 == 0, h > 0;
        }
        Some((rw, rh))
    } else {
        None
    }
}

pub fn scale_quarter(w: u32, h: u32) -> (r: Option<(u32, u32)>)
    ensures
        match r {
            Some((rw, rh)) => rw * 4 == w && rh * 4 == h && rw % 2 == 0 && rh % 2 == 0 && rw > 0 && rh > 0,
            None => !(w > 0 && h > 0 && w % 8 == 0 && h % 8 == 0),
        },
{
    if w > 0 && h > 0 && w % 8 == 0 && h % 8 == 0 {
        let (rw, rh) = (w / 4, h / 4);
        proof {
            assert(rw * 4 == w && rw % 2 == 0 && rw > 0) by (bit_vector)
                requires rw == w / 4, w % 8 == 0, w > 0;
            assert(rh * 4 == h && rh % 2 == 0 && rh > 0) by (bit_vector)
                requires rh == h / 4, h % 8 == 0, h > 0;
        }
        Some((rw, rh))
    } else {
        None
    }
}
```

**Pitfall (from the spike):** one function with a variable divisor `d` passed seeds 0, 2 and 3 and hit the
resource limit under seeds 1 and 7. Two constant-factor functions with `by (bit_vector)` facts are stable
under 10 seeds. → 0 errors under seeds 0–9.
**Commit:** `core: scale_half / scale_quarter, exact and even, bit-vector proofs`

### Task 3.6: speed percentages

```rust
// "25% faster" -> 1.25x and "at 80% speed" -> 0.8x, in exact per-mille.
// None when the result would fall outside the validator's 0.25x..4x.
pub fn speed_percent_faster(p: u64) -> (r: Option<u64>)
    ensures
        match r {
            Some(f) => f == 1000 + 10 * p && 1000 < f <= 4000,
            None => p == 0 || p > 300,
        },
{
    if p == 0 || p > 300 { None } else { Some(1000 + 10 * p) }
}

pub fn speed_at_percent(p: u64) -> (r: Option<u64>)
    ensures
        match r {
            Some(f) => f == 10 * p && 250 <= f <= 4000,
            None => p < 25 || p > 400,
        },
{
    if p < 25 || p > 400 { None } else { Some(10 * p) }
}
```

**Commit:** `core: speed percentages in exact per-mille`

### Task 3.7: seconds text

```rust
// Seconds text such as "12.5": whole seconds + millis, lossless. Printing
// the digits happens outside the core; a round-trip test guards it.
pub fn to_secs(ms: u64) -> (r: (u64, u64))
    ensures r.0 * 1000 + r.1 == ms, r.1 < 1000,
{
    (ms / 1000, ms % 1000)
}
```

**Commit:** `core: to_secs lossless whole/millis split`

### Task 3.8: kept intervals for graders

**Objective:** an executable "which source instants survive", proved equal to `survives` point-wise, for
the release-benchmark cross-check (Task 3.19e).

```rust
// Kept source intervals [a, b) of a single-input edit, as data a grader can
// compare against: at most two pieces, proved equal to `survives` point-wise.
pub open spec fn in_piece(p: Option<(u64, u64)>, t: int) -> bool {
    match p {
        Some((a, b)) => a <= t && t < b,
        None => false,
    }
}

pub open spec fn piece_ok(p: Option<(u64, u64)>, c: Clip) -> bool {
    match p {
        Some((a, b)) => a < b && b <= c.duration_ms,
        None => true,
    }
}

pub fn kept_intervals(e: Edit, c: Clip) -> (r: Option<(Option<(u64, u64)>, Option<(u64, u64)>)>)
    ensures
        match r {
            Some(p) => accepted(e, c) && piece_ok(p.0, c) && piece_ok(p.1, c)
                && forall|t: int| 0 <= t < c.duration_ms ==>
                    (#[trigger] survives(e, t) <==> (in_piece(p.0, t) || in_piece(p.1, t))),
            None => !accepted(e, c),
        },
{
    if !validate(e, c) {
        return None;
    }
    let p = match e {
        Edit::Trim { start_ms, end_ms } => (Some((start_ms, end_ms)), None),
        Edit::CutOut { start_ms, end_ms } => {
            let a = if start_ms > 0 { Some((0u64, start_ms)) } else { None };
            let b = if end_ms < c.duration_ms { Some((end_ms, c.duration_ms)) } else { None };
            (a, b)
        }
        _ => if c.duration_ms > 0 { (Some((0u64, c.duration_ms)), None) } else { (None, None) },
    };
    Some(p)
}
```

**Pitfall:** returning a `Vec` and stating membership with `exists` failed, because the solver needs
witnesses. The fixed two-slot result states the same thing without `exists`. → 0 errors.
**Commit:** `core: kept_intervals proved equal to survives`

### Task 3.8b: ambiguity, the other reading (added with P3-D11; landed)

`flip_spec` / `flip` (total) return the other keep/remove reading of the same span. The proof fn
`readings_disagree` shows `intent_survives(i, c, t) == !intent_survives(flip_spec(i), c, t)` for every t
and every clip. Two mutants (the other reading of "first N" taken as "last N"; flip returning the same
reading) are killed.

### Task 3.9: mutants and seeds

**Files:** `scripts/mutate_core.sh`. Change the stability loop to `for seed in 0 1 2 3 4 5 6 7 8 9`. Add these
mutants; each was killed in the spike:

| Mutant | Old → new |
|---|---|
| try_label skips the runtime validator | `if validate(e, c) { Some(e) } else { None }` → `Some(e)` |
| pct_of truncates instead of refusing | `if p % 100 == 0 {` → `if true {` |
| half size accepts a source whose half is odd | `w % 4 == 0 && h % 4 == 0 {` (in `scale_half`) → `w % 2 == 0 && h % 2 == 0 {` |
| quarter size divides by 2 | `(w / 4, h / 4)` → `(w / 2, h / 2)` |
| "25% faster" read as +25 per-mille | `Some(1000 + 10 * p)` → `Some(1000 + p)` |
| "at 80% speed" read as 80 per-mille | `Some(10 * p)` → `Some(p)` |
| seconds text drops the fraction | `(ms / 1000, ms % 1000)` → `(ms / 1000, 0)` |
| try_label forgets the KeepLast bound | `Intent::KeepLast { ms } => ms <= c.duration_ms,` (in try_label) → `=> true,` |
| kept_intervals drops cut_out's tail | `let b = if end_ms < …` → `let b: Option<(u64, u64)> = None;` |
| kept_intervals keeps part of the cut | `Some((0u64, start_ms))` → `Some((0u64, start_ms + 1))` |

Expected: `stable:` line, 22 `killed:`, exit 0.
**Commit:** `test: 10 data-generation mutants; stability over seeds 0-9`

### Task 3.10: stock-Rust smoke test

Append to the core's test module (passed in the spike on Rust 1.99):

```rust
    // inside `mod tests`
    #[test]
    fn data_gen_entry_points() {
        let c = Clip { duration_ms: 30000, width: Some(1280), height: Some(720), has_audio: true };
        assert_eq!(try_label(Intent::RemoveLast { ms: 5000 }, c), Some(Edit::CutOut { start_ms: 25000, end_ms: 30000 }));
        assert_eq!(try_label(Intent::KeepLast { ms: 40000 }, c), None);
        assert_eq!(try_label(Intent::KeepFirst { ms: 30000 }, c), None); // whole clip: no-op
        assert_eq!(pct_of(30000, 25), Some(7500));
        assert_eq!(pct_of(10010, 25), None);
        assert_eq!(scale_half(1280, 720), Some((640, 360)));
        assert_eq!(scale_quarter(1280, 720), Some((320, 180)));
        assert_eq!(scale_quarter(720, 1280), Some((180, 320)));
        assert_eq!(scale_half(1918, 1080), None);
        assert_eq!(speed_percent_faster(25), Some(1250));
        assert_eq!(speed_at_percent(80), Some(800));
        assert_eq!(to_secs(12500), (12, 500));
        assert_eq!(kept_intervals(Edit::CutOut { start_ms: 0, end_ms: 5000 }, c), Some((None, Some((5000, 30000)))));
    }
```

**Commit:** `test: data-generation entry points run on stock Rust`

---

## App side (Tasks 3.11–3.20, specified, not yet executed)

### Task 3.11: SplitMix64

**Files:** `crates/minicut/src/rng.rs`. A `SplitMix64 { state: u64 }` with `next_u64`, `below(n)`
(rejection sampling, no modulo bias), `pick(&[T])` and `shuffle`. Test: the first three outputs for seed 0
equal the reference values `0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F`.
**Commit:** `minicut: SplitMix64 rng (no new dependency)`

### Task 3.12: number rendering

**Files:** `crates/minicut/src/render.rs`.
- `secs_text(ms)` uses `to_secs`: `12000 → "12"`, `12500 → "12.5"`, `12050 → "12.05"`, never a trailing zero.
- `short_text(ms)` = `secs_text + "s"`.
- `clock_text(ms)` uses `to_clock`: `72000 → "1:12"`, `72500 → "1:12.5"`.
- `pct_text(p)`.

Test: **exhaustive** round-trip over every ms in 0..=3,600,000. Parse each text back by hand (split on
`:` and `.`) and require the original ms. This covers the unproved digit printing completely in the range
the sampler uses.
**Commit:** `minicut: lossless number rendering, exhaustive round-trip`

### Task 3.13: `default_output` port

**Files:** `crates/minicut/src/naming.rs`, `crates/minicut/tests/golden/default_output.json`,
`scripts/export_default_output.py`.
- The script calls `minicut.build.default_output` for every tool × the 30 `FILENAMES` from
  `generate_dataset.py` × the formats/containers, and writes the golden file.
- The Rust test must match every entry, including `aac → .m4a` and concat's `concat_output.mp4`.

**Commit:** `minicut: default_output port, golden-checked against Python`

### Task 3.14: system prompt parity

**Files:** `crates/minicut/src/prompts.rs` (`pub const SYSTEM_PROMPT`), a `minicut system-prompt`
subcommand, and `tests/test_rust_prompts.py`. The test asserts the binary's output equals
`minicut.llm.prompts.SYSTEM_PROMPT` byte for byte, and skips if `target/debug/minicut` is absent. Add it to
the CI rust job after `cargo build`.
**Commit:** `minicut: system prompt const, parity-tested with prompts.py`

### Task 3.15: template table

**Files:** `crates/minicut/src/templates.rs`, generated `docs/templates.md`, and a `minicut templates
--markdown` subcommand.

```rust
pub enum IntentKind { KeepRange, RemoveRange, KeepFirst, RemoveFirst, KeepLast, RemoveLast,
                      ScaleExact, ScaleHalf, ScaleQuarter, SpeedFactor, SpeedFaster, SpeedAtPercent,
                      Convert, Extract, Mute, Concat }
pub enum NumStyle { Secs, Short, Clock, Pct }
pub struct Template {
    pub id: &'static str,           // v1 train id with its exact wording, or "v3_..."
    pub kind: IntentKind,
    pub style: NumStyle,
    pub text: &'static str,         // placeholders: {file} {a} {b} {n} {w} {h} {pct} {codec} {container} {fmt}
    pub applies: fn(&Ctx) -> bool,  // e.g. "middle" needs 0 < a && b < duration
    pub note: &'static str,         // reviewer's reason the wording means `kind`
}
```

Seed rows from the v1 train wording: `trim_t1..t5`, `cut_t1..t5`, `train_trim_x1..x3`, `train_cut_x1..x4`,
plus the scale/speed/convert/extract/mute/concat train and extra pools. Each row records its kind:
- `trim_t4` / `train_trim_x2` ("cut X down to / cut X to a-b") → KeepRange, note "verb 'cut' + 'down to' = keep".
- `cut_t4` / `train_cut_x4` ("the middle a to b") → RemoveRange, applies only when 0 < a and b < duration.
- `trim_t5` with start = 0 → KeepFirst; `cut_t1` with start = 0 → RemoveFirst.
- **Ambiguous table** (P3-D11), a separate `AMBIGUOUS` list, never used for tool rows: "trim the
  {first,last} {pct}%", "trim the {first,last} half" and "cut {file} to the first {a}" (the verb flips the
  meaning). Each row names the intent kind of its first reading; the second is `flip(first)`. Rows render
  as clarify with `missing: []` and a two-suggestion question from the vocabulary table.
- **Vocabulary table** `VOCAB: [(IntentKind, &str)]`: one canonical phrasing per kind, e.g. KeepFirst →
  "keep only the first {a}", RemoveFirst → "cut the first {a} off", KeepRange → "keep only {a} to {b}",
  RemoveRange → "cut out {a} to {b}". Test: each VOCAB phrasing equals the text of some non-ambiguous
  template of the same kind.

Tests:
- `docs/templates.md` equals the `minicut templates --markdown` output;
- no id is in `eval/freeze.json` `heldout_template_ids`;
- every v1 id's text equals its v1 wording (golden from `eval/generate_eval.py` / `training/generate_dataset.py`,
  exported by script);
- every `IntentKind` has at least two templates.

**Commit:** `minicut: reviewed template table with intent kinds and applicability`

### Task 3.16: clarify rows

**Files:** `crates/minicut/src/clarify.rs`.
- `ClarifyIntent { kind: Kind, provided: u16 }`, with the input bit always set because the app binds files.
- `missing` comes from `minicut_core::missing` and is mapped to names in bit order
  (`["start","end"]`, `["width","height"]`, `["codec","container"]`, …).
- The request wordings come from `CLARIFY_AMBIG`. Each is assigned the *set* of candidate kinds it could
  mean: "trim clip.mp4" → {Trim}; "cut clip.mp4" → {Trim, CutOut}.
- Question text: one sentence naming the missing arguments, plus one VOCAB suggestion per candidate kind
  (P3-D11). A test asserts every clarify question contains at least one VOCAB phrasing.
- Reading-ambiguous rows (Task 3.15 `AMBIGUOUS`) are built here too: `missing: []`, with a question
  rendering `VOCAB[kind(i)]` and `VOCAB[kind(flip(i))]` with the user's numbers.

Test: for every kind and provided-subset, the names equal required − provided.
**Commit:** `minicut: clarify rows with computed missing lists`

### Task 3.17: sampler and emitter: `minicut label`

**Files:** `crates/minicut/src/label.rs`, and a `label` subcommand:
`minicut label --seed N --counts counts.json --out rows.jsonl --meta meta.jsonl`.
- Fixture facts are a table copied from `generate_dataset.py`: `TRAIN_FIXTURES` durations/dims; the reserved
  fixture is not present.
- Times are sampled on the 0.5 s grid.
- The intent → `try_label` path applies; a `None` is counted and resampled, never written.
- Filenames come from the `FILENAMES` pool.
- Assistant JSON is serialized field by field in `grammar::TOOLS` order, with `output` from Task 3.13.
- Meta per row: `{kind, tool, fixture, template_id, intent, seed, generator: "minicut-label/<version>"}`.

Tests:
- the same seed gives byte-identical output;
- every assistant line's key order equals the TOOLS order for its tool;
- a 1,000-row sample: every assistant text passes `wire::validate` against the fixture probe facts.

**Commit:** `minicut: label subcommand, intent-first rows`

### Task 3.18: freeze gate

**Files:** `training/build_verified.py`, `tests/test_build_verified.py`.
- Reads rows + meta.
- Drops (with counts by reason): held-out template IDs, the reserved fixture, (prompt, canon) and
  (template_id, canon) collisions, bare eval prompts, and duplicates.
- Writes `train_verified_v1.jsonl` + meta + `stats.json`.
- Asserts every kept row is byte-identical to its input line.

Tests: an injected collision is dropped; an injected held-out id is dropped; a kept row is unchanged.
**Commit:** `training: drop-only freeze gate for verified rows`

### Task 3.19: acceptance: `scripts/accept_dataset.py`

Five checks, each printing counts and failing on any miss:
- **(a) Pilot filters.** Refactor `training/build_pilot.py`'s inline `continue` branches into
  `pilot_drop_reason(prompt, call) -> str | None`, without changing behaviour. Regression: re-running
  `python -m training.build_pilot --out <tmp>` reproduces `train_pilot_v1.jsonl` with SHA-256
  `5f7d9b1d1325ebfa6f848c473dda1f3f2f6b9bf342409cdf2e22db648fecca02`. Then the verified set must give **0**
  drop reasons.
- **(b) Runtime validator.** `minicut.validate.validate` accepts every label against its fixture's probe.
- **(c) Real grammar parser.** Every assistant target passes `test-gbnf-validator` against
  `minicut grammar` output. This is local, since CI has no llama.cpp; give the binary path with
  `--gbnf-validator`.
- **(d) Tokenizer preflight.** `python -m training.train_lora --data … --preflight` passes (local, cached
  tokenizer).
- **(e) Train/eval meaning agreement.** For every trim/cut_out case in `eval/release/cases.jsonl`, the Rust
  `kept_intervals` (via `minicut meaning --call … --probe …`, Task 3.20) equals the source intervals implied by
  `bench.expected_timeline`.

**Commit:** `test: dataset acceptance (pilot filters, validator, grammar, preflight, release meaning)`

### Task 3.20: `minicut meaning`

**Files:** `crates/minicut/src/main.rs`. Prints `{"kept": [[a_ms, b_ms], ...]}` from `kept_intervals`, or
exit 1 if the call is not accepted.
**Commit:** `minicut: meaning subcommand for graders`

### Task 3.21: docs

Update `training/README.md` (a "verified dataset v1" section with the command, seed, counts, drop stats and
SHA-256), `PROGRESS.md`, and `docs/rust-port.md` (Phase 3 status).
**Commit:** `docs: verified dataset v1`

---

## Exit experiment (after merge; GPU; not part of this plan's tasks)

Train the bounded pilot on `train_verified_v1.jsonl` with the same recipe, seed, step cap and factory as
`pilot-20261121-191106` (`python -m training.factory pilot …`). Board it against stock and pilot v1 on the
200 development cases with the existing gate:
- positive net task gain;
- better trim/cut_out;
- no new keep/remove inversion.

The difference between pilot v1 and pilot-verified isolates label quality, because size and mix are matched
(open question 2). Release scoring stays gated as before.
