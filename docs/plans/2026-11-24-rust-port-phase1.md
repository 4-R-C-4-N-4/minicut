# Rust port, Phase 0–1: verified core + contract surfaces

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** A cargo workspace with a Verus-verified `minicut-core` (validate, meaning, intent labels, clock,
atempo, clarify) and a `minicut` binary that emits the grammar and validates calls, at verdict parity with
`minicut/validate.py`.

**Architecture:** see [../rust-port.md](../rust-port.md). `minicut-core` is entirely inside `verus!`, with no I/O,
strings or floats. `minicut` owns serde, the float→integer boundary, and the CLI. Python stays the shipped
tool until Phase 2's parity gate.

**Tech stack:** Rust stable 1.99 (build), Verus `0.2026.11.22.1ce6a59` + rustup `1.100.0` (verification only),
`vstd = "=0.0.0-2026-11-22-0344"`, serde/serde_json.

**Provenance:** every code block below was run on 2026-11-24 in a prototype workspace with this layout.
Results: Verus `16 verified, 0 errors`; `scripts/mutate_core.sh` 8/8 killed; `cargo test` (stock 1.99) passes;
grammar golden test passes; parity 20,000/0 mismatches. A planted 10 ms drift was caught by parity (16
mismatches) and *not* by Verus, as intended. The CI YAML is the only part that has not run.

## Patterns to copy

- Tool table and field order: `minicut/grammar/generate_grammar.py` at commit `b460cfd` (closed PR #3, branch `dafny-grammar`); `grammar.rs` mirrors it.
- Validation semantics: `minicut/validate.py` (no-op rules at lines with `NOOP_TRIM_TOL_S`, `f == 1`, same-size scale).
- Probe facts: `minicut/probe.py` `Probe` (`duration_s`, `width`, `height`, `has_audio`).
- Verifier-friendly style: one `spec fn` per rule, exec fn with `ensures result == spec`, lemmas with empty bodies where Z3 can manage.

## Done when

- `scripts/verus.sh` → `16 verified, 0 errors` (or more, never fewer)
- `scripts/mutate_core.sh` → all `killed`, exit 0
- `cargo fmt --all --check && cargo test --workspace` → pass on stable
- `python scripts/parity_validate.py --bin target/debug/minicut --n 20000` → `0 mismatches`
- `python -m pytest tests -q` still passes (Python untouched)

---

## Phase 0: groundwork

### Task 0.1: Ignore build and Verus artifacts

**Objective:** Keep cargo output and the downloaded verifier out of git.

**Files:**
- Modify: `.gitignore`

**Step 1:** Append:

```
target/
.verus/
```

**Step 2:** `git status --short` shows only `.gitignore` changed.

**Step 3:** Commit: `git commit -am "chore: ignore cargo target and .verus"`

### Task 0.2: Pinned Verus setup + verify script

**Objective:** One command downloads the pinned Verus release, installs its toolchain, and verifies the core.

**Files:**
- Create: `scripts/verus.sh` (chmod +x)

```bash
#!/usr/bin/env bash
# Fetch the pinned Verus release into .verus/ (once) and verify the core.
#   scripts/verus.sh            verify crates/minicut-core
#   scripts/verus.sh --setup    only download + install the toolchain
set -euo pipefail
VERUS_RELEASE="0.2026.11.22.1ce6a59"
VERUS_TOOLCHAIN="1.100.0-x86_64-unknown-linux-gnu"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/.verus/$VERUS_RELEASE"

if [ ! -x "$DEST/verus-x86-linux/verus" ]; then
  mkdir -p "$DEST"
  url="https://github.com/verus-lang/verus/releases/download/release%2F$VERUS_RELEASE/verus-$VERUS_RELEASE-x86-linux.zip"
  curl -fsSL "$url" -o "$DEST/verus.zip"
  unzip -q -o "$DEST/verus.zip" -d "$DEST"
  rm "$DEST/verus.zip"
fi
rustup toolchain list | grep -q "^$VERUS_TOOLCHAIN" || rustup toolchain install "$VERUS_TOOLCHAIN" --profile minimal
[ "${1:-}" = "--setup" ] && exit 0

export PATH="$DEST/verus-x86-linux:$PATH"
cd "$ROOT/crates/minicut-core"
cargo verus verify
```

**Verify:** `scripts/verus.sh --setup` exits 0 and `.verus/0.2026.11.22.1ce6a59/verus-x86-linux/verus` exists. (Full verification runs once the core crate exists, Task 1.1.)

Note: cargo caches verification. An unchanged crate prints only `Finished`, and any content change re-verifies. A broken proof exits 101 with `verification results:: N verified, M errors`.

**Commit:** `git add scripts/verus.sh && git commit -m "build: pinned Verus setup script"`

### Task 0.3: Golden grammar file

**Objective:** Freeze the llama.cpp-validated strict grammar from PR #3 as the reference the Rust emitter must match.

**Files:**
- Create: `crates/minicut/tests/golden/minicut.gbnf`

```bash
mkdir -p crates/minicut/tests/golden
git fetch origin dafny-grammar
git show origin/dafny-grammar:minicut/grammar/minicut.gbnf > crates/minicut/tests/golden/minicut.gbnf
sha256sum crates/minicut/tests/golden/minicut.gbnf   # record in the commit message
```

This file was checked with `~/programs/llama.cpp/build/bin/test-gbnf-validator`: 1262/1262 unique training and eval targets accepted, 6/6 known-bad calls rejected (see the PR #3 review). PR #3 is closed. Replace the file's header comment, which names files that do not exist on main, with a provenance comment. The test ignores comment lines.

**Commit:** `git commit -m "test: golden strict grammar from PR #3 (sha256 ...)"`

### Task 0.4: CI workflow

**Objective:** Run Python tests, Rust tests + parity, and Verus + mutation checks on every PR.

**Files:**
- Create: `.github/workflows/ci.yml`

```yaml
name: ci

on:
  push:
    branches: [main]
  pull_request:

jobs:
  python:
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: sudo apt-get update && sudo apt-get install -y ffmpeg
      - run: pip install -e ".[dev]"
      - run: python -m factory.generate --out factory/fixtures --all
      - run: python -m pytest tests -q
      - run: python -m eval.release controls

  rust:
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -e ".[dev]"
      - run: cargo fmt --all --check
      - run: cargo test --workspace
      - run: cargo build
      - run: python scripts/parity_validate.py --bin target/debug/minicut --n 20000

  verus:
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: actions/cache@v4
        with:
          path: |
            .verus
            ~/.rustup/toolchains/1.100.0-x86_64-unknown-linux-gnu
          key: verus-0.2026.11.22.1ce6a59
      - run: scripts/verus.sh
      - run: scripts/mutate_core.sh
```

Not run on GitHub yet. The rust and verus jobs fail until Phase 1 lands, so land this commit at the end of Phase 1, or land only the `python` job now. That job alone would have caught PR #2's broken import.

**Commit:** `git commit -m "ci: python, rust and verus jobs"`

---

## Phase 1: verified core

### Task 1.1: Workspace + empty verified crate

**Objective:** A cargo workspace whose core crate is verified by Verus.

**Files:**
- Create: `Cargo.toml`
- Create: `crates/minicut-core/Cargo.toml`
- Create: `crates/minicut-core/src/lib.rs`

**Step 1:** Root `Cargo.toml`:

```toml
[workspace]
resolver = "2"
members = ["crates/minicut-core", "crates/minicut"]

[workspace.dependencies]
vstd = "=0.0.0-2026-11-22-0344"
```

**Step 2:** `crates/minicut-core/Cargo.toml`:

```toml
[package]
name = "minicut-core"
version = "0.1.0"
edition = "2021"

[dependencies]
vstd = { workspace = true }

[package.metadata.verus]
verify = true
```

**Step 3:** `crates/minicut-core/src/lib.rs` starts as the header plus an empty macro body:

```rust
// minicut-core: the verified heart of MiniCut. No I/O, no floats, no strings.
// Times are integer milliseconds; speed is an integer per-mille factor.
use vstd::prelude::*;

verus! {
} // verus!
```

Keep `pub const NOOP_TRIM_TOL_MS` with the types in Task 1.2. Each later task appends one section above the closing `} // verus!`.

**Step 4:** Run: `scripts/verus.sh`
Expected: `verification results:: 0 verified, 0 errors` (or a count of type-only items).

**Step 5:** `cargo build -p minicut-core` with stock stable succeeds. Verification is erased.

**Commit:** `git commit -m "core: verified crate skeleton"`

### Task 1.2: Core types

**Objective:** Integer-unit types for edits and probed clips; no strings, no floats.

**Files:**
- Modify: `crates/minicut-core/src/lib.rs`

Add (allowlists are enums, so an unlisted codec cannot be built):

```rust
// ---------------------------------------------------------------- types ----
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Codec { Libx264, Libx265, Vp9 }

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Container { Mp4, Mkv, Webm }

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AudioFormat { Mp3, Wav, Aac }

/// One edit, already bound to files by the application. File names are
/// handled outside the verified core; the core sees the input count.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Edit {
    Trim { start_ms: u64, end_ms: u64 },
    CutOut { start_ms: u64, end_ms: u64 },
    Concat { n_inputs: u64 },
    Scale { width: u32, height: u32 },
    Speed { factor_pm: u64 },
    Convert { codec: Codec, container: Container, crf: u8 },
    ExtractAudio { format: AudioFormat },
    Mute,
}

/// What ffprobe told us about the (first) input. `None` dims = no video.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Clip { pub duration_ms: u64, pub width: Option<u32>, pub height: Option<u32>, pub has_audio: bool }

pub const NOOP_TRIM_TOL_MS: u64 = 50;
```

Run: `scripts/verus.sh`
Expected: 0 errors.

**Commit:** `git commit -m "core: Edit, Clip, allowlist enums"`

### Task 1.3: Validator proved equal to its spec

**Objective:** `validate(e, c) == accepted(e, c)` for every edit and clip, with the no-op rule as its own predicate.

**Files:**
- Modify: `crates/minicut-core/src/lib.rs`

**Step 1 (spec first):** add the two predicates and `accepted`:

```rust
// ------------------------------------------- spec: well-formed (validate.py) --
pub open spec fn well_formed(e: Edit, c: Clip) -> bool {
    match e {
        Edit::Trim { start_ms, end_ms } => start_ms < end_ms && end_ms <= c.duration_ms,
        Edit::CutOut { start_ms, end_ms } => start_ms < end_ms && end_ms <= c.duration_ms,
        Edit::Concat { n_inputs } => n_inputs >= 2,
        Edit::Scale { width, height } => width > 0 && height > 0 && width % 2 == 0 && height % 2 == 0,
        Edit::Speed { factor_pm } => 250 <= factor_pm <= 4000,
        Edit::Convert { crf, .. } => crf <= 51,
        Edit::ExtractAudio { .. } => true,   // parity with validate.py (see contract doc)
        Edit::Mute => true,
    }
}

// ------------------------------------- spec: changes something (no-op rule) --
pub open spec fn changes_something(e: Edit, c: Clip) -> bool {
    match e {
        Edit::Trim { start_ms, end_ms } =>
            !(start_ms <= NOOP_TRIM_TOL_MS && end_ms + NOOP_TRIM_TOL_MS >= c.duration_ms),
        Edit::Scale { width, height } => !(c.width == Some(width) && c.height == Some(height)),
        Edit::Speed { factor_pm } => factor_pm != 1000,
        _ => true,
    }
}

pub open spec fn accepted(e: Edit, c: Clip) -> bool {
    well_formed(e, c) && changes_something(e, c)
}
```

**Step 2 (failing):** add `validate` with the `ensures` but body `{ true }`. Run: `scripts/verus.sh`
Expected: `postcondition not satisfied`.

**Step 3:** replace the body:

```rust
// ------------------------------------------------------ exec: the validator --
pub fn validate(e: Edit, c: Clip) -> (ok: bool)
    ensures ok == accepted(e, c)
{
    let wf = match e {
        Edit::Trim { start_ms, end_ms } => start_ms < end_ms && end_ms <= c.duration_ms,
        Edit::CutOut { start_ms, end_ms } => start_ms < end_ms && end_ms <= c.duration_ms,
        Edit::Concat { n_inputs } => n_inputs >= 2,
        Edit::Scale { width, height } => width > 0 && height > 0 && width % 2 == 0 && height % 2 == 0,
        Edit::Speed { factor_pm } => 250 <= factor_pm && factor_pm <= 4000,
        Edit::Convert { crf, .. } => crf <= 51,
        Edit::ExtractAudio { .. } => true,
        Edit::Mute => true,
    };
    if !wf { return false; }
    match e {
        Edit::Trim { start_ms, end_ms } =>
            !(start_ms <= NOOP_TRIM_TOL_MS && end_ms as u128 + NOOP_TRIM_TOL_MS as u128 >= c.duration_ms as u128),
        Edit::Scale { width, height } => !(c.width == Some(width) && c.height == Some(height)),
        Edit::Speed { factor_pm } => factor_pm != 1000,
        _ => true,
    }
}
```

The `u128` cast avoids overflow in `end_ms + 50`. Without it Verus reports a possible arithmetic overflow, which is correct.

**Step 4:** Run: `scripts/verus.sh`
Expected: 0 errors.

**Commit:** `git commit -m "core: validate proved equal to accepted()"`

### Task 1.4: Meaning: what survives

**Objective:** State each edit's surviving duration and prove trim/cut_out partition the clip.

**Files:**
- Modify: `crates/minicut-core/src/lib.rs`

```rust
// ------------------------------------------- spec: meaning (what survives) --
pub open spec fn kept_ms(e: Edit, c: Clip) -> int {
    match e {
        Edit::Trim { start_ms, end_ms } => end_ms - start_ms,
        Edit::CutOut { start_ms, end_ms } => c.duration_ms - (end_ms - start_ms),
        Edit::Speed { factor_pm } => c.duration_ms * 1000 / factor_pm as int,
        _ => c.duration_ms as int,
    }
}

pub proof fn trim_cut_partition(s: u64, e: u64, c: Clip)
    requires well_formed(Edit::Trim { start_ms: s, end_ms: e }, c)
    ensures
        kept_ms(Edit::Trim { start_ms: s, end_ms: e }, c)
            + kept_ms(Edit::CutOut { start_ms: s, end_ms: e }, c) == c.duration_ms,
        kept_ms(Edit::Trim { start_ms: s, end_ms: e }, c) > 0,
        kept_ms(Edit::CutOut { start_ms: s, end_ms: e }, c) >= 0,
{
}
```

Run: `scripts/verus.sh`
Expected: 0 errors. The lemma needs no proof body; if Verus needs hints here, the spec is wrong.

**Commit:** `git commit -m "core: kept_ms meaning, partition lemma"`

### Task 1.5: Intent → label, proved

**Objective:** Labels for training data come from a function proved to preserve meaning and the keep/remove contract.

**Files:**
- Modify: `crates/minicut-core/src/lib.rs`

```rust
// ------------------------------------------------ intents -> labels (data) --
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Intent {
    KeepRange { start_ms: u64, end_ms: u64 },
    RemoveRange { start_ms: u64, end_ms: u64 },
    KeepFirst { ms: u64 },
    RemoveFirst { ms: u64 },
    KeepLast { ms: u64 },
    RemoveLast { ms: u64 },
}

/// The intended surviving duration, stated without mentioning any tool.
pub open spec fn intent_kept(i: Intent, c: Clip) -> int {
    match i {
        Intent::KeepRange { start_ms, end_ms } => end_ms - start_ms,
        Intent::RemoveRange { start_ms, end_ms } => c.duration_ms - (end_ms - start_ms),
        Intent::KeepFirst { ms } => ms as int,
        Intent::RemoveFirst { ms } => c.duration_ms - ms,
        Intent::KeepLast { ms } => ms as int,
        Intent::RemoveLast { ms } => c.duration_ms - ms,
    }
}

/// Which tool family the request means: keep-intents are trims,
/// remove-intents are cut_outs. This is the keep/remove contract.
pub open spec fn is_keep(i: Intent) -> bool {
    match i {
        Intent::KeepRange { .. } | Intent::KeepFirst { .. } | Intent::KeepLast { .. } => true,
        _ => false,
    }
}

pub fn label(i: Intent, c: Clip) -> (e: Edit)
    requires
        match i {
            Intent::KeepLast { ms } => ms <= c.duration_ms,
            Intent::RemoveLast { ms } => ms <= c.duration_ms,
            _ => true,
        },
    ensures
        kept_ms(e, c) == intent_kept(i, c),
        is_keep(i) <==> (e is Trim),
        !is_keep(i) <==> (e is CutOut),
{
    match i {
        Intent::KeepRange { start_ms, end_ms } => Edit::Trim { start_ms, end_ms },
        Intent::RemoveRange { start_ms, end_ms } => Edit::CutOut { start_ms, end_ms },
        Intent::KeepFirst { ms } => Edit::Trim { start_ms: 0, end_ms: ms },
        Intent::RemoveFirst { ms } => Edit::CutOut { start_ms: 0, end_ms: ms },
        Intent::KeepLast { ms } => Edit::Trim { start_ms: c.duration_ms - ms, end_ms: c.duration_ms },
        Intent::RemoveLast { ms } => Edit::CutOut { start_ms: c.duration_ms - ms, end_ms: c.duration_ms },
    }
}
```

Run: `scripts/verus.sh`
Expected: 0 errors. This is the function Phase 3's data generator calls, and the reason the v1 keep/remove inversions cannot appear in generated labels.

**Commit:** `git commit -m "core: intent labels with keep/remove contract"`

### Task 1.6: Lossless mm:ss clock

**Objective:** Rendering a time as minutes/seconds/millis loses nothing (the v1 mm:ss bug).

**Files:**
- Modify: `crates/minicut-core/src/lib.rs`

```rust
// --------------------------------------------- clock rendering (mm:ss.fff) --
pub fn to_clock(ms: u64) -> (r: (u64, u64, u64))
    ensures r.0 * 60000 + r.1 * 1000 + r.2 == ms, r.1 < 60, r.2 < 1000,
{
    let rem = ms % 60000;
    (ms / 60000, rem / 1000, rem % 1000)
}
```

Run: `scripts/verus.sh`
Expected: 0 errors.

**Commit:** `git commit -m "core: lossless clock"`

### Task 1.7: atempo stages

**Objective:** Speed factors decompose into ffmpeg atempo stages within 0.5–2.0× whose product is the factor to 0.001×.

**Files:**
- Modify: `crates/minicut-core/src/lib.rs`

```rust
// ------------------------------------------------------- atempo chain ------
// ffmpeg atempo takes 0.5..2.0 per stage here; 0.25..4.0 needs <= 2 stages.
pub open spec fn stage_ok(s: u64) -> bool { 500 <= s <= 2000 }

pub fn atempo_stages(f: u64) -> (r: (u64, Option<u64>))
    requires 250 <= f <= 4000
    ensures
        stage_ok(r.0),
        match r.1 {
            None => r.0 == f,
            Some(s2) => stage_ok(s2) && (r.0 * s2) / 1000 <= f && f <= (r.0 * s2) / 1000 + 1,
        },
{
    if f > 2000 {
        (2000, Some(f / 2))
    } else if f < 500 {
        (500, Some(f * 2))
    } else {
        (f, None)
    }
}
```

This replaces `minicut/build.py:_atempo_chain`. That loop handles at most one extra stage in the 0.25..4.0 range; here it is explicit.

Run: `scripts/verus.sh`
Expected: 0 errors.

**Commit:** `git commit -m "core: atempo stages proved in range"`

### Task 1.8: Clarify missing, computed

**Objective:** `missing` is exactly required − provided, as a bitmask proved bit by bit.

**Files:**
- Modify: `crates/minicut-core/src/lib.rs`

```rust
// --------------------------------------------- clarify: missing is computed --
// Arg bits: 0 input, 1 start, 2 end, 3 inputs, 4 width, 5 height, 6 factor,
// 7 codec, 8 container, 9 format.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind { Trim, CutOut, Concat, Scale, Speed, Convert, ExtractAudio, Mute }

pub open spec fn required_spec(k: Kind) -> u16 {
    match k {
        Kind::Trim => 0b0000000111,
        Kind::CutOut => 0b0000000111,
        Kind::Concat => 0b0000001000,
        Kind::Scale => 0b0000110001,
        Kind::Speed => 0b0001000001,
        Kind::Convert => 0b0110000001,
        Kind::ExtractAudio => 0b1000000001,
        Kind::Mute => 0b0000000001,
    }
}

pub fn required(k: Kind) -> (r: u16)
    ensures r == required_spec(k)
{
    match k {
        Kind::Trim => 0b0000000111,
        Kind::CutOut => 0b0000000111,
        Kind::Concat => 0b0000001000,
        Kind::Scale => 0b0000110001,
        Kind::Speed => 0b0001000001,
        Kind::Convert => 0b0110000001,
        Kind::ExtractAudio => 0b1000000001,
        Kind::Mute => 0b0000000001,
    }
}

pub fn missing(k: Kind, provided: u16) -> (m: u16)
    ensures
        forall|b: u16| #![auto] b < 16 ==>
            ((m >> b) & 1 == 1 <==> ((required_spec(k) >> b) & 1 == 1 && (provided >> b) & 1 == 0)),
{
    let r = required(k);
    let m = r & !provided;
    assert(forall|b: u16| #![auto] b < 16 ==>
        (((r & !provided) >> b) & 1 == 1 <==> ((r >> b) & 1 == 1 && (provided >> b) & 1 == 0)))
        by (bit_vector);
    m
}
```

The wire layer maps bit names to strings (`input`, `start`, ...). In the CLI the app binds files, so pass `provided` with bit 0 (`input`) or bit 3 (`inputs`) set. That reproduces frozen labels such as "convert it" → `["codec", "container"]`.

Run: `scripts/verus.sh`
Expected: `verification results:: 16 verified, 0 errors` in total.

**Commit:** `git commit -m "core: clarify missing = required - provided"`

### Task 1.9: Mutation check

**Objective:** Prove the proofs have teeth: eight planted bugs must each fail verification.

**Files:**
- Create: `scripts/mutate_core.sh` (chmod +x)

```bash
#!/usr/bin/env bash
# Mutation check for the verified core: every planted bug must make Verus
# fail. A surviving mutant means a proof is vacuous or a spec is too weak.
#   scripts/mutate_core.sh       (run scripts/verus.sh --setup first)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VERUS="$(ls -d "$ROOT"/.verus/*/verus-x86-linux | tail -1)/verus"
SRC="$ROOT/crates/minicut-core/src/lib.rs"
WORK="$(mktemp -d -p "${TMPDIR:-$ROOT/target}")"
trap 'rm -rf "$WORK"' EXIT
survived=0

mutant() {  # mutant "<description>" "<exact old text>" "<new text>"
  python3 - "$SRC" "$WORK/m.rs" "$2" "$3" <<'EOF' || { echo "BAD PATTERN: $1"; survived=1; return; }
import sys
src, dst, old, new = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
text = open(src).read()
if old not in text:
    sys.exit(1)
open(dst, "w").write(text.replace(old, new, 1))
EOF
  if "$VERUS" --crate-type=lib "$WORK/m.rs" >/dev/null 2>&1; then
    echo "SURVIVED: $1"; survived=1
  else
    echo "killed:   $1"
  fi
}

mutant "validate forgets speed-1x no-op" \
  "Edit::Speed { factor_pm } => factor_pm != 1000,
        _ => true,
    }
}

// ----" \
  "Edit::Speed { factor_pm } => true,
        _ => true,
    }
}

// ----"
mutant "validate allows odd dims" \
  "Edit::Scale { width, height } => width > 0 && height > 0 && width % 2 == 0 && height % 2 == 0,
        Edit::Speed { factor_pm } => 250 <= factor_pm && factor_pm <= 4000," \
  "Edit::Scale { width, height } => width > 0 && height > 0,
        Edit::Speed { factor_pm } => 250 <= factor_pm && factor_pm <= 4000,"
mutant "atempo wrong second stage" "(2000, Some(f / 2))" "(2000, Some(f / 3))"
mutant "atempo stage out of range" "(500, Some(f * 2))" "(250, Some(f * 4))"
mutant "clarify missing lists provided args" "let m = r & !provided;" "let m = r & provided;"
mutant "keep/remove inversion" \
  "Intent::KeepLast { ms } => Edit::Trim { start_ms: c.duration_ms - ms, end_ms: c.duration_ms }," \
  "Intent::KeepLast { ms } => Edit::CutOut { start_ms: c.duration_ms - ms, end_ms: c.duration_ms },"
mutant "off-by-one label" \
  "Intent::KeepFirst { ms } => Edit::Trim { start_ms: 0, end_ms: ms }," \
  "Intent::KeepFirst { ms } => Edit::Trim { start_ms: 0, end_ms: ms + 1 },"
mutant "clock drops the fraction (v1 dataset bug)" \
  "(ms / 60000, rem / 1000, rem % 1000)" "(ms / 60000, rem / 1000, 0)"

exit $survived
```

Run: `scripts/mutate_core.sh`. Expected: eight `killed:` lines, exit 0. A `SURVIVED` line is a weak spec; fix the spec, never the mutant.

**Commit:** `git commit -m "test: mutation check for verified core"`

### Task 1.10: Core unit tests on stock Rust

**Objective:** The verified code runs as ordinary Rust and gives the expected values.

**Files:**
- Modify: `crates/minicut-core/src/lib.rs` (append outside `verus!`)

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_on_stock_rust() {
        let c = Clip { duration_ms: 30000, width: Some(1280), height: Some(720), has_audio: true };
        assert!(validate(label(Intent::RemoveFirst { ms: 10000 }, c), c));
        assert!(!validate(Edit::Speed { factor_pm: 1000 }, c));
        assert!(!validate(Edit::Trim { start_ms: 0, end_ms: 30000 }, c));
        assert_eq!(atempo_stages(4000), (2000, Some(2000)));
        assert_eq!(atempo_stages(250), (500, Some(500)));
        assert_eq!(to_clock(72500), (1, 12, 500));
        assert_eq!(missing(Kind::Trim, 0b1), 0b110);
    }
}
```

Run: `cargo test -p minicut-core`. Expected: `1 passed`.

**Commit:** `git commit -m "test: core runs on stock rust"`

### Task 1.11: `minicut grammar` from one table

**Objective:** The binary emits the grammar; its rules equal the golden file.

**Files:**
- Create: `crates/minicut/Cargo.toml`
- Create: `crates/minicut/src/grammar.rs`
- Create: `crates/minicut/src/main.rs`
- Create: `crates/minicut/tests/grammar.rs`

**Step 1 (failing test):** `crates/minicut/tests/grammar.rs`:

```rust
// The emitted grammar's rules must equal the llama.cpp-validated golden file.
use std::process::Command;

fn rules(s: &str) -> Vec<String> {
    s.lines()
        .filter(|l| !l.starts_with('#'))
        .map(String::from)
        .collect()
}

#[test]
fn grammar_rules_match_golden() {
    let out = Command::new(env!("CARGO_BIN_EXE_minicut"))
        .arg("grammar")
        .output()
        .unwrap();
    assert!(out.status.success());
    let got = String::from_utf8(out.stdout).unwrap();
    let golden = include_str!("golden/minicut.gbnf");
    assert_eq!(rules(&got), rules(golden));
}
```

Comment lines are excluded: the header names the Rust generator, and every rule line must match.

**Step 2:** `crates/minicut/Cargo.toml`:

```toml
[package]
name = "minicut"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "minicut"
path = "src/main.rs"

[dependencies]
minicut-core = { path = "../minicut-core" }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
```

**Step 3:** `crates/minicut/src/grammar.rs`:

```rust
//! GBNF grammar for llama.cpp, emitted from one table of tool shapes.
//! Rule lines must equal the PR #3 grammar (validated with llama.cpp's
//! test-gbnf-validator); the header comment is this crate's own.

/// JSON value kinds a field may take.
#[derive(Clone, Copy)]
pub enum Ty {
    Str,
    Number,
    Int,
    StrList,
    StrList2,
    Enum(&'static [&'static str]),
}

pub struct Field {
    pub name: &'static str,
    pub ty: Ty,
    pub required: bool,
}

const fn req(name: &'static str, ty: Ty) -> Field {
    Field {
        name,
        ty,
        required: true,
    }
}
const fn opt(name: &'static str, ty: Ty) -> Field {
    Field {
        name,
        ty,
        required: false,
    }
}

pub const CODECS: &[&str] = &["libx264", "libx265", "vp9"];
pub const CONTAINERS: &[&str] = &["mp4", "mkv", "webm"];
pub const AUDIO_FORMATS: &[&str] = &["mp3", "wav", "aac"];

const OUT: Field = opt("output", Ty::Str);

/// Every tool the model may call, fields in canonical order.
pub const TOOLS: &[(&str, &[Field])] = &[
    (
        "trim",
        &[
            req("input", Ty::Str),
            req("start", Ty::Number),
            req("end", Ty::Number),
            OUT,
        ],
    ),
    (
        "cut_out",
        &[
            req("input", Ty::Str),
            req("start", Ty::Number),
            req("end", Ty::Number),
            OUT,
        ],
    ),
    ("concat", &[req("inputs", Ty::StrList2), OUT]),
    (
        "scale",
        &[
            req("input", Ty::Str),
            req("width", Ty::Int),
            req("height", Ty::Int),
            OUT,
        ],
    ),
    (
        "speed",
        &[req("input", Ty::Str), req("factor", Ty::Number), OUT],
    ),
    (
        "convert",
        &[
            req("input", Ty::Str),
            req("codec", Ty::Enum(CODECS)),
            req("container", Ty::Enum(CONTAINERS)),
            opt("crf", Ty::Int),
            OUT,
        ],
    ),
    (
        "extract_audio",
        &[
            req("input", Ty::Str),
            req("format", Ty::Enum(AUDIO_FORMATS)),
            OUT,
        ],
    ),
    ("mute", &[req("input", Ty::Str), OUT]),
    (
        "clarify",
        &[req("question", Ty::Str), opt("missing", Ty::StrList)],
    ),
];

fn lit(s: &str) -> String {
    format!("\"\\\"{s}\\\"\"")
}

fn rule_name(tool: &str) -> String {
    // llama.cpp rule names allow [a-zA-Z0-9-] only.
    format!("{}-call", tool.replace('_', "-"))
}

fn value(ty: Ty) -> String {
    match ty {
        Ty::Str => "string".into(),
        Ty::Number => "number".into(),
        Ty::Int => "int".into(),
        Ty::StrList => "str-list".into(),
        Ty::StrList2 => "str-list-2".into(),
        Ty::Enum(vals) => format!(
            "({})",
            vals.iter().map(|v| lit(v)).collect::<Vec<_>>().join(" | ")
        ),
    }
}

pub fn gbnf() -> String {
    let mut out = String::from(
        "# MiniCut GBNF grammar for llama.cpp.\n\
         # GENERATED by `minicut grammar` from crates/minicut/src/grammar.rs. Do not edit.\n",
    );
    let names: Vec<String> = TOOLS.iter().map(|(t, _)| rule_name(t)).collect();
    out += &format!("root ::= {}\n", names.join(" | "));
    for (tool, fields) in TOOLS {
        let mut rule = format!(
            "{} ::= \"{{\" ws {} ws \":\" ws {}",
            rule_name(tool),
            lit("tool"),
            lit(tool)
        );
        for f in *fields {
            let frag = format!("ws \",\" ws {} ws \":\" ws {}", lit(f.name), value(f.ty));
            if f.required {
                rule += &format!(" {frag}");
            } else {
                rule += &format!(" ({frag})?");
            }
        }
        rule += " ws \"}\"\n";
        out += &rule;
    }
    out += "string ::= \"\\\"\" ([^\"\\\\] | \"\\\\\" .)* \"\\\"\"\n";
    out += "number ::= \"-\"? [0-9]+ (\".\" [0-9]+)?\n";
    out += "int ::= [0-9]+\n";
    out += "str-list ::= \"[\" ws (string (ws \",\" ws string)*)? ws \"]\"\n";
    out += "str-list-2 ::= \"[\" ws string ws \",\" ws string (ws \",\" ws string)* ws \"]\"\n";
    out += "ws ::= [ \\t\\n]*\n";
    out
}
```

**Step 4:** `crates/minicut/src/main.rs` (the full file after Task 1.12 is below; for this task, keep only the `grammar` arm and the fallback):

```rust
mod grammar;
mod wire;

use std::process::ExitCode;

const USAGE: &str = "usage: minicut grammar | minicut validate --call JSON --probe JSON";

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(|s| s.as_str()) {
        Some("grammar") => {
            print!("{}", grammar::gbnf());
            ExitCode::SUCCESS
        }
        Some("validate") if args.iter().any(|a| a == "--batch") => {
            // JSONL on stdin: {"call": <raw call string>, "probe": {...}} -> {"ok": bool}
            use std::io::BufRead;
            for line in std::io::stdin().lock().lines() {
                let line = line.expect("stdin");
                let row: serde_json::Value = serde_json::from_str(&line).expect("batch row");
                let probe: wire::WireProbe =
                    serde_json::from_value(row["probe"].clone()).expect("probe");
                let ok = wire::validate(row["call"].as_str().expect("call string"), &probe).is_ok();
                println!("{}", serde_json::json!({"ok": ok}));
            }
            ExitCode::SUCCESS
        }
        Some("validate") => {
            let (Some(call), Some(probe)) = (flag(&args, "--call"), flag(&args, "--probe")) else {
                eprintln!("{USAGE}");
                return ExitCode::from(2);
            };
            let probe: wire::WireProbe = match serde_json::from_str(probe) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("bad --probe: {e}");
                    return ExitCode::from(2);
                }
            };
            let result = wire::validate(call, &probe);
            let out = match &result {
                Ok(()) => serde_json::json!({"ok": true}),
                Err(e) => serde_json::json!({"ok": false, "error": e}),
            };
            println!("{out}");
            if result.is_ok() {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(1)
            }
        }
        _ => {
            eprintln!("{USAGE}");
            ExitCode::from(2)
        }
    }
}
```

**Step 5:** `cargo test -p minicut --test grammar`. Expected: `1 passed`.

**Step 6 (real parser):** `target/debug/minicut grammar > "$TMPDIR/g.gbnf"` and run `~/programs/llama.cpp/build/bin/test-gbnf-validator "$TMPDIR/g.gbnf" <file>` on one valid trim call (expect `Input string is valid`) and on `{"tool": "speed", "input": "a.mp4", "speed": 2.0}` (expect `invalid`).

**Commit:** `git commit -m "minicut: grammar subcommand matches golden"`

### Task 1.12: Wire layer + `minicut validate`

**Objective:** Strict JSON parsing, the single float→integer boundary, and the validate subcommand.

**Files:**
- Create: `crates/minicut/src/wire.rs`
- Modify: `crates/minicut/src/main.rs` (full file shown in Task 1.11)

```rust
//! The model's JSON, parsed strictly, then converted to core types.
//! This is the only place floats and strings meet the verified core.
use minicut_core::{AudioFormat, Clip, Codec, Container, Edit};
use serde::Deserialize;
use std::path::{Component, Path};

#[derive(Deserialize, Debug, Clone)]
#[serde(tag = "tool", rename_all = "snake_case", deny_unknown_fields)]
pub enum WireCall {
    Trim {
        input: String,
        start: f64,
        end: f64,
        output: Option<String>,
    },
    CutOut {
        input: String,
        start: f64,
        end: f64,
        output: Option<String>,
    },
    Concat {
        inputs: Vec<String>,
        output: Option<String>,
    },
    Scale {
        input: String,
        width: u32,
        height: u32,
        output: Option<String>,
    },
    Speed {
        input: String,
        factor: f64,
        output: Option<String>,
    },
    Convert {
        input: String,
        codec: WireCodec,
        container: WireContainer,
        crf: Option<u8>,
        output: Option<String>,
    },
    ExtractAudio {
        input: String,
        format: WireAudio,
        output: Option<String>,
    },
    Mute {
        input: String,
        output: Option<String>,
    },
    Clarify {
        question: String,
        missing: Option<Vec<String>>,
    },
}

#[derive(Deserialize, Debug, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum WireCodec {
    Libx264,
    Libx265,
    Vp9,
}

#[derive(Deserialize, Debug, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum WireContainer {
    Mp4,
    Mkv,
    Webm,
}

#[derive(Deserialize, Debug, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum WireAudio {
    Mp3,
    Wav,
    Aac,
}

/// ffprobe facts the validator needs, as `minicut validate --probe` takes them.
#[derive(Deserialize, Debug, Clone, Copy)]
pub struct WireProbe {
    pub duration_s: f64,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub has_audio: bool,
}

pub enum Parsed {
    Edit(Edit),
    Clarify {
        question: String,
        missing: Vec<String>,
    },
}

/// Seconds -> nearest ms. None for negative, NaN, infinite or absurd values.
pub fn secs_to_ms(x: f64) -> Option<u64> {
    if !x.is_finite() || x < 0.0 || x > 1.0e9 {
        return None;
    }
    Some((x * 1000.0).round() as u64)
}

/// Probe duration -> ms, rounded down so `end <= duration` never admits
/// an end past the real clip.
pub fn probe_ms(x: f64) -> u64 {
    if !x.is_finite() || x <= 0.0 {
        0
    } else {
        (x * 1000.0).floor() as u64
    }
}

pub fn clip(p: &WireProbe) -> Clip {
    Clip {
        duration_ms: probe_ms(p.duration_s),
        width: p.width,
        height: p.height,
        has_audio: p.has_audio,
    }
}

fn check_path(f: &str) -> Result<(), String> {
    if f.is_empty() || f.contains('\0') {
        return Err("paths must be non-empty strings".into());
    }
    let p = Path::new(f);
    if p.is_absolute() || p.components().any(|c| c == Component::ParentDir) {
        return Err(format!(
            "path must be relative and inside the workdir: {f:?}"
        ));
    }
    Ok(())
}

fn paths(call: &WireCall) -> Vec<&str> {
    use WireCall::*;
    let (files, out): (Vec<&str>, &Option<String>) = match call {
        Trim { input, output, .. }
        | CutOut { input, output, .. }
        | Scale { input, output, .. }
        | Speed { input, output, .. }
        | Convert { input, output, .. }
        | ExtractAudio { input, output, .. }
        | Mute { input, output } => (vec![input.as_str()], output),
        Concat { inputs, output } => (inputs.iter().map(|s| s.as_str()).collect(), output),
        Clarify { .. } => return vec![],
    };
    files.into_iter().chain(out.as_deref()).collect()
}

pub fn parse(raw: &str) -> Result<WireCall, String> {
    serde_json::from_str(raw).map_err(|e| format!("invalid tool call: {e}"))
}

/// Structural + path checks, then conversion to the core `Edit`.
pub fn to_parsed(call: &WireCall) -> Result<Parsed, String> {
    use WireCall::*;
    if let Clarify { question, missing } = call {
        if question.trim().is_empty() {
            return Err("clarify needs a non-empty question".into());
        }
        return Ok(Parsed::Clarify {
            question: question.clone(),
            missing: missing.clone().unwrap_or_default(),
        });
    }
    for f in paths(call) {
        check_path(f)?;
    }
    let span = |s: f64, e: f64| match (secs_to_ms(s), secs_to_ms(e)) {
        (Some(a), Some(b)) => Ok((a, b)),
        _ => Err("start/end must be non-negative numbers (seconds)".to_string()),
    };
    let edit = match *call {
        Trim { start, end, .. } => {
            let (s, e) = span(start, end)?;
            Edit::Trim {
                start_ms: s,
                end_ms: e,
            }
        }
        CutOut { start, end, .. } => {
            let (s, e) = span(start, end)?;
            Edit::CutOut {
                start_ms: s,
                end_ms: e,
            }
        }
        Concat { ref inputs, .. } => Edit::Concat {
            n_inputs: inputs.len() as u64,
        },
        Scale { width, height, .. } => Edit::Scale { width, height },
        Speed { factor, .. } => {
            let pm = secs_to_ms(factor)
                .ok_or_else(|| format!("speed factor must be 0.25..4.0, got {factor}"))?;
            Edit::Speed { factor_pm: pm }
        }
        Convert {
            codec,
            container,
            crf,
            ..
        } => Edit::Convert {
            codec: match codec {
                WireCodec::Libx264 => Codec::Libx264,
                WireCodec::Libx265 => Codec::Libx265,
                WireCodec::Vp9 => Codec::Vp9,
            },
            container: match container {
                WireContainer::Mp4 => Container::Mp4,
                WireContainer::Mkv => Container::Mkv,
                WireContainer::Webm => Container::Webm,
            },
            crf: crf.unwrap_or(23),
        },
        ExtractAudio { format, .. } => Edit::ExtractAudio {
            format: match format {
                WireAudio::Mp3 => AudioFormat::Mp3,
                WireAudio::Wav => AudioFormat::Wav,
                WireAudio::Aac => AudioFormat::Aac,
            },
        },
        Mute { .. } => Edit::Mute,
        Clarify { .. } => unreachable!(),
    };
    Ok(Parsed::Edit(edit))
}

/// Full validation: the Rust counterpart of `minicut.validate.validate`.
pub fn validate(raw: &str, probe: &WireProbe) -> Result<(), String> {
    let call = parse(raw)?;
    match to_parsed(&call)? {
        Parsed::Clarify { .. } => Ok(()),
        Parsed::Edit(e) => {
            if minicut_core::validate(e, clip(probe)) {
                Ok(())
            } else {
                Err(format!("rejected: {e:?}"))
            }
        }
    }
}
```

Rules this encodes: unknown fields are rejected (`deny_unknown_fields`). `width`/`height`/`crf` reject floats and bools, as `validate.py` does. Seconds round to the nearest ms and the factor to 0.001×. Probe duration floors to ms. Path checks match `validate.py` (empty, NUL, absolute, `..`).

Smoke: `target/debug/minicut validate --call '{"tool":"trim","input":"clip.mp4","start":0,"end":30}' --probe '{"duration_s":30.0,"width":1280,"height":720,"has_audio":true}'` prints `{"ok":false,...}`, exit 1 (whole-clip trim is a no-op).

**Commit:** `git commit -m "minicut: wire layer and validate subcommand"`

### Task 1.13: Differential parity with validate.py

**Objective:** 20,000 sampled calls get the same verdict from Rust and Python.

**Files:**
- Create: `scripts/parity_validate.py`

```python
"""Differential test: Rust `minicut validate` vs Python `minicut.validate`.

Samples (call, probe) pairs on the ms / 0.001-factor grid, including
edges, and requires the same accept/reject verdict from both. Known,
intended divergences are listed in DIVERGENCES and skipped by name.

    python scripts/parity_validate.py --bin target/debug/minicut --n 20000
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from minicut.probe import Probe  # noqa: E402
from minicut.validate import validate  # noqa: E402

TOOLS = ["trim", "cut_out", "concat", "scale", "speed", "convert", "extract_audio", "mute", "clarify"]


def ms(rng, hi_ms):
    return rng.randint(0, hi_ms) / 1000


def sample(rng: random.Random):
    dur_ms = rng.choice([1000, 10000, 30000, 30023, 60000, rng.randint(500, 120000)])
    has_video = rng.random() > 0.1
    probe = {"duration_s": dur_ms / 1000,
             "width": rng.choice([1280, 640, 1920]) if has_video else None,
             "height": rng.choice([720, 360, 1080]) if has_video else None,
             "has_audio": rng.random() > 0.2}
    tool = rng.choice(TOOLS)
    call: dict = {"tool": tool}
    if tool == "concat":
        call["inputs"] = ["a.mp4"] * rng.choice([1, 2, 3])
    elif tool != "clarify":
        call["input"] = rng.choice(["clip.mp4", "clip.mp4", "../x.mp4", "/abs.mp4", ""])
    if tool in ("trim", "cut_out"):
        edge = [0, 50, dur_ms - 50, dur_ms, dur_ms + 1]
        pick = lambda: rng.choice(edge) / 1000 if rng.random() < 0.4 else ms(rng, dur_ms + 2000)
        call["start"], call["end"] = pick(), pick()
    elif tool == "scale":
        call["width"] = rng.choice([0, 2, 639, 640, 1280, 1920, probe["width"] or 2])
        call["height"] = rng.choice([0, 2, 359, 360, 720, 1080, probe["height"] or 2])
    elif tool == "speed":
        call["factor"] = rng.choice([0.249, 0.25, 0.5, 1.0, 1.25, 2.0, 4.0, 4.001, rng.randint(200, 4100) / 1000])
    elif tool == "convert":
        call["codec"] = rng.choice(["libx264", "libx265", "vp9"])
        call["container"] = rng.choice(["mp4", "mkv", "webm"])
        if rng.random() < 0.5:
            call["crf"] = rng.choice([0, 23, 51])
    elif tool == "extract_audio":
        call["format"] = rng.choice(["mp3", "wav", "aac"])
    elif tool == "clarify":
        call["question"] = rng.choice(["Which times?", "   "])
        call["missing"] = ["start", "end"]
    return call, probe


def py_ok(call, probe) -> bool:
    p = Probe(path=Path("clip.mp4"), duration_s=probe["duration_s"], width=probe["width"],
              height=probe["height"], fps=30.0, video_codec=None, audio_codec=None,
              has_audio=probe["has_audio"], has_video=probe["width"] is not None, format_name=None)
    return validate(call, p, Path(".")) == []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rows = [sample(rng) for _ in range(args.n)]
    feed = "".join(json.dumps({"call": json.dumps(c), "probe": p}) + "\n" for c, p in rows)
    out = subprocess.run([args.bin, "validate", "--batch"], input=feed, capture_output=True,
                         text=True, check=True).stdout.splitlines()
    assert len(out) == len(rows)
    mismatches = Counter()
    examples = {}
    for (call, probe), line in zip(rows, out):
        rust, py = json.loads(line)["ok"], py_ok(call, probe)
        if rust != py:
            key = f"{call['tool']}: rust={rust} python={py}"
            mismatches[key] += 1
            examples.setdefault(key, (call, probe))
    print(f"{len(rows)} cases, {sum(mismatches.values())} mismatches")
    for k, v in mismatches.most_common():
        print(f"  {v:5d}  {k}  e.g. {json.dumps(examples[k][0])} probe={json.dumps(examples[k][1])}")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Run: `cargo build && python scripts/parity_validate.py --bin target/debug/minicut --n 20000`.
Expected (prototype result): `20000 cases, 0 mismatches`. Every tool gets both verdicts (e.g. trim 281 accept / 2018 reject, concat 1475 / 727).

**Teeth check (do once, do not commit):** set `NOOP_TRIM_TOL_MS` to 40, rebuild, and rerun. The prototype reported `16 mismatches` (e.g. trim 0.05–9.95 on a 10 s clip). Verus still passed, because spec and code share the constant. That is why this test exists: Verus proves code = spec, and this test checks spec = shipped behaviour. Restore 50.

Note: open question 5 (Python accepts an extra `inputs` key on non-concat tools) never shows up because the sampler does not generate it. Add it to the sampler together with a named skip once the question is answered.

**Commit:** `git commit -m "test: differential parity rust vs validate.py"`

### Task 1.14: Docs

**Objective:** Record Phase 1 status.

**Files:**
- Modify: `PROGRESS.md`
- Modify: `README.md` (layout list)
- Modify: `docs/rust-port.md` (status line)

Add a PROGRESS.md section with the verify count, mutation result, parity count and the commands (`scripts/verus.sh`, `scripts/mutate_core.sh`, `cargo test --workspace`, `scripts/parity_validate.py`). In the README layout list add `crates/` with one line. No user-facing behaviour changes in Phase 1.

**Commit:** `git commit -m "docs: rust port phase 1 status"`
