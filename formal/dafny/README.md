# MiniCut, formally — a Dafny dictionary for v1

A formal dictionary for MiniCut v1: the 8 tools + `clarify` from
`config/tools.yaml`, written as Dafny datatypes with the validity
conditions from `minicut/validate.py` and the expected-output
conditions from `minicut/verify.py`.

Ivy's framing: applications are language translation. Formalizing
gives you a dictionary — one canonical formal meaning that many
natural-language phrasings map to — which is an interface for model
training as well as verifiable code. The model translates NL into a
value of type `Tool`; the wrapper only ever runs values that satisfy
`ValidSingle` / `ValidConcat`. This file is that dictionary, proved.

## What it proves

`minicut.dfy` (Dafny 4.10.0, **32 verified, 0 errors** — see
"How to verify" below):

| Claim | Dafny name | Grounded in |
|---|---|---|
| trim + cut_out partition the input: their durations sum to the input duration | `TrimCutOutPartition` | `verify.py` trim/cut_out durations |
| speed divides duration by the factor, stays positive | `SpeedDuration` | `verify.py` speed check |
| scale / convert preserve duration (scale also fixes the requested dims) | `ScalePreservesDuration`, `ConvertPreservesDurationAndDims` | `verify.py` scale/convert checks |
| extract_audio and mute are mirror images: opposite surviving stream | `ExtractVsMuteMirror` | `verify.py` stream checks |
| concat duration is the sum of its inputs, in order; concat needs 2+ inputs | `ConcatDurationIsSum`, `ConcatNeedsTwo` | `verify.py` concat check |
| the `minicut_chain` grows by exactly one token per edit, prefix preserved | `ChainAppend`, `ChainAppendTwice`, `ChainRecordsOpsInOrder` | `chain.py`, `wrapper.py` |
| a bumped output name is never the input name; a safe candidate never overwrites | `BumpedNameDiffers`, `CandidateIsSafe` | `wrapper.py` `_publish`, `build.py` `default_output` |
| clarify's `missing` is exactly required-minus-provided, and empty when complete | `ClarifyExact`, `ClarifyEmptyWhenComplete`, `ClarifyNamesEveryMissingArg` | `spec.md` §5, `config/tools.yaml` |

Two structural facts are enforced by construction, not by a
runtime check:

- **Allowlists are datatypes.** `Codec` has exactly 3 values
  (`libx264`, `libx265`, `vp9`), `Container` 3, `AudioFormat` 3.
  There is no value of type `Codec` outside the allowlist, so an
  unlisted codec cannot be constructed as a `Tool`.
- **Arg names are the type.** `Speed` takes `factor`, `Concat`
  takes `inputs`. The v1 baseline failures (right tool, wrong field
  name: `"speed"` for `"factor"`, `"input"` for `"inputs"`) cannot
  exist as a value of type `Tool`.

## How to verify

Install Dafny, then from the repo root:

```bash
dafny verify formal/dafny/minicut.dfy
```

Expected output: `Dafny program verifier finished with 32 verified, 0 errors`.

Setup options (Dafny 4.10.0 was used to check this file):

- **VS Code:** install the extension `dafny-lang.ide-vscode` — it
  offers to download Dafny the first time you open a `.dfy` file.
  On Linux/Mac it still needs .NET 8.
- **macOS:** `brew install dafny`, then `dafny verify formal/dafny/minicut.dfy`.
- **Linux:** download the Dafny release zip for your platform and
  run its `dafny` binary; it needs the .NET 8 runtime
  (`dotnet-runtime-8.0`).

## Cross-check against Python (no Dafny needed)

`tests/test_formal_dafny.py` encodes the Dafny validity rules and
duration formulas in Python and checks they agree with
`minicut.validate`, `minicut.chain`, `minicut.wrapper._publish` and
`minicut.build.default_output` on sampled cases. Run it with the rest of the suite:

```bash
python -m pytest tests/ -q
```

If the wrapper changes and this file does not (or vice versa), that
test fails first. It is not a Dafny verifier — it guards against
drift between the spec and the code.

## The grammar interface

The dictionary is no longer just a spec the model ignores. The
GBNF grammar the model generates under is produced from the same
`config/tools.yaml` by `minicut/grammar/generate_grammar.py`:

- Each tool gets its own alternative, with exactly the field names
  in `tools.yaml`, in canonical order, with the right JSON types
  and enum values. Required fields are required; `output`, `crf`
  and `missing` are optional.
- A wrong field name cannot be emitted at all. The v1 failures
  (`"speed"` for `"factor"`, `"input"` for `"inputs"`) are stopped
  at generation instead of being caught by `validate.py` after.
- Ranges, `start < end`, even dims and everything needing the
  probe stay in `validate.py` — the grammar cannot see the clip.

`tests/test_grammar.py` checks the three-way agreement:
the committed `.gbnf` is exactly what the generator emits from
`tools.yaml`, each tool's rule accepts exactly its fields, the
few-shot examples conform, and the Dafny `Tool` datatype has
exactly the args `tools.yaml` lists. The llama.cpp server is the
thing that actually parses the grammar; the eval harness measures
whether the stricter grammar helps or hurts the model (baseline_v3
95.5% was measured with the old, loose grammar).

## What it does NOT prove

- **That ffmpeg produces the modelled output.** Dafny models the
  *specification* the wrapper checks with ffprobe; it does not
  execute ffmpeg, read pixels, or hear tones. Frame-colour and
  sine-tone checks stay in `factory/` and `minicut/verify.py`.
- **File-system facts.** Path safety (relative, inside the workdir,
  no `..`), output existence, and the `_v2` bump loop's termination
  are not modelled. `NeverOverwrite` is proved as a string property;
  the loop that finds a free name is not.
- **Exact `minicut_chain` token strings.** Dafny models the chain
  as a sequence of tool tokens and proves append-only growth. The
  exact tokens (`trim:12-45`, `cutout:12-45`, `speed:2`) are number
  formatting in `chain.py` and are not modelled here.
- **Tolerances.** `verify.py` checks durations within ±0.15s
  (±0.3s for concat/speed). Dafny uses exact real arithmetic; the
  tolerance is an implementation check, not part of the spec.
- **The model's translation.** Dafny proves what a `Tool` value
  *means* once you have one. It does not prove the model picks the
  right one — that is what the eval (`eval/`, baseline_v3 95.5%)
  measures, and what the contrast pairs in the dictionary are for.

## Known divergences from the Python wrapper

These are deliberate spec choices, not oversights. Each is also
asserted in `tests/test_formal_dafny.py` where it can be.

1. **`extract_audio` on a silent input: Dafny says invalid,
   Python says valid.** There is no audio to extract. `validate.py`
   does not check `hasAudio` for this tool today; a silent input
   passes validation and fails verification instead. The Dafny rule
   states the intended contract. Fixing `validate.py` to match is a
   separate, small change — this PR does not make it.
2. **Clarify's `missing` list: Dafny requires exactness, Python
   does not check it.** `validate.py` only checks the question is
   non-empty. The exactness rule (`missing` == required − provided)
   is the strictest contract in the file because baseline_v3 got it
   wrong 4 times. Enforcing it in the wrapper is future work.
3. **Clarify's question: Dafny checks non-empty, Python also
   strips whitespace.** A whitespace-only question passes Dafny
   and fails Python. Minor; the Dafny rule is the weaker one here.
4. **Scale dims: Dafny requires `>= 2 and even`.** `tools.yaml`
   says min 2; `validate.py` says positive and even. The two agree
   (the smallest even positive int is 2), so this is a restatement,
   not a divergence.
5. **No-op edits: Dafny says valid, Python rejects them.**
   `validate.py` now refuses edits that would change nothing — a
   full-clip trim, a same-size scale, speed 1x. Dafny's
   `ValidSingle` is about the call being well-formed against the
   clip, not about it being useful, so it does not include that
   policy. Whether the no-op rule belongs in the model is open;
   the test asserts the disagreement either way.

## What this PR does not touch

- No change to any tool's meaning, the wrapper, the factory, the
  frozen eval (`eval/freeze.json`), the training dataset, or any
  baseline result. The GBNF grammar *is* changed: it is now
  generated from `config/tools.yaml` and pins field names per tool
  (see "The grammar interface" above). Baseline_v3 95.5% was
  measured with the old grammar; a re-baseline is needed to compare
  under the new one.
- No change to `minicut/` runtime code except the grammar itself
  and its generator (`minicut/grammar/generate_grammar.py`), plus
  two quoting fixes in `config/tools.yaml` (`list[str]` is not
  valid unquoted YAML in a flow mapping — nothing in the repo had
  ever parsed the file before this generator).
- The dictionary draft that motivated this file lives outside the
  repo at `workspace/formal-verification/dictionary-minicut-v1.md`;
  this file is the proved subset of it (the lilguy half of that
  draft is not in this PR).
