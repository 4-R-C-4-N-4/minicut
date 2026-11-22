// minicut.dfy — a formal dictionary for MiniCut v1.
//
// What this is: the 8 v1 tools + clarify from `config/tools.yaml`,
// written as Dafny datatypes with the validity conditions from
// `minicut/validate.py` and the expected-output conditions from
// `minicut/verify.py`. One canonical formal meaning per tool. The model
// translates natural language into a value of type `Tool`; the wrapper
// only ever runs values that satisfy `ValidSingle` / `ValidConcat`.
//
// Source of truth for args and ranges: `config/tools.yaml` (v1) and
// `spec.md` §5. Grammar (`minicut/grammar/minicut.gbnf`) enforces the
// *shape* of a tool call; this file pins down the *meaning*.
//
// What it proves (see README.md in this directory for the full list):
//   - trim / cut_out partition the input: their durations sum to the
//     input duration, and "what survives?" separates them.
//   - speed, scale, convert, mute and extract_audio preserve duration
//     (with the exact postconditions from `verify.py`).
//   - concat duration is the sum of its inputs, in order.
//   - the `minicut_chain` grows by exactly one token per edit, never
//     rewrites history, and the wrapper never overwrites its input.
//   - clarify's `missing` list is *exactly* required-minus-provided.
//     That exactness is the strictest contract in the file, because
//     the remaining baseline_v3 failures were clarify's missing list.
//
// What it does NOT prove (honest scope, see README.md):
//   - that ffmpeg actually produces the modelled output. Dafny models
//     the *specification* the wrapper checks with ffprobe; it does not
//     execute ffmpeg, read pixels, or hear tones.
//   - file-system facts: path safety, output existence, the `_v2`
//     bump loop's termination. `NeverOverwrite` is proved as a
//     string property; the loop that finds a free name is not.
//   - the exact `minicut_chain` token strings (`trim:12-45` etc.).
//     Dafny models the chain as a sequence of tool tokens and proves
//     append-only growth; the number formatting lives in
//     `minicut/chain.py`.
//
// Verify with: `dafny verify formal/dafny/minicut.dfy`
// (Dafny 4.10.0, checked 2026-11-22. See README.md for setup.)

// ---------------------------------------------------------------------------
// 1. Allowlists — enforced by construction.
//
// `Codec`, `Container` and `AudioFormat` have exactly the values in
// `config/tools.yaml`. There is no value of type `Codec` outside the
// allowlist, so an unlisted codec cannot be constructed as a `Tool`.
// That is the formal version of `validate.py`'s allowlist checks.
// ---------------------------------------------------------------------------
datatype Codec = Libx264 | Libx265 | Vp9
datatype Container = Mp4 | Mkv | Webm
datatype AudioFormat = Mp3 | Wav | Aac

// ---------------------------------------------------------------------------
// 2. The dictionary: one variant per tool, arg names exactly as in
//    `config/tools.yaml`.
//    - Speed takes `factor`, not `speed`; Concat takes `inputs`, not
//      `input`. Those were the v1 baseline failures (55.0%
//      verification, arg accuracy 48%): right tool, wrong field name.
//      As a type, that failure cannot exist.
//    - `crf` has a default (23) in `tools.yaml`, so it is present in
//      the datatype but not in `RequiredArgs` (see §7).
//    - Clarify is not an edit: it carries a question and a missing
//      list, and produces no file.
// ---------------------------------------------------------------------------
datatype Tool =
  Trim(start: real, end: real)
  | CutOut(start: real, end: real)
  | Concat(inputs: seq<string>)
  | Scale(width: int, height: int)
  | Speed(factor: real)
  | Convert(codec: Codec, container: Container, crf: int)
  | ExtractAudio(format: AudioFormat)
  | Mute
  | Clarify(question: string, missing: seq<string>)

// A probed input clip: what `ffprobe` reports and the wrapper checks.
// `duration` is in seconds (real, as in `probe.py`); dims are in
// pixels; the stream flags are what `verify.py` checks.
datatype Clip = Clip(
  duration: real,
  width: int,
  height: int,
  hasAudio: bool,
  hasVideo: bool
)

// ---------------------------------------------------------------------------
// 3. Validity — the semantic checks from `minicut/validate.py`.
// ---------------------------------------------------------------------------

// trim / cut_out: `0 <= start < end <= duration` (validate.py).
predicate ValidInterval(start: real, end: real, duration: real) {
  0.0 <= start && start < end && end <= duration
}

// scale: dims must be even for yuv420p (spec §5, validate.py), and
// `tools.yaml` sets min 2. The two together give `>= 2 and even`:
// the smallest even positive int is 2, so this matches both sources.
predicate ValidScaleDims(width: int, height: int) {
  2 <= width && 2 <= height && width % 2 == 0 && height % 2 == 0
}

// Single-input tools only. Concat takes a list of clips (see
// `ValidConcat` below) and Clarify is not an edit, so both are
// `false` here by design, not by omission.
predicate ValidSingle(t: Tool, c: Clip) {
  match t
  case Trim(s, e) => ValidInterval(s, e, c.duration)
  case CutOut(s, e) => ValidInterval(s, e, c.duration)
  case Concat(_) => false
  case Scale(w, h) => ValidScaleDims(w, h)
  case Speed(f) => 0.25 <= f <= 4.0
  case Convert(_, _, crf) => 0 <= crf <= 51
  // extract_audio needs audio to extract. This is stricter than
  // `validate.py`, which does not check `hasAudio` for this tool:
  // a silent input passes validation today and fails verification
  // instead. The Dafny rule states the intended contract; see
  // README.md ("Known divergences").
  case ExtractAudio(_) => c.hasAudio
  case Mute => true
  case Clarify(_, _) => false
}

// Concat: at least 2 inputs, and the clip list is in the same order
// as the `inputs` list, so index i of one is index i of the other.
// Order is the meaning here: the duration sum does not capture it,
// the sequence does.
predicate ValidConcat(t: Tool, clips: seq<Clip>) {
  match t
  case Concat(inputs) =>
    |inputs| == |clips| && |clips| >= 2 &&
    (forall i :: 0 <= i < |clips| ==> 0.0 <= clips[i].duration)
  case _ => false
}

// Clarify: a non-empty question. Python also strips whitespace
// (`validate.py`); Dafny models non-empty only — see README.md.
predicate ValidClarify(t: Tool) {
  match t
  case Clarify(q, _) => q != ""
  case _ => false
}

// ---------------------------------------------------------------------------
// 4. Expected outputs — the postconditions from `minicut/verify.py`.
//    These functions are total: for Concat / Clarify they return the
//    input value as a placeholder, and the lemmas in §5 only use them
//    under `ValidSingle`, where they have their stated meaning.
// ---------------------------------------------------------------------------

// Duration after a single-input edit. trim keeps [start, end];
// cut_out keeps the rest; speed divides by the factor; everything
// else preserves duration (`verify.py` checks each of these).
// The precondition is the validity rule: the formula for `Speed`
// divides by `factor`, so it is only defined for valid calls
// (where `0.25 <= factor`, hence `factor != 0`).
function DurationAfter(t: Tool, c: Clip): real
  requires ValidSingle(t, c)
{  match t
  case Trim(s, e) => e - s
  case CutOut(s, e) => c.duration - (e - s)
  case Concat(_) => c.duration
  case Scale(_, _) => c.duration
  case Speed(f) => c.duration / f
  case Convert(_, _, _) => c.duration
  case ExtractAudio(_) => c.duration
  case Mute => c.duration
  case Clarify(_, _) => c.duration
}

function WidthAfter(t: Tool, c: Clip): int {
  match t
  case Scale(w, _) => w
  case _ => c.width
}

function HeightAfter(t: Tool, c: Clip): int {
  match t
  case Scale(_, h) => h
  case _ => c.height
}

// trim / cut_out / scale / speed / convert re-encode audio with AAC
// (or copy it), so audio survives iff it was there. extract_audio
// keeps only audio; mute keeps only video. See §5 lemmas.
function HasAudioAfter(t: Tool, c: Clip): bool {
  match t
  case ExtractAudio(_) => true
  case Mute => false
  case _ => c.hasAudio
}

function HasVideoAfter(t: Tool, c: Clip): bool {
  match t
  case ExtractAudio(_) => false
  case _ => c.hasVideo
}

// Concat duration: the sum of the input durations, in order.
function SumClipDurations(clips: seq<Clip>): real
  decreases clips
{
  if |clips| == 0 then 0.0 else clips[0].duration + SumClipDurations(clips[1..])
}

// ---------------------------------------------------------------------------
// 5. Lemmas — what the dictionary proves.
// ---------------------------------------------------------------------------

// --- trim vs cut_out: the partition (dictionary D1/D2). ---
// The separator test is "what survives?": trim keeps only
// [start, end]; cut_out keeps everything except [start, end].
// Together they partition the input, which is why the eval's
// trim/cut_out confusion (1 case in baseline_v3) and cut_out
// start/end inversions (4 cases) are meaning errors, not arg errors.
lemma TrimCutOutPartition(c: Clip, s: real, e: real)
  requires ValidSingle(Trim(s, e), c)
  requires ValidSingle(CutOut(s, e), c)
  ensures DurationAfter(Trim(s, e), c) + DurationAfter(CutOut(s, e), c) == c.duration
  ensures 0.0 < DurationAfter(Trim(s, e), c)
  ensures 0.0 <= DurationAfter(CutOut(s, e), c)
{
}

// --- speed: duration divides by the factor. ---
lemma SpeedDuration(c: Clip, f: real)
  requires ValidSingle(Speed(f), c)
  requires 0.0 < c.duration
  ensures DurationAfter(Speed(f), c) == c.duration / f
  ensures 0.0 < DurationAfter(Speed(f), c)
{
}

// --- scale / convert / mute / extract_audio preserve duration. ---
lemma ScalePreservesDuration(c: Clip, w: int, h: int)
  requires ValidSingle(Scale(w, h), c)
  ensures DurationAfter(Scale(w, h), c) == c.duration
  ensures WidthAfter(Scale(w, h), c) == w
  ensures HeightAfter(Scale(w, h), c) == h
  ensures HasAudioAfter(Scale(w, h), c) == c.hasAudio
{
}

lemma ConvertPreservesDurationAndDims(c: Clip, codec: Codec, container: Container, crf: int)
  requires ValidSingle(Convert(codec, container, crf), c)
  ensures DurationAfter(Convert(codec, container, crf), c) == c.duration
  ensures WidthAfter(Convert(codec, container, crf), c) == c.width
  ensures HeightAfter(Convert(codec, container, crf), c) == c.height
{
}

// --- extract_audio and mute are mirror images. ---
// Same input, opposite surviving stream: the cleanest contrast pair
// in v1 (dictionary D7/D8).
lemma ExtractVsMuteMirror(c: Clip)
  requires ValidSingle(ExtractAudio(Mp3), c)
  requires ValidSingle(Mute, c)
  ensures HasAudioAfter(ExtractAudio(Mp3), c)
  ensures !HasVideoAfter(ExtractAudio(Mp3), c)
  ensures DurationAfter(ExtractAudio(Mp3), c) == c.duration
  ensures !HasAudioAfter(Mute, c)
  ensures HasVideoAfter(Mute, c) == c.hasVideo
  ensures DurationAfter(Mute, c) == c.duration
{
}

// --- concat: duration is the sum, in order. ---
lemma SumClipDurationsNonneg(clips: seq<Clip>)
  requires forall i :: 0 <= i < |clips| ==> 0.0 <= clips[i].duration
  ensures 0.0 <= SumClipDurations(clips)
{
  if |clips| == 0 {
  } else {
    assert forall i :: 0 <= i < |clips[1..]| ==> 0.0 <= clips[1..][i].duration;
    SumClipDurationsNonneg(clips[1..]);
  }
}

lemma ConcatDurationIsSum(a: Clip, b: Clip)
  requires 0.0 <= a.duration && 0.0 <= b.duration
  ensures SumClipDurations([a, b]) == a.duration + b.duration
{
  assert SumClipDurations([a, b]) == a.duration + SumClipDurations([b]);
  assert SumClipDurations([b]) == b.duration + SumClipDurations([]);
}

lemma ConcatNeedsTwo(t: Tool, clips: seq<Clip>)
  requires ValidConcat(t, clips)
  ensures |clips| >= 2
  ensures 0.0 <= SumClipDurations(clips)
{
  SumClipDurationsNonneg(clips);
}

// ---------------------------------------------------------------------------
// 6. The chain — edit history in metadata, not in the filename.
//    (`minicut/chain.py`: `minicut_chain` tag, tokens joined by `;`;
//    `minicut/wrapper.py`: the wrapper appends one token per edit and
//    never modifies the input in place.)
//
//    Dafny models the chain as a sequence of tool tokens and proves
//    append-only growth. The exact token strings (`trim:12-45`,
//    `cutout:12-45`, `speed:2.0`, ...) are number formatting in
//    `chain.py` and are not modelled here: `ToolToken` is the tool
//    name, which is also the token the wrapper uses for the output
//    filename stem (`build.py`: `stem_<tool>.<ext>`).
// ---------------------------------------------------------------------------
function ToolToken(t: Tool): string {
  match t
  case Trim(_, _) => "trim"
  case CutOut(_, _) => "cutout"
  case Concat(_) => "concat"
  case Scale(_, _) => "scale"
  case Speed(_) => "speed"
  case Convert(_, _, _) => "convert"
  case ExtractAudio(_) => "extract"
  case Mute => "mute"
  case Clarify(_, _) => "clarify"
}

function ChainAppend(chain: seq<string>, op: string): seq<string>
  ensures |ChainAppend(chain, op)| == |chain| + 1
  ensures ChainAppend(chain, op)[|chain|] == op
  ensures ChainAppend(chain, op)[..|chain|] == chain
{
  chain + [op]
}

lemma ChainAppendTwice(chain: seq<string>, op1: string, op2: string)
  ensures |ChainAppend(ChainAppend(chain, op1), op2)| == |chain| + 2
  ensures ChainAppend(ChainAppend(chain, op1), op2)[|chain|] == op1
  ensures ChainAppend(ChainAppend(chain, op1), op2)[|chain| + 1] == op2
  ensures ChainAppend(ChainAppend(chain, op1), op2)[..|chain|] == chain
{
}

// A two-op edit is two dictionary entries applied in order, and the
// chain records both, in order, with no rewrite. A v2 multi-step plan
// is a *sequence* of entries — the dictionary needs no new entry type
// for it, only sequencing.
lemma ChainRecordsOpsInOrder(chain: seq<string>, t1: Tool, t2: Tool)
  ensures |ChainAppend(ChainAppend(chain, ToolToken(t1)), ToolToken(t2))| == |chain| + 2
{
  ChainAppendTwice(chain, ToolToken(t1), ToolToken(t2));
}

// ---------------------------------------------------------------------------
// 7. Never overwrite — the wrapper owns output names.
//    (`minicut/wrapper.py`: `_resolve_output` derives the name from
//    the input stem + one op token, bumps `_v2`, `_v3`, ... on
//    collision, and bumps away from any input path too. The model's
//    `output` field is ignored entirely.)
//
//    Dafny proves the string property the bump relies on: a bumped
//    name is never the input name. The loop that searches for a free
//    name, and file-system existence, are not modelled.
// ---------------------------------------------------------------------------
predicate NeverOverwrite(input: string, output: string) {
  input != output
}

lemma BumpedNameDiffers(name: string)
  ensures NeverOverwrite(name, name + "_v2")
{
  assert |name + "_v2"| == |name| + 3;
}

lemma BumpRescues(input: string)
  ensures NeverOverwrite(input, input + "_v2")
{
  BumpedNameDiffers(input);
}

// If the derived candidate is neither the input nor already taken,
// using it is safe. (`existing` is the set of names already present
// in the workdir.)
lemma CandidateIsSafe(input: string, candidate: string, existing: set<string>)
  requires candidate != input && candidate !in existing
  ensures NeverOverwrite(input, candidate)
{
}

// ---------------------------------------------------------------------------
// 8. Clarify — the unknown entry, with the strictest contract.
//    (`spec.md` §5: missing arg -> clarification, not a guessed time.)
//
//    `RequiredArgs` is per tool *kind*, from `config/tools.yaml`
//    (`crf` excluded: it has a default of 23). `ClarifyExact` says
//    the `missing` list is *exactly* required-minus-provided —
//    no missing arg omitted, no present arg listed, no extras.
//    That exactness is what baseline_v3 got wrong 4 times.
// ---------------------------------------------------------------------------
datatype ToolKind =
  TrimK | CutOutK | ConcatK | ScaleK | SpeedK |
  ConvertK | ExtractAudioK | MuteK | ClarifyK

function KindOf(t: Tool): ToolKind {
  match t
  case Trim(_, _) => TrimK
  case CutOut(_, _) => CutOutK
  case Concat(_) => ConcatK
  case Scale(_, _) => ScaleK
  case Speed(_) => SpeedK
  case Convert(_, _, _) => ConvertK
  case ExtractAudio(_) => ExtractAudioK
  case Mute => MuteK
  case Clarify(_, _) => ClarifyK
}

function RequiredArgs(k: ToolKind): set<string> {
  match k
  case TrimK => {"input", "start", "end"}
  case CutOutK => {"input", "start", "end"}
  case ConcatK => {"inputs"}
  case ScaleK => {"input", "width", "height"}
  case SpeedK => {"input", "factor"}
  case ConvertK => {"input", "codec", "container"}
  case ExtractAudioK => {"input", "format"}
  case MuteK => {"input"}
  case ClarifyK => {"question"}
}

function SeqToSet(s: seq<string>): set<string>
  decreases s
{
  if |s| == 0 then {} else {s[0]} + SeqToSet(s[1..])
}

lemma SeqToSetEmpty(s: seq<string>)
  ensures SeqToSet(s) == {} <==> s == []
{
  if |s| == 0 {
  } else {
    assert s[0] in SeqToSet(s);
  }
}

lemma SeqToSetSubset(s: seq<string>, t: set<string>)
  requires forall i :: 0 <= i < |s| ==> s[i] in t
  ensures SeqToSet(s) <= t
{
  if |s| == 0 {
  } else {
    assert forall i :: 0 <= i < |s[1..]| ==> s[1..][i] in t;
    SeqToSetSubset(s[1..], t);
  }
}

predicate ClarifyExact(kind: ToolKind, provided: set<string>, missing: seq<string>) {
  SeqToSet(missing) == RequiredArgs(kind) - provided
}

lemma ClarifyMissingOnlyRequired(kind: ToolKind, provided: set<string>, missing: seq<string>)
  requires ClarifyExact(kind, provided, missing)
  ensures SeqToSet(missing) <= RequiredArgs(kind)
{
}

lemma ClarifyEmptyWhenComplete(kind: ToolKind, provided: set<string>, missing: seq<string>)
  requires ClarifyExact(kind, provided, missing)
  requires RequiredArgs(kind) <= provided
  ensures missing == []
{
  SeqToSetEmpty(missing);
}

lemma ClarifyNamesEveryMissingArg(kind: ToolKind, provided: set<string>, missing: seq<string>, arg: string)
  requires ClarifyExact(kind, provided, missing)
  requires arg in RequiredArgs(kind) && arg !in provided
  ensures arg in SeqToSet(missing)
{
}
