//! The template table: the human-review surface of the training-data
//! generator (docs/plans/2026-11-24-rust-port-phase3.md, P3-D4 and P3-D11).
//!
//! No proof can say that a phrase means an intent. Each row here records
//! that judgment once: an id, the one intent kind the wording means, the
//! number style it renders with, when it applies, and the reviewer's
//! reason. Labels never come from this file. They come from the verified
//! core (`try_label` and friends), keyed by `kind`.
//!
//! Ambiguous wording never produces a tool label (P3-D11). It lives in
//! `AMBIGUOUS` and becomes a clarify that offers our own canonical
//! phrasings, from `VOCAB`.
//!
//! Placeholders: {file} {a} {b} (times), {pct}, {w} {h}, {f} (speed factor),
//! {p} (speed percent), {codec} {container} {fmt}, {n} ("two"/"three"),
//! {times} ("twice"/"three times").

/// What a request means, independent of any tool name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum IntentKind {
    KeepRange,
    RemoveRange,
    KeepFirst,
    RemoveFirst,
    KeepLast,
    RemoveLast,
    ScaleExact,
    ScaleHalf,
    ScaleQuarter,
    SpeedFactor,
    SpeedFaster,
    SpeedAtPercent,
    Convert,
    Extract,
    Mute,
    Concat,
}

/// How the numbers in a template are written.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Num {
    /// "12.5" (seconds, `render::secs_text`)
    Secs,
    /// "1:12.5" (`render::clock_text`)
    Clock,
    /// "25%" of the clip, rendered only when `pct_of` is exact
    Pct,
    /// no time numbers
    NoTime,
}

/// When a template's wording is true of the sampled values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Applies {
    Always,
    /// start == 0 (the "first N" phrasings of a v1 conditional template)
    StartZero,
    /// start > 0
    StartNonZero,
    /// 0 < start and end < duration: "the middle a to b"
    Middle,
    FactorAbove1,
    FactorBelow1,
    TwoInputs,
    ThreeInputs,
    /// both dimensions smaller than the clip's: "scale down"
    BothSmaller,
    /// both dimensions larger than the clip's: "scale up"
    BothLarger,
    /// target is exactly 1280x720: "720p"
    Is720p,
    /// target is exactly 960x540: "540p"
    Is540p,
    /// format is mp3 (the wording names mp3)
    Mp3,
}

pub struct Template {
    /// A v1 train id (wording identical to v1) or a new "v3_" id.
    pub id: &'static str,
    pub kind: IntentKind,
    pub num: Num,
    pub text: &'static str,
    pub applies: Applies,
    pub note: &'static str,
}

impl Template {
    /// True when the label cannot be computed from the request text alone:
    /// it needs the clip's duration (last-N, percentages) or size (half,
    /// quarter). The model never sees the probe (cli.compose_prompt), so these
    /// rows are not used for tool labels (pre-GPU plan: "drop prompts that
    /// need hidden fixture information"). Their proofs stay for a future
    /// intent-level output where the app resolves them against the probe.
    pub fn needs_probe(&self) -> bool {
        self.num == Num::Pct
            || matches!(
                self.kind,
                IntentKind::KeepLast
                    | IntentKind::RemoveLast
                    | IntentKind::ScaleHalf
                    | IntentKind::ScaleQuarter
            )
    }
}

/// Vocabulary phrasing of `kind` in a given number style, for suggestions that
/// must repeat the user's own terms (a percentage stays a percentage).
pub fn vocab_in(kind: IntentKind, num: Num) -> Option<&'static Template> {
    TEMPLATES
        .iter()
        .find(|t| t.kind == kind && t.num == num && t.applies == Applies::Always)
}

const fn t(
    id: &'static str,
    kind: IntentKind,
    num: Num,
    text: &'static str,
    applies: Applies,
    note: &'static str,
) -> Template {
    Template {
        id,
        kind,
        num,
        text,
        applies,
        note,
    }
}

use Applies as A;
use IntentKind as K;
use Num as N;

pub const TEMPLATES: &[Template] = &[
    // ---- keep a range (trim) ----
    t(
        "trim_t1",
        K::KeepRange,
        N::Secs,
        "trim {file} from {a} seconds to {b} seconds",
        A::Always,
        "'trim from a to b' names what is kept",
    ),
    t(
        "trim_t2",
        K::KeepRange,
        N::Clock,
        "trim {file} from {a} to {b}",
        A::Always,
        "v1 trim_t2 wording; clock now keeps fractions (v1 _mmss truncated them)",
    ),
    t(
        "trim_t3",
        K::KeepRange,
        N::Secs,
        "keep only {a} to {b} seconds of {file}",
        A::Always,
        "'keep only' is unambiguous; canonical KeepRange phrasing",
    ),
    t(
        "trim_t4",
        K::KeepRange,
        N::Secs,
        "cut {file} down to {a}-{b} seconds",
        A::Always,
        "verb trap: 'cut ... down to' names what remains, so keep",
    ),
    t(
        "trim_t5",
        K::KeepRange,
        N::Secs,
        "trim {file} from {a} to {b}",
        A::StartNonZero,
        "v1 trim_t5 start>0 branch",
    ),
    t(
        "trim_t5",
        K::KeepFirst,
        N::Secs,
        "trim {file} to the first {b} seconds",
        A::StartZero,
        "'trim to the first N' names what remains, so keep-first",
    ),
    t(
        "train_trim_x1",
        K::KeepRange,
        N::Secs,
        "trim {file} from {a}s to {b}s",
        A::Always,
        "short-unit variant of trim_t1",
    ),
    t(
        "train_trim_x2",
        K::KeepRange,
        N::Secs,
        "cut {file} to {a}-{b}",
        A::Always,
        "verb trap: 'cut X to a-b' names the result, so keep",
    ),
    t(
        "train_trim_x3",
        K::KeepRange,
        N::Secs,
        "keep the {a} to {b} second part of {file}",
        A::Always,
        "'keep' is unambiguous",
    ),
    // ---- remove a range (cut_out) ----
    t(
        "cut_t1",
        K::RemoveRange,
        N::Secs,
        "remove {a} to {b} seconds from {file}",
        A::StartNonZero,
        "v1 cut_t1 start>0 branch; 'remove' names what goes",
    ),
    t(
        "cut_t1",
        K::RemoveFirst,
        N::Secs,
        "cut the first {b} seconds off {file}",
        A::StartZero,
        "'cut ... off' removes; canonical RemoveFirst phrasing",
    ),
    t(
        "cut_t2",
        K::RemoveRange,
        N::Secs,
        "remove {a} to {b} seconds from {file}",
        A::Always,
        "canonical RemoveRange phrasing",
    ),
    t(
        "cut_t3",
        K::RemoveRange,
        N::Secs,
        "cut out {a}-{b} from {file}",
        A::Always,
        "'cut out' removes",
    ),
    t(
        "cut_t4",
        K::RemoveRange,
        N::Secs,
        "delete the middle {a} to {b} seconds of {file}",
        A::Middle,
        "'the middle' is false unless 0 < a and b < duration",
    ),
    t(
        "cut_t5",
        K::RemoveRange,
        N::Secs,
        "drop seconds {a} through {b} of {file}",
        A::Always,
        "'drop' removes",
    ),
    t(
        "train_cut_x1",
        K::RemoveRange,
        N::Secs,
        "take out {a} to {b} seconds from {file}",
        A::Always,
        "'take out' removes",
    ),
    t(
        "train_cut_x2",
        K::RemoveRange,
        N::Secs,
        "delete seconds {a} through {b} from {file}",
        A::Always,
        "'delete' removes",
    ),
    t(
        "train_cut_x3",
        K::RemoveRange,
        N::Secs,
        "cut the section from {a} to {b} seconds out of {file}",
        A::Always,
        "'cut ... out of' removes",
    ),
    t(
        "train_cut_x4",
        K::RemoveRange,
        N::Secs,
        "remove the middle {a} to {b} seconds from {file}",
        A::Middle,
        "'the middle' needs 0 < a and b < duration",
    ),
    t(
        "v3_cut_clock",
        K::RemoveRange,
        N::Clock,
        "remove {a} to {b} from {file}",
        A::Always,
        "clock-time RemoveRange; fractions kept",
    ),
    // ---- first / last ----
    t(
        "v3_keep_first",
        K::KeepFirst,
        N::Secs,
        "keep only the first {b} seconds of {file}",
        A::Always,
        "canonical KeepFirst phrasing",
    ),
    t(
        "v3_keep_first_pct",
        K::KeepFirst,
        N::Pct,
        "keep only the first {pct} of {file}",
        A::Always,
        "'keep only' makes the percentage unambiguous",
    ),
    t(
        "v3_remove_first",
        K::RemoveFirst,
        N::Secs,
        "remove the first {b} seconds of {file}",
        A::Always,
        "'remove' names what goes",
    ),
    t(
        "v3_remove_first_pct",
        K::RemoveFirst,
        N::Pct,
        "cut the first {pct} off {file}",
        A::Always,
        "'cut ... off' removes (v1 train_cut_x_pct wording)",
    ),
    t(
        "v3_keep_last",
        K::KeepLast,
        N::Secs,
        "keep only the last {a} seconds of {file}",
        A::Always,
        "canonical KeepLast phrasing ({a} is the kept length)",
    ),
    t(
        "v3_keep_last_pct",
        K::KeepLast,
        N::Pct,
        "keep only the last {pct} of {file}",
        A::Always,
        "'keep only' makes the percentage unambiguous",
    ),
    t(
        "v3_remove_last",
        K::RemoveLast,
        N::Secs,
        "cut the last {a} seconds off {file}",
        A::Always,
        "canonical RemoveLast phrasing ({a} is the removed length)",
    ),
    t(
        "v3_remove_last_pct",
        K::RemoveLast,
        N::Pct,
        "cut the last {pct} off {file}",
        A::Always,
        "'cut ... off' removes",
    ),
    // ---- scale ----
    t(
        "scale_t1",
        K::ScaleExact,
        N::NoTime,
        "make {file} {w}x{h}",
        A::Always,
        "exact size",
    ),
    t(
        "scale_t2",
        K::ScaleExact,
        N::NoTime,
        "resize {file} to {w} by {h}",
        A::Always,
        "exact size",
    ),
    t(
        "scale_t3",
        K::ScaleExact,
        N::NoTime,
        "scale {file} to 720p",
        A::Is720p,
        "720p means 1280x720 in this tool's vocabulary",
    ),
    t(
        "scale_t3",
        K::ScaleExact,
        N::NoTime,
        "resize {file} to {w}x{h}",
        A::Always,
        "v1 scale_t3 non-720p branch",
    ),
    t(
        "train_scale_x1",
        K::ScaleExact,
        N::NoTime,
        "scale {file} down to {w}x{h}",
        A::BothSmaller,
        "v1 picked up/down by tuple order; now only when both dims shrink",
    ),
    t(
        "train_scale_x1",
        K::ScaleExact,
        N::NoTime,
        "scale {file} up to {w}x{h}",
        A::BothLarger,
        "only when both dims grow",
    ),
    t(
        "train_scale_x2",
        K::ScaleExact,
        N::NoTime,
        "make {file} 540p",
        A::Is540p,
        "540p means 960x540 here",
    ),
    t(
        "v3_half_size",
        K::ScaleHalf,
        N::NoTime,
        "shrink {file} to half size",
        A::Always,
        "label from scale_half (exact, even)",
    ),
    t(
        "v3_half_size_2",
        K::ScaleHalf,
        N::NoTime,
        "make {file} half its size",
        A::Always,
        "label from scale_half",
    ),
    t(
        "v3_quarter_size",
        K::ScaleQuarter,
        N::NoTime,
        "shrink {file} to a quarter of its size",
        A::Always,
        "label from scale_quarter (exact, even)",
    ),
    t(
        "v3_quarter_size_2",
        K::ScaleQuarter,
        N::NoTime,
        "make {file} a quarter of its size",
        A::Always,
        "label from scale_quarter",
    ),
    // ---- speed ----
    t(
        "speed_t1",
        K::SpeedFactor,
        N::NoTime,
        "speed up {file} by {f}x",
        A::FactorAbove1,
        "faster",
    ),
    t(
        "speed_t1",
        K::SpeedFactor,
        N::NoTime,
        "slow {file} down to {f}x",
        A::FactorBelow1,
        "slower",
    ),
    t(
        "speed_t2",
        K::SpeedFactor,
        N::NoTime,
        "make {file} {f} times as fast",
        A::FactorAbove1,
        "faster",
    ),
    t(
        "speed_t2",
        K::SpeedFactor,
        N::NoTime,
        "play {file} at {f}x speed",
        A::FactorBelow1,
        "slower",
    ),
    t(
        "speed_t4",
        K::SpeedFactor,
        N::NoTime,
        "run {file} at {f}x speed",
        A::Always,
        "either way",
    ),
    t(
        "train_speed_x2",
        K::SpeedFactor,
        N::NoTime,
        "change {file} to play at {f}x",
        A::Always,
        "either way",
    ),
    t(
        "train_speed_x1",
        K::SpeedFaster,
        N::NoTime,
        "make {file} {p}% faster",
        A::Always,
        "label from speed_percent_faster: 1 + p/100",
    ),
    t(
        "v3_speed_up_pct",
        K::SpeedFaster,
        N::NoTime,
        "speed {file} up by {p}%",
        A::Always,
        "label from speed_percent_faster",
    ),
    t(
        "train_speed_x1",
        K::SpeedAtPercent,
        N::NoTime,
        "slow {file} to {p}% speed",
        A::Always,
        "label from speed_at_percent: p/100",
    ),
    t(
        "v3_play_at_pct",
        K::SpeedAtPercent,
        N::NoTime,
        "play {file} at {p}% speed",
        A::Always,
        "label from speed_at_percent",
    ),
    // ---- convert ----
    t(
        "convert_t1",
        K::Convert,
        N::NoTime,
        "convert {file} to {container} with {codec}",
        A::Always,
        "",
    ),
    t(
        "convert_t2",
        K::Convert,
        N::NoTime,
        "re-encode {file} as {codec} in a .{container} file",
        A::Always,
        "",
    ),
    t(
        "convert_t3",
        K::Convert,
        N::NoTime,
        "save {file} as {container} using {codec}",
        A::Always,
        "",
    ),
    t(
        "train_convert_x1",
        K::Convert,
        N::NoTime,
        "save {file} as a .{container} file with {codec}",
        A::Always,
        "",
    ),
    t(
        "train_convert_x2",
        K::Convert,
        N::NoTime,
        "change {file} to {container} format using {codec}",
        A::Always,
        "",
    ),
    // ---- extract audio ----
    t(
        "extract_t1",
        K::Extract,
        N::NoTime,
        "extract the audio from {file} as {fmt}",
        A::Always,
        "",
    ),
    t(
        "extract_t2",
        K::Extract,
        N::NoTime,
        "save the audio from {file} to {fmt}",
        A::Always,
        "",
    ),
    t(
        "extract_t3",
        K::Extract,
        N::NoTime,
        "pull the soundtrack out of {file} as an {fmt} file",
        A::Always,
        "",
    ),
    t(
        "train_extract_x1",
        K::Extract,
        N::NoTime,
        "get the mp3 from {file}",
        A::Mp3,
        "names mp3",
    ),
    t(
        "train_extract_x2",
        K::Extract,
        N::NoTime,
        "rip the mp3 out of {file}",
        A::Mp3,
        "names mp3",
    ),
    t(
        "train_extract_x3",
        K::Extract,
        N::NoTime,
        "save {file}'s audio as mp3",
        A::Mp3,
        "names mp3",
    ),
    t(
        "train_extract_x5",
        K::Extract,
        N::NoTime,
        "make an mp3 from {file}",
        A::Mp3,
        "names mp3",
    ),
    t(
        "train_extract_x6",
        K::Extract,
        N::NoTime,
        "pull the mp3 from {file}",
        A::Mp3,
        "names mp3",
    ),
    t(
        "train_extract_x8",
        K::Extract,
        N::NoTime,
        "isolate the audio in {file} as mp3",
        A::Mp3,
        "names mp3",
    ),
    t(
        "train_extract_x7",
        K::Extract,
        N::NoTime,
        "audio only from {file} as mp3",
        A::Mp3,
        "names mp3; v1 trained on it",
    ),
    t(
        "train_extract_x9",
        K::Extract,
        N::NoTime,
        "grab the audio from {file} in mp3",
        A::Mp3,
        "names mp3",
    ),
    t(
        "train_extract_x10",
        K::Extract,
        N::NoTime,
        "turn {file} into an mp3",
        A::Mp3,
        "an mp3 from a video is its audio",
    ),
    // ---- mute ----
    t(
        "mute_t1",
        K::Mute,
        N::NoTime,
        "mute {file}",
        A::Always,
        "canonical Mute phrasing",
    ),
    t(
        "mute_t2",
        K::Mute,
        N::NoTime,
        "remove the audio from {file}",
        A::Always,
        "",
    ),
    t(
        "mute_t3",
        K::Mute,
        N::NoTime,
        "make a silent version of {file}",
        A::Always,
        "",
    ),
    t(
        "mute_t4",
        K::Mute,
        N::NoTime,
        "drop the audio track from {file}",
        A::Always,
        "",
    ),
    t(
        "train_mute_x1",
        K::Mute,
        N::NoTime,
        "silence the audio in {file}",
        A::Always,
        "",
    ),
    t(
        "train_mute_x2",
        K::Mute,
        N::NoTime,
        "make {file} silent",
        A::Always,
        "",
    ),
    t(
        "train_mute_x3",
        K::Mute,
        N::NoTime,
        "turn off the sound in {file}",
        A::Always,
        "",
    ),
    t(
        "train_mute_x4",
        K::Mute,
        N::NoTime,
        "remove all sound from {file}",
        A::Always,
        "",
    ),
    t(
        "train_mute_x5",
        K::Mute,
        N::NoTime,
        "drop the sound from {file}",
        A::Always,
        "",
    ),
    t(
        "train_mute_x7",
        K::Mute,
        N::NoTime,
        "strip the audio track from {file}",
        A::Always,
        "",
    ),
    t(
        "train_mute_x8",
        K::Mute,
        N::NoTime,
        "remove the audio track from {file}",
        A::Always,
        "",
    ),
    t(
        "train_mute_x9",
        K::Mute,
        N::NoTime,
        "kill the audio in {file}",
        A::Always,
        "",
    ),
    t(
        "train_mute_x10",
        K::Mute,
        N::NoTime,
        "drop all audio from {file}",
        A::Always,
        "",
    ),
    // ---- concat (inputs are copies of {file}; order is the meaning) ----
    t(
        "concat_t1",
        K::Concat,
        N::NoTime,
        "join {file} to itself",
        A::TwoInputs,
        "",
    ),
    t(
        "concat_t1",
        K::Concat,
        N::NoTime,
        "join {file} to itself three times",
        A::ThreeInputs,
        "",
    ),
    t(
        "concat_t2",
        K::Concat,
        N::NoTime,
        "make one video from {file} twice",
        A::TwoInputs,
        "",
    ),
    t(
        "concat_t2",
        K::Concat,
        N::NoTime,
        "make one video from {file} three times in a row",
        A::ThreeInputs,
        "",
    ),
    t(
        "concat_t3",
        K::Concat,
        N::NoTime,
        "concatenate {file} with itself",
        A::TwoInputs,
        "v1 used it for 3 inputs too; 'with itself' says two, so two only",
    ),
    t(
        "train_concat_x1",
        K::Concat,
        N::NoTime,
        "put {n} copies of {file} together",
        A::Always,
        "{n} is two/three",
    ),
    t(
        "train_concat_x2",
        K::Concat,
        N::NoTime,
        "merge {file} with itself",
        A::TwoInputs,
        "'with itself' says two",
    ),
    t(
        "train_concat_x3",
        K::Concat,
        N::NoTime,
        "append {file} to itself",
        A::TwoInputs,
        "'to itself' says two",
    ),
    t(
        "train_concat_x4",
        K::Concat,
        N::NoTime,
        "loop {file} {times}",
        A::Always,
        "{times} is twice/three times",
    ),
    t(
        "train_concat_x5",
        K::Concat,
        N::NoTime,
        "make a video with {file} {times} in a row",
        A::Always,
        "",
    ),
];

/// v1 wording deliberately not used, with the reason (shown in docs/templates.md).
pub const EXCLUDED: &[(&str, &str, &str)] = &[
    (
        "concat_t4",
        "combine {file} with itself into one file",
        "word-for-word the held-out template concat_h3: training on it leaks held-out wording",
    ),
    (
        "train_extract_x4",
        "export the soundtrack as mp3",
        "names no file; the app binds files, but the request text alone cannot say which",
    ),
    (
        "speed_t3",
        "speed {file} up to {f}x",
        "v1 used it for factors below 1 ('speed up to 0.8x'), which contradicts itself",
    ),
];

/// How the second reading of ambiguous wording is chosen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Other {
    /// The keep/remove reading of the same span: `minicut_core::flip`. The
    /// core proves the two readings disagree at every instant.
    Flip,
    /// The other reading is something MiniCut cannot do; only `reading`
    /// is offered, and the question says what is not supported.
    Unsupported(&'static str),
}

pub struct Ambiguous {
    pub id: &'static str,
    pub text: &'static str,
    pub num: Num,
    /// The first reading.
    pub reading: IntentKind,
    /// A percentage the wording implies without saying it ("half" = 50).
    pub fixed_pct: Option<u64>,
    pub other: Other,
    pub note: &'static str,
}

pub const AMBIGUOUS: &[Ambiguous] = &[
    Ambiguous {
        id: "v3_amb_trim_first_pct",
        text: "trim the first {pct} of {file}",
        num: N::Pct,
        reading: K::KeepFirst,
        fixed_pct: None,
        other: Other::Flip,
        note: "v1 labelled this keep-first; 'trim the first 25%' is read both ways",
    },
    Ambiguous {
        id: "v3_amb_trim_last_pct",
        text: "trim the last {pct} of {file}",
        num: N::Pct,
        reading: K::KeepLast,
        fixed_pct: None,
        other: Other::Flip,
        note: "same as above, at the end",
    },
    Ambiguous {
        id: "v3_amb_trim_first_half",
        text: "trim the first half of {file}",
        num: N::NoTime,
        reading: K::KeepFirst,
        fixed_pct: Some(50),
        other: Other::Flip,
        note: "keep or remove the first half",
    },
    Ambiguous {
        id: "v3_amb_trim_last_half",
        text: "trim the last half of {file}",
        num: N::NoTime,
        reading: K::KeepLast,
        fixed_pct: Some(50),
        other: Other::Flip,
        note: "keep or remove the last half",
    },
    Ambiguous {
        id: "v3_amb_cut_first",
        text: "cut the first {b} seconds of {file}",
        num: N::Secs,
        reading: K::RemoveFirst,
        fixed_pct: None,
        other: Other::Flip,
        note: "without 'off' or 'down to', 'cut the first N seconds of X' is read both ways",
    },
    Ambiguous {
        id: "v3_amb_quiet",
        text: "make {file} quiet",
        num: N::NoTime,
        reading: K::Mute,
        fixed_pct: None,
        other: Other::Unsupported("lower the volume"),
        note: "v1 train_mute_x6; 'quiet' may mean quieter, which MiniCut cannot do",
    },
];

/// Canonical phrasing per intent kind: the vocabulary offered as suggestions
/// in clarify questions (P3-D11). Each entry names a row of `TEMPLATES`; a
/// test checks the row exists, has that kind, and applies unconditionally.
pub const VOCAB: &[(IntentKind, &str)] = &[
    (K::KeepRange, "trim_t3"),
    (K::RemoveRange, "cut_t2"),
    (K::KeepFirst, "v3_keep_first"),
    (K::RemoveFirst, "v3_remove_first"),
    (K::KeepLast, "v3_keep_last"),
    (K::RemoveLast, "v3_remove_last"),
    (K::ScaleExact, "scale_t1"),
    (K::SpeedFactor, "speed_t4"),
    (K::Convert, "convert_t1"),
    (K::Extract, "extract_t1"),
    (K::Mute, "mute_t1"),
    (K::Concat, "train_concat_x1"),
];

/// The vocabulary template for a kind (with the given number style, when the
/// kind has several canonical styles).
pub fn vocab(kind: IntentKind) -> Option<&'static Template> {
    let id = VOCAB.iter().find(|(k, _)| *k == kind)?.1;
    TEMPLATES
        .iter()
        .find(|t| t.id == id && t.kind == kind && t.applies == Applies::Always)
}

/// docs/templates.md: the reviewed table, generated (a test keeps it in sync).
pub fn markdown() -> String {
    let mut s = String::from(
        "# MiniCut template table (generated)\n\n\
         Generated by `minicut templates --markdown` from `crates/minicut/src/templates.rs`.\n\
         Do not edit by hand. Each row records one human judgment: this wording means this\n\
         intent kind. Labels never come from here; they come from the verified core.\n\n\
         ## Templates\n\nThe label column says whether the label follows from the text alone (`text`) or needs the\nclip's duration or size (`probe`). Only `text` rows are used for tool labels.\n\n| id | kind | numbers | applies when | label | wording | why |\n|---|---|---|---|---|---|---|\n",
    );
    for t in TEMPLATES {
        s += &format!(
            "| `{}` | {:?} | {:?} | {:?} | {} | {} | {} |\n",
            t.id,
            t.kind,
            t.num,
            t.applies,
            if t.needs_probe() { "probe" } else { "text" },
            t.text,
            t.note
        );
    }
    s += "\n## Ambiguous wording: always a clarify, never a label (P3-D11)\n\n\
          | id | wording | numbers | first reading | other reading | why |\n|---|---|---|---|---|---|\n";
    for a in AMBIGUOUS {
        let other = match a.other {
            Other::Flip => "flip (keep ↔ remove, same span)".to_string(),
            Other::Unsupported(what) => format!("unsupported: {what}"),
        };
        let numbers = match a.fixed_pct {
            Some(p) => format!("{:?} ({p}%)", a.num),
            None => format!("{:?}", a.num),
        };
        s += &format!(
            "| `{}` | {} | {} | {:?} | {} | {} |\n",
            a.id, a.text, numbers, a.reading, other, a.note
        );
    }
    s += "\n## Vocabulary offered in clarify questions\n\n| kind | template | wording |\n|---|---|---|\n";
    for (k, id) in VOCAB {
        let text = vocab(*k).map(|t| t.text).unwrap_or("(missing)");
        s += &format!("| {k:?} | `{id}` | {text} |\n");
    }
    s += "\n## Excluded v1 wording\n\n| id | wording | why |\n|---|---|---|\n";
    for (id, text, why) in EXCLUDED {
        s += &format!("| `{id}` | {text} | {why} |\n");
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, BTreeSet};

    const HELDOUT: &[&str] = &[
        "clarify_h1",
        "clarify_h2",
        "concat_h1",
        "concat_h2",
        "concat_h3",
        "convert_h1",
        "convert_h2",
        "convert_h3",
        "cut_h1",
        "cut_h2",
        "cut_h3",
        "extract_h1",
        "extract_h2",
        "extract_h3",
        "mute_h1",
        "mute_h2",
        "mute_h3",
        "mute_h4",
        "scale_h1",
        "scale_h2",
        "scale_h3",
        "speed_h1",
        "speed_h2",
        "speed_h3",
        "trim_h1",
        "trim_h2",
        "trim_h3",
    ];

    #[test]
    fn no_heldout_ids() {
        for t in TEMPLATES {
            assert!(!HELDOUT.contains(&t.id), "{}", t.id);
        }
        for a in AMBIGUOUS {
            assert!(!HELDOUT.contains(&a.id), "{}", a.id);
        }
    }

    #[test]
    fn every_vocab_entry_is_an_unconditional_template_of_its_kind() {
        for (k, id) in VOCAB {
            let v = vocab(*k).unwrap_or_else(|| panic!("no vocab template for {k:?} ({id})"));
            assert_eq!(v.kind, *k);
        }
        // every time intent has a vocabulary phrasing, so every flip has a suggestion
        for k in [
            K::KeepRange,
            K::RemoveRange,
            K::KeepFirst,
            K::RemoveFirst,
            K::KeepLast,
            K::RemoveLast,
        ] {
            assert!(vocab(k).is_some(), "{k:?}");
        }
    }

    #[test]
    fn ambiguous_wording_never_appears_as_a_template() {
        let texts: BTreeSet<&str> = TEMPLATES.iter().map(|t| t.text).collect();
        for a in AMBIGUOUS {
            assert!(
                !texts.contains(a.text),
                "ambiguous wording is also a template: {}",
                a.text
            );
        }
        for (_, text, _) in EXCLUDED {
            assert!(
                !texts.contains(text),
                "excluded wording is still a template: {text}"
            );
        }
    }

    #[test]
    fn halves_carry_their_percentage() {
        for a in AMBIGUOUS {
            assert_eq!(a.text.contains("half"), a.fixed_pct == Some(50), "{}", a.id);
            if a.num == Num::NoTime && a.other == Other::Flip {
                assert!(a.fixed_pct.is_some(), "{}: a flip needs a span", a.id);
            }
        }
    }

    #[test]
    fn percentage_readings_can_be_suggested_as_percentages() {
        for k in [K::KeepFirst, K::RemoveFirst, K::KeepLast, K::RemoveLast] {
            assert!(
                vocab_in(k, Num::Pct).is_some(),
                "{k:?} has no percentage phrasing"
            );
            assert!(
                vocab_in(k, Num::Secs).is_some(),
                "{k:?} has no seconds phrasing"
            );
        }
    }

    #[test]
    fn flip_readings_are_time_intents() {
        let time = [
            K::KeepRange,
            K::RemoveRange,
            K::KeepFirst,
            K::RemoveFirst,
            K::KeepLast,
            K::RemoveLast,
        ];
        for a in AMBIGUOUS {
            if a.other == Other::Flip {
                assert!(time.contains(&a.reading), "{}", a.id);
            }
        }
    }

    #[test]
    fn every_kind_has_at_least_two_templates() {
        let mut n: BTreeMap<IntentKind, usize> = BTreeMap::new();
        for t in TEMPLATES {
            *n.entry(t.kind).or_default() += 1;
        }
        for (k, c) in &n {
            assert!(*c >= 2, "{k:?} has only {c} template(s)");
        }
        assert_eq!(n.len(), 16, "every IntentKind has templates: {n:?}");
    }

    #[test]
    fn placeholders_match_number_style() {
        for t in TEMPLATES {
            let has_time = t.text.contains("{a}") || t.text.contains("{b}");
            match t.num {
                Num::Secs | Num::Clock => assert!(has_time, "{} {}", t.id, t.text),
                Num::Pct => assert!(t.text.contains("{pct}") && !has_time, "{}", t.id),
                Num::NoTime => assert!(!has_time && !t.text.contains("{pct}"), "{}", t.id),
            }
            assert!(
                t.text.contains("{file}"),
                "every template names the file: {}",
                t.id
            );
        }
    }

    #[test]
    fn rows_are_unique() {
        let mut seen = BTreeSet::new();
        for t in TEMPLATES {
            assert!(
                seen.insert((t.id, t.text)),
                "duplicate row {} {}",
                t.id,
                t.text
            );
        }
    }

    /// Sentinel rendering matching scripts/export_v1_templates.py's ARGS.
    fn sentinel(t: &Template) -> String {
        let (a, b) = match (t.num, t.applies) {
            (Num::Clock, Applies::StartZero) => ("0:00", "3:42"),
            (Num::Clock, _) => ("1:51", "3:42"),
            (_, Applies::StartZero) => ("0", "222"),
            _ => ("111", "222"),
        };
        let (w, h) = match t.applies {
            Applies::Is720p => ("1280", "720"),
            Applies::Is540p => ("960", "540"),
            Applies::BothLarger => ("1920", "1080"),
            _ => ("333", "444"),
        };
        let f = if t.applies == Applies::FactorBelow1 {
            "0.8"
        } else {
            "1.25"
        };
        let p = if t.kind == IntentKind::SpeedAtPercent {
            "80"
        } else {
            "25"
        };
        let (n, times) = if t.applies == Applies::ThreeInputs {
            ("three", "three times")
        } else {
            ("two", "twice")
        };
        t.text
            .replace("{file}", "clip.mp4")
            .replace("{a}", a)
            .replace("{b}", b)
            .replace("{w}", w)
            .replace("{h}", h)
            .replace("{f}", f)
            .replace("{p}", p)
            .replace("{codec}", "libx264")
            .replace("{container}", "mkv")
            .replace("{fmt}", "mp3")
            .replace("{n}", n)
            .replace("{times}", times)
    }

    #[test]
    fn reused_v1_ids_keep_v1_wording() {
        let golden: BTreeMap<String, Vec<String>> =
            serde_json::from_str(include_str!("../tests/golden/v1_templates.json")).unwrap();
        let mut checked = 0;
        for t in TEMPLATES.iter().filter(|t| !t.id.starts_with("v3_")) {
            let v1 = golden
                .get(t.id)
                .unwrap_or_else(|| panic!("{} is not a v1 train id", t.id));
            let ours = sentinel(t);
            assert!(
                v1.contains(&ours),
                "{}: {ours:?} is not v1 wording {v1:?}",
                t.id
            );
            checked += 1;
        }
        assert!(checked > 60, "checked {checked}");
    }
}
