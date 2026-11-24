// minicut-core: the verified heart of MiniCut. No I/O, no floats, no strings.
// Times are integer milliseconds; speed is an integer per-mille factor.
use vstd::prelude::*;

verus! {

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

} // verus!
