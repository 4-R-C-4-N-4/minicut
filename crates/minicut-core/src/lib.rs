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

} // verus!
