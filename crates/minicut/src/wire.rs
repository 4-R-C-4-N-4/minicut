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
/// An accepted clarify comes back as `Parsed::Clarify` for the caller to show.
pub fn validate(raw: &str, probe: &WireProbe) -> Result<Parsed, String> {
    let call = parse(raw)?;
    let parsed = to_parsed(&call)?;
    if let Parsed::Edit(e) = parsed {
        if !minicut_core::validate(e, clip(probe)) {
            return Err(format!("rejected: {e:?}"));
        }
    }
    Ok(parsed)
}
