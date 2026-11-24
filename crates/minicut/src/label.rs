//! `minicut label`: intent-first training rows (plan Task 3.17, P3-D1..D11).
//!
//! A row is made by sampling an *intent* (what the request means), choosing a
//! reviewed template of that intent kind (templates.rs), rendering the user
//! text, and taking the label ONLY from the verified core: `try_label`,
//! `pct_of`, `scale_half`/`scale_quarter`, the speed functions, `flip` and
//! `missing`. Ambiguous wording becomes a clarify that offers our own
//! vocabulary (P3-D11). Eval-freeze policy is NOT applied here; the Python
//! gate (training/build_verified.py) drops, never edits (P3-D3).

use crate::clarify::{self, Request, REQUESTS};
use crate::naming::default_output;
use crate::prompts::SYSTEM_PROMPT;
use crate::render::{clock_text, pct_text, secs_text};
use crate::rng::SplitMix64;
use crate::templates::{
    vocab, vocab_in, Ambiguous, Applies, IntentKind as K, Num, Other, Template, AMBIGUOUS,
    TEMPLATES,
};
use minicut_core::{self as mc, Clip, Edit, Intent, Kind};
use std::collections::BTreeSet;

pub const GENERATOR: &str = concat!("minicut-label/", env!("CARGO_PKG_VERSION"));

/// Training fixtures (training/generate_dataset.py TRAIN_FIXTURES; the
/// reserved long_60s_1080p is deliberately absent). Durations are the nominal
/// lengths, never above the probed ones, so ranges stay inside the real clip.
pub struct Fixture {
    pub name: &'static str,
    pub clip: Clip,
}

pub const FIXTURES: &[Fixture] = &[
    Fixture {
        name: "base_30s",
        clip: Clip {
            duration_ms: 30000,
            width: Some(1280),
            height: Some(720),
            has_audio: true,
        },
    },
    Fixture {
        name: "short_10s",
        clip: Clip {
            duration_ms: 10000,
            width: Some(1280),
            height: Some(720),
            has_audio: true,
        },
    },
    Fixture {
        name: "silent_30s",
        clip: Clip {
            duration_ms: 30000,
            width: Some(1280),
            height: Some(720),
            has_audio: false,
        },
    },
    Fixture {
        name: "portrait_30s",
        clip: Clip {
            duration_ms: 30000,
            width: Some(720),
            height: Some(1280),
            has_audio: true,
        },
    },
];

/// training/generate_dataset.py FILENAMES.
pub const FILENAMES: &[&str] = &[
    "clip.mp4",
    "video.mp4",
    "my_video.mp4",
    "input.mp4",
    "clip2.mp4",
    "footage.mp4",
    "a.mp4",
    "b.mp4",
    "test.mp4",
    "myclip.mp4",
    "vid.mp4",
    "movie.mp4",
    "recording.mp4",
    "clip_01.mp4",
    "final.mp4",
    "edit.mp4",
    "source.mp4",
    "cam.mp4",
    "take1.mp4",
    "scene.mp4",
    "clip3.mp4",
    "raw.mp4",
    "cut.mp4",
    "part1.mp4",
    "v1.mp4",
    "demo.mp4",
    "file.mp4",
    "media.mp4",
    "video1.mp4",
    "output1.mp4",
];

const SCALE_PAIRS: &[(u32, u32)] = &[
    (960, 540),
    (800, 450),
    (1024, 576),
    (320, 180),
    (480, 270),
    (1440, 810),
    (640, 480),
    (360, 640),
    (426, 240),
];
const SPEED_PM: &[u64] = &[600, 750, 800, 1200, 1250, 1750, 2250, 2500, 3000, 3500];
const FASTER_PCT: &[u64] = &[20, 25, 75, 100, 150, 200];
const AT_PCT: &[u64] = &[60, 70, 75, 80, 90];
const TIME_PCT: &[u64] = &[10, 20, 25, 40, 50, 75];
const CRF_DEFAULT: u8 = 23; // no template states a CRF, so the label uses the default

/// Rows per tool; "clarify" is split into argument-missing and
/// reading-ambiguous (plan open question 2: the pilot's mix and size).
pub struct Counts {
    pub per_tool: Vec<(&'static str, usize)>,
    pub clarify_missing: usize,
    pub clarify_reading: usize,
}

impl Counts {
    pub fn pilot_matched() -> Self {
        Counts {
            per_tool: vec![
                ("trim", 306),
                ("cut_out", 368),
                ("concat", 223),
                ("scale", 255),
                ("speed", 300),
                ("convert", 193),
                ("extract_audio", 244),
                ("mute", 120),
            ],
            clarify_missing: 250,
            clarify_reading: 60,
        }
    }
}

// ---------------------------------------------------------------- JSON ------
/// A JSON value written exactly like Python's `json.dumps` defaults
/// (", " / ": " separators, ASCII-only, insertion order), so targets
/// tokenize like the pilot's.
pub enum J {
    S(String),
    F(f64),
    I(u64),
    A(Vec<J>),
    O(Vec<(&'static str, J)>),
    Null,
}

fn py_str(s: &str) -> String {
    let mut o = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => o += "\\\"",
            '\\' => o += "\\\\",
            '\n' => o += "\\n",
            '\r' => o += "\\r",
            '\t' => o += "\\t",
            c if (c as u32) < 0x20 || (c as u32) > 0x7e => {
                let mut buf = [0u16; 2];
                for u in c.encode_utf16(&mut buf) {
                    o += &format!("\\u{u:04x}");
                }
            }
            c => o.push(c),
        }
    }
    o + "\""
}

pub fn dumps(j: &J) -> String {
    match j {
        J::S(s) => py_str(s),
        // serde_json writes the shortest round-trip repr with a ".0" for
        // integral values, as Python's repr(float) does.
        J::F(x) => serde_json::to_string(x).expect("finite"),
        J::I(n) => n.to_string(),
        J::Null => "null".to_string(),
        J::A(xs) => format!("[{}]", xs.iter().map(dumps).collect::<Vec<_>>().join(", ")),
        J::O(kv) => format!(
            "{{{}}}",
            kv.iter()
                .map(|(k, v)| format!("{}: {}", py_str(k), dumps(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn secs(ms: u64) -> J {
    J::F(ms as f64 / 1000.0)
}

// ------------------------------------------------------------ labels --------
fn codec_name(c: mc::Codec) -> &'static str {
    match c {
        mc::Codec::Libx264 => "libx264",
        mc::Codec::Libx265 => "libx265",
        mc::Codec::Vp9 => "vp9",
    }
}
fn container_name(c: mc::Container) -> &'static str {
    match c {
        mc::Container::Mp4 => "mp4",
        mc::Container::Mkv => "mkv",
        mc::Container::Webm => "webm",
    }
}

/// The assistant JSON for a verified edit, in grammar field order, with the
/// application's output name (naming.rs, golden-tested against Python).
pub fn edit_json(e: Edit, file: &str) -> J {
    let input = ("input", J::S(file.to_string()));
    let out = |tool: &str, container: Option<&str>, format: Option<&str>| {
        (
            "output",
            J::S(default_output(tool, Some(file), container, format)),
        )
    };
    match e {
        Edit::Trim { start_ms, end_ms } => J::O(vec![
            ("tool", J::S("trim".into())),
            input,
            ("start", secs(start_ms)),
            ("end", secs(end_ms)),
            out("trim", None, None),
        ]),
        Edit::CutOut { start_ms, end_ms } => J::O(vec![
            ("tool", J::S("cut_out".into())),
            input,
            ("start", secs(start_ms)),
            ("end", secs(end_ms)),
            out("cut_out", None, None),
        ]),
        Edit::Concat { n_inputs } => J::O(vec![
            ("tool", J::S("concat".into())),
            (
                "inputs",
                J::A((0..n_inputs).map(|_| J::S(file.to_string())).collect()),
            ),
            ("output", J::S(default_output("concat", None, None, None))),
        ]),
        Edit::Scale { width, height } => J::O(vec![
            ("tool", J::S("scale".into())),
            input,
            ("width", J::I(width as u64)),
            ("height", J::I(height as u64)),
            out("scale", None, None),
        ]),
        Edit::Speed { factor_pm } => J::O(vec![
            ("tool", J::S("speed".into())),
            input,
            ("factor", J::F(factor_pm as f64 / 1000.0)),
            out("speed", None, None),
        ]),
        Edit::Convert {
            codec,
            container,
            crf,
        } => J::O(vec![
            ("tool", J::S("convert".into())),
            input,
            ("codec", J::S(codec_name(codec).into())),
            ("container", J::S(container_name(container).into())),
            ("crf", J::I(crf as u64)),
            out("convert", Some(container_name(container)), None),
        ]),
        Edit::ExtractAudio { format } => {
            let f = match format {
                mc::AudioFormat::Mp3 => "mp3",
                mc::AudioFormat::Wav => "wav",
                mc::AudioFormat::Aac => "aac",
            };
            J::O(vec![
                ("tool", J::S("extract_audio".into())),
                input,
                ("format", J::S(f.into())),
                out("extract_audio", None, Some(f)),
            ])
        }
        Edit::Mute => J::O(vec![
            ("tool", J::S("mute".into())),
            input,
            out("mute", None, None),
        ]),
    }
}

pub fn clarify_json(question: &str, missing: &[&str]) -> J {
    J::O(vec![
        ("tool", J::S("clarify".into())),
        ("question", J::S(question.to_string())),
        (
            "missing",
            J::A(missing.iter().map(|m| J::S(m.to_string())).collect()),
        ),
    ])
}

// ------------------------------------------------------------- rendering ----
/// Values a template may mention.
#[derive(Default, Clone)]
pub struct Fill {
    pub file: String,
    pub a: Option<u64>,
    pub b: Option<u64>,
    pub pct: Option<u64>,
    pub w: Option<u32>,
    pub h: Option<u32>,
    pub f_pm: Option<u64>,
    pub p: Option<u64>,
    pub codec: Option<&'static str>,
    pub container: Option<&'static str>,
    pub fmt: Option<&'static str>,
    pub n: Option<u64>,
}

pub fn render(text: &str, num: Num, v: &Fill) -> String {
    let time = |ms: Option<u64>| match (num, ms) {
        (Num::Clock, Some(x)) => clock_text(x),
        (_, Some(x)) => secs_text(x),
        (_, None) => String::new(),
    };
    let mut s = text.replace("{file}", &v.file);
    let subs: [(&str, Option<String>); 12] = [
        ("{a}", v.a.map(|x| time(Some(x)))),
        ("{b}", v.b.map(|x| time(Some(x)))),
        ("{pct}", v.pct.map(pct_text)),
        ("{w}", v.w.map(|x| x.to_string())),
        ("{h}", v.h.map(|x| x.to_string())),
        ("{f}", v.f_pm.map(secs_text)),
        ("{p}", v.p.map(|x| x.to_string())),
        ("{codec}", v.codec.map(String::from)),
        ("{container}", v.container.map(String::from)),
        ("{fmt}", v.fmt.map(String::from)),
        (
            "{n}",
            v.n.map(|n| if n == 3 { "three" } else { "two" }.to_string()),
        ),
        (
            "{times}",
            v.n.map(|n| if n == 3 { "three times" } else { "twice" }.to_string()),
        ),
    ];
    for (k, val) in subs {
        if s.contains(k) {
            s = s.replace(
                k,
                &val.unwrap_or_else(|| panic!("template needs {k}: {text}")),
            );
        }
    }
    assert!(!s.contains('{'), "unfilled placeholder in {s:?}");
    s
}

fn applies(a: Applies, start: u64, end: u64, c: &Clip, v: &Fill) -> bool {
    let (w, h) = (v.w.unwrap_or(0), v.h.unwrap_or(0));
    let (cw, ch) = (c.width.unwrap_or(0), c.height.unwrap_or(0));
    match a {
        Applies::Always => true,
        Applies::StartZero => start == 0,
        Applies::StartNonZero => start > 0,
        Applies::Middle => start > 0 && end < c.duration_ms,
        Applies::FactorAbove1 => v.f_pm.is_some_and(|f| f > 1000),
        Applies::FactorBelow1 => v.f_pm.is_some_and(|f| f < 1000),
        Applies::TwoInputs => v.n == Some(2),
        Applies::ThreeInputs => v.n == Some(3),
        Applies::BothSmaller => w < cw && h < ch,
        Applies::BothLarger => w > cw && h > ch,
        Applies::Is720p => (w, h) == (1280, 720),
        Applies::Is540p => (w, h) == (960, 540),
        Applies::Mp3 => v.fmt == Some("mp3"),
    }
}

/// The time span an intent talks about, as (start, end) in source ms.
fn span(i: Intent, c: &Clip) -> (u64, u64) {
    match i {
        Intent::KeepRange { start_ms, end_ms } | Intent::RemoveRange { start_ms, end_ms } => {
            (start_ms, end_ms)
        }
        Intent::KeepFirst { ms } | Intent::RemoveFirst { ms } => (0, ms),
        Intent::KeepLast { ms } | Intent::RemoveLast { ms } => {
            (c.duration_ms.saturating_sub(ms), c.duration_ms)
        }
    }
}

fn intent_kind(i: Intent) -> K {
    match i {
        Intent::KeepRange { .. } => K::KeepRange,
        Intent::RemoveRange { .. } => K::RemoveRange,
        Intent::KeepFirst { .. } => K::KeepFirst,
        Intent::RemoveFirst { .. } => K::RemoveFirst,
        Intent::KeepLast { .. } => K::KeepLast,
        Intent::RemoveLast { .. } => K::RemoveLast,
    }
}

fn make_intent(k: K, a: u64, b: u64) -> Intent {
    match k {
        K::KeepRange => Intent::KeepRange {
            start_ms: a,
            end_ms: b,
        },
        K::RemoveRange => Intent::RemoveRange {
            start_ms: a,
            end_ms: b,
        },
        K::KeepFirst => Intent::KeepFirst { ms: b },
        K::RemoveFirst => Intent::RemoveFirst { ms: b },
        K::KeepLast => Intent::KeepLast { ms: a },
        K::RemoveLast => Intent::RemoveLast { ms: a },
        _ => unreachable!("not a time intent: {k:?}"),
    }
}

/// Placeholder values for a time intent: ranges use {a}..{b}; first-N uses
/// {b} = N; last-N uses {a} = N.
fn time_fill(i: Intent, file: &str) -> Fill {
    let mut v = Fill {
        file: file.to_string(),
        ..Default::default()
    };
    match i {
        Intent::KeepRange { start_ms, end_ms } | Intent::RemoveRange { start_ms, end_ms } => {
            v.a = Some(start_ms);
            v.b = Some(end_ms);
        }
        Intent::KeepFirst { ms } | Intent::RemoveFirst { ms } => v.b = Some(ms),
        Intent::KeepLast { ms } | Intent::RemoveLast { ms } => v.a = Some(ms),
    }
    v
}

// --------------------------------------------------------------- rows -------
pub struct Row {
    pub prompt: String,
    pub assistant: String,
    pub meta: J,
}

impl Row {
    pub fn line(&self) -> String {
        dumps(&J::O(vec![(
            "messages",
            J::A(vec![
                J::O(vec![
                    ("role", J::S("system".into())),
                    ("content", J::S(SYSTEM_PROMPT.into())),
                ]),
                J::O(vec![
                    ("role", J::S("user".into())),
                    ("content", J::S(self.prompt.clone())),
                ]),
                J::O(vec![
                    ("role", J::S("assistant".into())),
                    ("content", J::S(self.assistant.clone())),
                ]),
            ]),
        )]))
    }
}

fn meta(kind: &str, tool: &str, fixture: Option<&str>, tid: &str, intent: String, seed: u64) -> J {
    J::O(vec![
        ("kind", J::S(kind.into())),
        ("tool", J::S(tool.into())),
        ("fixture", fixture.map_or(J::Null, |f| J::S(f.into()))),
        ("template_id", J::S(tid.into())),
        ("intent", J::S(intent)),
        ("seed", J::I(seed)),
        ("generator", J::S(GENERATOR.into())),
    ])
}

fn templates_of(kinds: &[K]) -> Vec<&'static Template> {
    TEMPLATES
        .iter()
        .filter(|t| kinds.contains(&t.kind))
        .collect()
}

/// Sample one time intent of a kind on `c`, using the template's number style.
fn sample_time_intent(r: &mut SplitMix64, t: &Template, c: &Clip) -> Option<Intent> {
    let d = c.duration_ms;
    let grid =
        |r: &mut SplitMix64, lo: u64, hi: u64| -> u64 { lo + 500 * r.below((hi - lo) / 500 + 1) };
    if t.num == Num::Pct {
        let pct = *r.pick(TIME_PCT);
        let ms = mc::pct_of(d, pct)?;
        return Some(make_intent(t.kind, ms, ms));
    }
    let (min_keep, min_rest) = (2000, 1000); // v1: trims keep >= 2 s; cuts leave >= 1 s
    match t.kind {
        K::KeepRange | K::RemoveRange => {
            let s = grid(r, 0, d - 2500);
            let e = grid(r, s + 500, d);
            let keep = if t.kind == K::KeepRange {
                e - s
            } else {
                d - (e - s)
            };
            let floor = if t.kind == K::KeepRange {
                min_keep
            } else {
                min_rest
            };
            (keep >= floor).then(|| make_intent(t.kind, s, e))
        }
        K::KeepFirst | K::KeepLast => Some(make_intent(
            t.kind,
            grid(r, min_keep, d - 500),
            grid(r, min_keep, d - 500),
        )),
        K::RemoveFirst | K::RemoveLast => Some(make_intent(
            t.kind,
            grid(r, 500, d - min_rest),
            grid(r, 500, d - min_rest),
        )),
        _ => None,
    }
    .map(|i| match i {
        // make_intent reads `a` for last-N and `b` for first-N; keep them equal
        Intent::KeepLast { ms }
        | Intent::RemoveLast { ms }
        | Intent::KeepFirst { ms }
        | Intent::RemoveFirst { ms } => make_intent(t.kind, ms, ms),
        other => other,
    })
}

pub struct Generator {
    rng: SplitMix64,
    seed: u64,
    seen: BTreeSet<(String, String)>,
    pub resampled: usize,
}

impl Generator {
    pub fn new(seed: u64) -> Self {
        Generator {
            rng: SplitMix64::new(seed),
            seed,
            seen: BTreeSet::new(),
            resampled: 0,
        }
    }

    fn file(&mut self) -> String {
        // arg-bearing tools keep clip.mp4 half the time (eval style), as v1 did
        if self.rng.below(2) == 0 {
            "clip.mp4".into()
        } else {
            (*self.rng.pick(FILENAMES)).into()
        }
    }

    fn fixture(&mut self, need_audio: bool) -> &'static Fixture {
        loop {
            let f = self.rng.pick(FIXTURES);
            if !need_audio || f.clip.has_audio {
                return f;
            }
        }
    }

    fn accept(&mut self, row: Row) -> Option<Row> {
        if self
            .seen
            .insert((row.prompt.clone(), row.assistant.clone()))
        {
            Some(row)
        } else {
            None
        }
    }

    /// One tool row, or None to resample (template did not apply, label
    /// rejected by the verified core, or a duplicate).
    fn tool_row(&mut self, tool: &str) -> Option<Row> {
        let kinds: &[K] = match tool {
            "trim" => &[K::KeepRange, K::KeepFirst, K::KeepLast],
            "cut_out" => &[K::RemoveRange, K::RemoveFirst, K::RemoveLast],
            "scale" => &[K::ScaleExact, K::ScaleHalf, K::ScaleQuarter],
            "speed" => &[K::SpeedFactor, K::SpeedFaster, K::SpeedAtPercent],
            "convert" => &[K::Convert],
            "extract_audio" => &[K::Extract],
            "mute" => &[K::Mute],
            "concat" => &[K::Concat],
            _ => panic!("unknown tool {tool}"),
        };
        // only templates whose label follows from the text (needs_probe)
        let pool: Vec<&Template> = templates_of(kinds)
            .into_iter()
            .filter(|t| !t.needs_probe())
            .collect();
        let t = *self.rng.pick(&pool);
        let fx = self.fixture(matches!(t.kind, K::Extract | K::Mute));
        let c = fx.clip;
        let file = self.file();
        let (edit, fill, intent_text, start, end): (Edit, Fill, String, u64, u64) = match t.kind {
            K::KeepRange
            | K::RemoveRange
            | K::KeepFirst
            | K::RemoveFirst
            | K::KeepLast
            | K::RemoveLast => {
                let i = sample_time_intent(&mut self.rng, t, &c)?;
                let e = mc::try_label(i, c)?; // the label: proved meaning + accepted
                let (s, en) = span(i, &c);
                let mut v = time_fill(i, &file);
                if t.num == Num::Pct {
                    let len = match i {
                        Intent::KeepFirst { ms }
                        | Intent::RemoveFirst { ms }
                        | Intent::KeepLast { ms }
                        | Intent::RemoveLast { ms } => ms,
                        _ => return None,
                    };
                    // recover the percentage that pct_of made exact
                    let pct = (1..=100).find(|p| mc::pct_of(c.duration_ms, *p) == Some(len))?;
                    v.pct = Some(pct);
                    v.a = None;
                    v.b = None;
                }
                (e, v, format!("{i:?}"), s, en)
            }
            K::ScaleExact | K::ScaleHalf | K::ScaleQuarter => {
                let (w, h) = match t.kind {
                    K::ScaleExact => *self.rng.pick(SCALE_PAIRS),
                    K::ScaleHalf => mc::scale_half(c.width?, c.height?)?,
                    _ => mc::scale_quarter(c.width?, c.height?)?,
                };
                let e = Edit::Scale {
                    width: w,
                    height: h,
                };
                let v = Fill {
                    file: file.clone(),
                    w: Some(w),
                    h: Some(h),
                    ..Default::default()
                };
                (e, v, format!("{:?}({w}x{h})", t.kind), 0, 0)
            }
            K::SpeedFactor | K::SpeedFaster | K::SpeedAtPercent => {
                let mut v = Fill {
                    file: file.clone(),
                    ..Default::default()
                };
                let f = match t.kind {
                    K::SpeedFactor => *self.rng.pick(SPEED_PM),
                    K::SpeedFaster => {
                        let p = *self.rng.pick(FASTER_PCT);
                        v.p = Some(p);
                        mc::speed_percent_faster(p)?
                    }
                    _ => {
                        let p = *self.rng.pick(AT_PCT);
                        v.p = Some(p);
                        mc::speed_at_percent(p)?
                    }
                };
                v.f_pm = Some(f);
                (
                    Edit::Speed { factor_pm: f },
                    v,
                    format!("{:?}({f}pm)", t.kind),
                    0,
                    0,
                )
            }
            K::Convert => {
                let (codec, container) = loop {
                    let codec =
                        *self
                            .rng
                            .pick(&[mc::Codec::Libx264, mc::Codec::Libx265, mc::Codec::Vp9]);
                    let container = *self.rng.pick(&[
                        mc::Container::Mp4,
                        mc::Container::Mkv,
                        mc::Container::Webm,
                    ]);
                    // WebM only allows VP8/VP9/AV1 video (v1 rule)
                    if container != mc::Container::Webm || codec == mc::Codec::Vp9 {
                        break (codec, container);
                    }
                };
                let v = Fill {
                    file: file.clone(),
                    codec: Some(codec_name(codec)),
                    container: Some(container_name(container)),
                    ..Default::default()
                };
                (
                    Edit::Convert {
                        codec,
                        container,
                        crf: CRF_DEFAULT,
                    },
                    v,
                    format!(
                        "Convert({}, {})",
                        codec_name(codec),
                        container_name(container)
                    ),
                    0,
                    0,
                )
            }
            K::Extract => {
                // v1 trained mp3 only: aac and wav are the frozen combo pool
                let v = Fill {
                    file: file.clone(),
                    fmt: Some("mp3"),
                    ..Default::default()
                };
                (
                    Edit::ExtractAudio {
                        format: mc::AudioFormat::Mp3,
                    },
                    v,
                    "Extract(mp3)".into(),
                    0,
                    0,
                )
            }
            K::Mute => (
                Edit::Mute,
                Fill {
                    file: file.clone(),
                    ..Default::default()
                },
                "Mute".into(),
                0,
                0,
            ),
            K::Concat => {
                let n = if self.rng.below(3) == 0 { 3 } else { 2 };
                let v = Fill {
                    file: file.clone(),
                    n: Some(n),
                    ..Default::default()
                };
                (
                    Edit::Concat { n_inputs: n },
                    v,
                    format!("Concat({n})"),
                    0,
                    0,
                )
            }
        };
        if !applies(t.applies, start, end, &c, &fill) {
            return None;
        }
        // Every label is checked by the verified validator before it is written.
        if !mc::validate(edit, c) {
            return None;
        }
        let prompt = render(t.text, t.num, &fill);
        let assistant = dumps(&edit_json(edit, &file));
        let m = meta("tool", tool, Some(fx.name), t.id, intent_text, self.seed);
        self.accept(Row {
            prompt,
            assistant,
            meta: m,
        })
    }

    /// Vocabulary suggestion for a tool kind, with fixed example values.
    fn example(&self, kind: Kind, file: &str) -> String {
        let (k, v) = match kind {
            Kind::Trim => (
                K::KeepRange,
                Fill {
                    a: Some(5000),
                    b: Some(12000),
                    ..Default::default()
                },
            ),
            Kind::CutOut => (
                K::RemoveRange,
                Fill {
                    a: Some(5000),
                    b: Some(12000),
                    ..Default::default()
                },
            ),
            Kind::Scale => (
                K::ScaleExact,
                Fill {
                    w: Some(960),
                    h: Some(540),
                    ..Default::default()
                },
            ),
            Kind::Speed => (
                K::SpeedFactor,
                Fill {
                    f_pm: Some(2000),
                    ..Default::default()
                },
            ),
            Kind::Convert => (
                K::Convert,
                Fill {
                    codec: Some("libx264"),
                    container: Some("mkv"),
                    ..Default::default()
                },
            ),
            Kind::ExtractAudio => (
                K::Extract,
                Fill {
                    fmt: Some("mp3"),
                    ..Default::default()
                },
            ),
            Kind::Concat => (
                K::Concat,
                Fill {
                    n: Some(2),
                    ..Default::default()
                },
            ),
            Kind::Mute => (K::Mute, Fill::default()),
        };
        let t = vocab(k).expect("every tool kind has a vocabulary phrasing");
        render(
            t.text,
            t.num,
            &Fill {
                file: file.to_string(),
                ..v
            },
        )
    }

    fn clarify_missing_row(&mut self) -> Option<Row> {
        let req: &Request = self.rng.pick(REQUESTS);
        // Suggestions use the file the user named; if the request names none,
        // a fixed example name, so the target depends only on the text.
        let file = if req.text.contains("{file}") {
            (*self.rng.pick(FILENAMES)).to_string()
        } else {
            "clip.mp4".to_string()
        };
        let missing = clarify::missing_names(req.kinds[0], req.provided);
        let suggestions: Vec<String> = req.kinds.iter().map(|k| self.example(*k, &file)).collect();
        let question = clarify::missing_question(req, &file, &suggestions);
        let prompt = req.text.replace("{file}", &file);
        let assistant = dumps(&clarify_json(&question, &missing));
        let m = meta(
            "clarify",
            "clarify",
            None,
            req.id,
            format!("{:?}", req.kinds),
            self.seed,
        );
        self.accept(Row {
            prompt,
            assistant,
            meta: m,
        })
    }

    fn clarify_reading_row(&mut self) -> Option<Row> {
        let amb: &Ambiguous = self.rng.pick(AMBIGUOUS);
        let fx = self.fixture(false);
        let c = fx.clip;
        let file = self.file();
        let (prompt, question, intent_text) = match amb.other {
            Other::Unsupported(what) => {
                let supported = self.example(Kind::Mute, &file);
                let prompt = render(
                    amb.text,
                    amb.num,
                    &Fill {
                        file: file.clone(),
                        ..Default::default()
                    },
                );
                (
                    prompt,
                    clarify::unsupported_question(&supported, what),
                    format!("{:?}|unsupported", amb.reading),
                )
            }
            Other::Flip => {
                let (ms, fill) = match (amb.fixed_pct, amb.num) {
                    (Some(p), _) => (
                        mc::pct_of(c.duration_ms, p)?,
                        Fill {
                            file: file.clone(),
                            ..Default::default()
                        },
                    ),
                    (None, Num::Pct) => {
                        let p = *self.rng.pick(TIME_PCT);
                        (
                            mc::pct_of(c.duration_ms, p)?,
                            Fill {
                                file: file.clone(),
                                pct: Some(p),
                                ..Default::default()
                            },
                        )
                    }
                    _ => {
                        let ms = 500 * (4 + self.rng.below((c.duration_ms - 2500) / 500 - 3));
                        (
                            ms,
                            Fill {
                                file: file.clone(),
                                b: Some(ms),
                                a: Some(ms),
                                ..Default::default()
                            },
                        )
                    }
                };
                let first = make_intent(amb.reading, ms, ms);
                let second = mc::flip(first);
                // both readings must be real, accepted edits, or asking is pointless
                mc::try_label(first, c)?;
                mc::try_label(second, c)?;
                // Suggest both readings in the user's own terms: a percentage
                // ("trim the first 25%", "the first half") stays a percentage;
                // seconds stay seconds. Never seconds derived from the hidden
                // clip duration, which the model cannot see.
                let pct = fill.pct.or(amb.fixed_pct);
                let say = |i: Intent| match pct {
                    Some(p) => {
                        let t = vocab_in(intent_kind(i), Num::Pct).expect("percentage vocabulary");
                        render(
                            t.text,
                            Num::Pct,
                            &Fill {
                                file: file.clone(),
                                pct: Some(p),
                                ..Default::default()
                            },
                        )
                    }
                    None => {
                        let t = vocab_in(intent_kind(i), Num::Secs).expect("seconds vocabulary");
                        render(t.text, Num::Secs, &time_fill(i, &file))
                    }
                };
                let prompt = render(amb.text, amb.num, &fill);
                (
                    prompt,
                    clarify::reading_question(&say(first), &say(second)),
                    format!("{first:?}|{second:?}"),
                )
            }
        };
        // reading-ambiguous: both readings are fully specified, so the proved
        // missing list is empty
        let assistant = dumps(&clarify_json(&question, &[]));
        let m = meta(
            "clarify",
            "clarify",
            Some(fx.name),
            amb.id,
            intent_text,
            self.seed,
        );
        self.accept(Row {
            prompt,
            assistant,
            meta: m,
        })
    }

    /// Generate all rows in a fixed order, then shuffle with the same RNG.
    pub fn run(&mut self, counts: &Counts) -> Vec<Row> {
        let mut rows = Vec::new();
        let mut take = |g: &mut Generator, n: usize, f: &dyn Fn(&mut Generator) -> Option<Row>| {
            let mut made = 0;
            let mut tries = 0;
            while made < n {
                tries += 1;
                assert!(tries < 200 * n + 1000, "cannot fill {n} rows");
                match f(g) {
                    Some(r) => {
                        rows.push(r);
                        made += 1;
                    }
                    None => g.resampled += 1,
                }
            }
        };
        for (tool, n) in &counts.per_tool {
            let tool: &'static str = tool;
            take(self, *n, &move |g: &mut Generator| g.tool_row(tool));
        }
        take(self, counts.clarify_missing, &|g: &mut Generator| {
            g.clarify_missing_row()
        });
        take(self, counts.clarify_reading, &|g: &mut Generator| {
            g.clarify_reading_row()
        });
        let mut order: Vec<usize> = (0..rows.len()).collect();
        self.rng.shuffle(&mut order);
        let mut slots: Vec<Option<Row>> = rows.into_iter().map(Some).collect();
        order
            .into_iter()
            .map(|i| slots[i].take().unwrap())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dumps_matches_python_json() {
        // python3 -c 'import json; print(json.dumps({"tool": "trim", "start": 12.0, "end": 7.5, "q": "a\"b—c"}))'
        let j = J::O(vec![
            ("tool", J::S("trim".into())),
            ("start", J::F(12.0)),
            ("end", J::F(7.5)),
            ("q", J::S("a\"b—c".into())),
        ]);
        assert_eq!(
            dumps(&j),
            r#"{"tool": "trim", "start": 12.0, "end": 7.5, "q": "a\"b\u2014c"}"#
        );
        assert_eq!(dumps(&J::F(0.05)), "0.05");
        assert_eq!(dumps(&J::F(3.0)), "3.0");
    }

    #[test]
    fn same_seed_same_bytes() {
        let c = Counts {
            per_tool: vec![("trim", 40), ("cut_out", 40), ("scale", 20), ("speed", 20)],
            clarify_missing: 20,
            clarify_reading: 10,
        };
        let a: Vec<String> = Generator::new(7).run(&c).iter().map(|r| r.line()).collect();
        let b: Vec<String> = Generator::new(7).run(&c).iter().map(|r| r.line()).collect();
        assert_eq!(a, b);
        let d: Vec<String> = Generator::new(8).run(&c).iter().map(|r| r.line()).collect();
        assert_ne!(a, d);
    }

    #[test]
    fn every_tool_target_passes_the_wire_validator_and_keeps_grammar_order() {
        let mut g = Generator::new(1);
        let rows = g.run(&Counts::pilot_matched());
        assert_eq!(rows.len(), 2319);
        for r in &rows {
            let v: serde_json::Value = serde_json::from_str(&r.assistant).unwrap();
            let tool = v["tool"].as_str().unwrap();
            // grammar field order: TOOLS table order, "tool" first
            let fields = crate::grammar::TOOLS
                .iter()
                .find(|(t, _)| *t == tool)
                .unwrap()
                .1;
            let order: Vec<&str> = fields.iter().map(|f| f.name).collect();
            let mut last = 0;
            for f in r.assistant.match_indices("\": ").map(|(i, _)| {
                let start = r.assistant[..i].rfind('"').unwrap() + 1;
                &r.assistant[start..i]
            }) {
                if f == "tool" {
                    continue;
                }
                if let Some(pos) = order.iter().position(|o| *o == f) {
                    assert!(
                        pos >= last,
                        "{tool}: field {f} out of grammar order in {}",
                        r.assistant
                    );
                    last = pos;
                }
            }
            if tool != "clarify" {
                let fx = r.meta_fixture();
                let probe = crate::wire::WireProbe {
                    duration_s: fx.clip.duration_ms as f64 / 1000.0,
                    width: fx.clip.width,
                    height: fx.clip.height,
                    has_audio: fx.clip.has_audio,
                };
                crate::wire::validate(&r.assistant, &probe)
                    .unwrap_or_else(|e| panic!("{e}: {} -> {}", r.prompt, r.assistant));
            }
        }
    }

    impl Row {
        fn meta_fixture(&self) -> &'static Fixture {
            let m = dumps(&self.meta);
            FIXTURES
                .iter()
                .find(|f| m.contains(&format!("\"fixture\": \"{}\"", f.name)))
                .unwrap()
        }
    }

    #[test]
    fn labels_never_depend_on_hidden_clip_facts() {
        // The model sees only the request text (cli.compose_prompt), never the
        // probe. If a label needed the clip's duration or size, the same text
        // would get different labels on different fixtures. Across several
        // full datasets, one prompt must always mean one target.
        let mut by_prompt: std::collections::BTreeMap<String, String> = Default::default();
        for seed in [1, 2, 3] {
            for r in Generator::new(seed).run(&Counts::pilot_matched()) {
                if let Some(prev) = by_prompt.insert(r.prompt.clone(), r.assistant.clone()) {
                    assert_eq!(
                        prev, r.assistant,
                        "same text, different label: {:?}",
                        r.prompt
                    );
                }
            }
        }
    }

    #[test]
    fn ambiguous_rows_offer_both_readings_and_missing_is_empty() {
        let mut g = Generator::new(3);
        for _ in 0..200 {
            if let Some(r) = g.clarify_reading_row() {
                let v: serde_json::Value = serde_json::from_str(&r.assistant).unwrap();
                assert_eq!(v["missing"], serde_json::json!([]), "{}", r.assistant);
                let q = v["question"].as_str().unwrap();
                assert!(q.starts_with("Do you"), "{q}");
            }
        }
    }
}
