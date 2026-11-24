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

// Speed: the kept duration is duration / factor, rounded down, stated
// multiplicatively, and it moves the right way: faster is shorter, slower is
// never shorter. (Pins the Speed arm of kept_ms; without this nothing did.)
pub proof fn speed_meaning(f: u64, c: Clip)
    requires well_formed(Edit::Speed { factor_pm: f }, c)
    ensures
        kept_ms(Edit::Speed { factor_pm: f }, c) * f <= c.duration_ms * 1000,
        c.duration_ms * 1000 < (kept_ms(Edit::Speed { factor_pm: f }, c) + 1) * f,
        f > 1000 && c.duration_ms > 0 ==> kept_ms(Edit::Speed { factor_pm: f }, c) < c.duration_ms,
        f < 1000 ==> kept_ms(Edit::Speed { factor_pm: f }, c) >= c.duration_ms,
{
    let k = kept_ms(Edit::Speed { factor_pm: f }, c);
    let d = c.duration_ms as int;
    let fi = f as int;
    assert(k == d * 1000 / fi);
    assert(k * fi <= d * 1000 && d * 1000 < (k + 1) * fi) by (nonlinear_arith)
        requires k == d * 1000 / fi, fi > 0;
    if f > 1000 && d > 0 {
        assert(k < d) by (nonlinear_arith)
            requires k * fi <= d * 1000, fi > 1000, d > 0, k >= 0;
    }
    if f < 1000 {
        assert(k >= d) by (nonlinear_arith)
            requires d * 1000 < (k + 1) * fi, fi < 1000, fi > 0, d >= 0;
    }
}

// scale, convert, extract_audio and mute keep the full duration
// (verify.py checks each of these within tolerance).
pub proof fn duration_preserving(e: Edit, c: Clip)
    requires
        e is Scale || e is Convert || e is ExtractAudio || e is Mute,
    ensures
        kept_ms(e, c) == c.duration_ms,
{
}

// Concat's meaning is the sum of all input durations, which needs every
// probed clip, not just the first. It is not modelled here: Phase 2 adds it
// with a seq<Clip>. kept_ms(Concat) is a placeholder and no lemma uses it.

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

/// Does source time `t` (ms) survive the edit? Trim keeps [start, end);
/// cut_out keeps everything outside it. Only trim/cut_out select time.
pub open spec fn survives(e: Edit, t: int) -> bool {
    match e {
        Edit::Trim { start_ms, end_ms } => start_ms <= t && t < end_ms,
        Edit::CutOut { start_ms, end_ms } => t < start_ms || end_ms <= t,
        _ => true,
    }
}

/// Which source times the request keeps, stated in the request's own terms
/// (no tool, no start/end fields): "remove the last 5 s" keeps t < dur - 5 s.
pub open spec fn intent_survives(i: Intent, c: Clip, t: int) -> bool {
    match i {
        Intent::KeepRange { start_ms, end_ms } => start_ms <= t && t < end_ms,
        Intent::RemoveRange { start_ms, end_ms } => !(start_ms <= t && t < end_ms),
        Intent::KeepFirst { ms } => t < ms,
        Intent::RemoveFirst { ms } => t >= ms,
        Intent::KeepLast { ms } => t >= c.duration_ms - ms,
        Intent::RemoveLast { ms } => t < c.duration_ms - ms,
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

pub(crate) fn label(i: Intent, c: Clip) -> (e: Edit)
    requires
        match i {
            Intent::KeepLast { ms } => ms <= c.duration_ms,
            Intent::RemoveLast { ms } => ms <= c.duration_ms,
            _ => true,
        },
    ensures
        e == label_spec(i, c),
        kept_ms(e, c) == intent_kept(i, c),
        // where: every source instant inside the clip survives exactly when
        // the request says it should
        forall|t: int| 0 <= t < c.duration_ms ==> #[trigger] survives(e, t) == intent_survives(i, c, t),
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

// --------------------------------------------- clock rendering (mm:ss.fff) --
pub fn to_clock(ms: u64) -> (r: (u64, u64, u64))
    ensures r.0 * 60000 + r.1 * 1000 + r.2 == ms, r.1 < 60, r.2 < 1000,
{
    let rem = ms % 60000;
    (ms / 60000, rem / 1000, rem % 1000)
}

// ------------------------------------------------------- atempo chain ------
// ffmpeg atempo takes 0.5..2.0 per stage here; 0.25..4.0 needs <= 2 stages.
pub open spec fn stage_ok(s: u64) -> bool { 500 <= s <= 2000 }

pub(crate) fn atempo_stages(f: u64) -> (r: (u64, Option<u64>))
    requires 250 <= f <= 4000
    ensures
        stage_ok(r.0),
        match r.1 {
            None => r.0 == f,
            // product of stages, in per-mille^2, is within 0.001x of f
            Some(s2) => stage_ok(s2) && r.0 * s2 <= f * 1000 && f * 1000 <= r.0 * s2 + 1000,
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

/// Total entry point for the app: None outside the validator's 0.25x..4x.
pub fn try_atempo(f: u64) -> (r: Option<(u64, Option<u64>)>)
    ensures
        match r {
            Some(st) => 250 <= f <= 4000 && stage_ok(st.0) && match st.1 {
                None => st.0 == f,
                Some(s2) => stage_ok(s2) && st.0 * s2 <= f * 1000 && f * 1000 <= st.0 * s2 + 1000,
            },
            None => !(250 <= f <= 4000),
        },
{
    if 250 <= f && f <= 4000 { Some(atempo_stages(f)) } else { None }
}

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

// Seconds text such as "12.5": whole seconds + millis, lossless. Printing
// the digits happens outside the core; a round-trip test guards it.
pub fn to_secs(ms: u64) -> (r: (u64, u64))
    ensures r.0 * 1000 + r.1 == ms, r.1 < 1000,
{
    (ms / 1000, ms % 1000)
}

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

// ------------------------------------------- ambiguity: the other reading ----
// "trim the first 25%" can mean keep-first or remove-first. `flip` gives the
// other reading of the same span; `readings_disagree` proves the two readings
// disagree at *every* instant of *every* clip, so a guess is entirely wrong
// under one of them. That is why ambiguous wording becomes a clarify that
// offers both readings (as canonical phrasings), never a guessed label.
pub open spec fn flip_spec(i: Intent) -> Intent {
    match i {
        Intent::KeepRange { start_ms, end_ms } => Intent::RemoveRange { start_ms, end_ms },
        Intent::RemoveRange { start_ms, end_ms } => Intent::KeepRange { start_ms, end_ms },
        Intent::KeepFirst { ms } => Intent::RemoveFirst { ms },
        Intent::RemoveFirst { ms } => Intent::KeepFirst { ms },
        Intent::KeepLast { ms } => Intent::RemoveLast { ms },
        Intent::RemoveLast { ms } => Intent::KeepLast { ms },
    }
}

pub fn flip(i: Intent) -> (o: Intent)
    ensures
        o == flip_spec(i),
        is_keep(i) != is_keep(o),
{
    match i {
        Intent::KeepRange { start_ms, end_ms } => Intent::RemoveRange { start_ms, end_ms },
        Intent::RemoveRange { start_ms, end_ms } => Intent::KeepRange { start_ms, end_ms },
        Intent::KeepFirst { ms } => Intent::RemoveFirst { ms },
        Intent::RemoveFirst { ms } => Intent::KeepFirst { ms },
        Intent::KeepLast { ms } => Intent::RemoveLast { ms },
        Intent::RemoveLast { ms } => Intent::KeepLast { ms },
    }
}

pub proof fn readings_disagree(i: Intent, c: Clip)
    ensures
        forall|t: int| #[trigger] intent_survives(i, c, t) == !intent_survives(flip_spec(i), c, t),
{
}

} // verus!

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_on_stock_rust() {
        let c = Clip {
            duration_ms: 30000,
            width: Some(1280),
            height: Some(720),
            has_audio: true,
        };
        assert!(validate(label(Intent::RemoveFirst { ms: 10000 }, c), c));
        assert!(!validate(Edit::Speed { factor_pm: 1000 }, c));
        assert!(!validate(
            Edit::Trim {
                start_ms: 0,
                end_ms: 30000
            },
            c
        ));
        assert!(!validate(
            Edit::Scale {
                width: 1280,
                height: 720
            },
            c
        ));
        assert_eq!(atempo_stages(4000), (2000, Some(2000)));
        assert_eq!(atempo_stages(250), (500, Some(500)));
        assert_eq!(atempo_stages(1250), (1250, None));
        assert_eq!(to_clock(72500), (1, 12, 500));
        assert_eq!(missing(Kind::Trim, 0b1), 0b110);
        assert_eq!(missing(Kind::Convert, 0b1), 0b0110000000);
    }
}
