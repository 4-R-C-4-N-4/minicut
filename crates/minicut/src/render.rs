//! Lossless, trailing-zero-free rendering of millisecond durations for UI text.

fn frac_text(f: u64) -> String {
    // f is < 1000 (a 3-digit fraction of a second in ms). Zero-pad to 3,
    // then trim trailing zeros and a trailing dot.
    if f == 0 {
        return String::new();
    }
    let mut s = format!("{f:03}");
    while s.ends_with('0') {
        s.pop();
    }
    format!(".{s}")
}

pub fn secs_text(ms: u64) -> String {
    let (s, f) = minicut_core::to_secs(ms);
    format!("{s}{}", frac_text(f))
}

pub fn clock_text(ms: u64) -> String {
    let (m, s, f) = minicut_core::to_clock(ms);
    format!("{m}:{s:02}{}", frac_text(f))
}

pub fn pct_text(p: u64) -> String {
    format!("{p}%")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn secs_text_examples() {
        assert_eq!(secs_text(12000), "12");
        assert_eq!(secs_text(12500), "12.5");
        assert_eq!(secs_text(12050), "12.05");
        assert_eq!(secs_text(12005), "12.005");
        assert_eq!(secs_text(0), "0");
    }

    #[test]
    fn clock_text_examples() {
        assert_eq!(clock_text(72000), "1:12");
        assert_eq!(clock_text(72500), "1:12.5");
        assert_eq!(clock_text(5000), "0:05");
        assert_eq!(clock_text(600000), "10:00");
    }

    #[test]
    fn pct_text_examples() {
        assert_eq!(pct_text(0), "0%");
        assert_eq!(pct_text(50), "50%");
        assert_eq!(pct_text(100), "100%");
    }

    /// Hand-written parsers mirroring the exact grammar secs_text/clock_text
    /// produce, used to round-trip every ms value and prove losslessness.
    fn parse_fraction(frac: &str) -> u64 {
        // frac has no leading dot, 0-3 digits; pad right to 3 with zeros.
        if frac.is_empty() {
            return 0;
        }
        let mut digits: Vec<char> = frac.chars().collect();
        while digits.len() < 3 {
            digits.push('0');
        }
        let s: String = digits.into_iter().collect();
        s.parse().unwrap()
    }

    fn parse_secs_text(s: &str) -> u64 {
        let (whole, frac) = match s.split_once('.') {
            Some((w, f)) => (w, f),
            None => (s, ""),
        };
        let whole: u64 = whole.parse().unwrap();
        whole * 1000 + parse_fraction(frac)
    }

    fn parse_clock_text(s: &str) -> u64 {
        let (mins, rest) = s.split_once(':').unwrap();
        let mins: u64 = mins.parse().unwrap();
        let (secs, frac) = match rest.split_once('.') {
            Some((w, f)) => (w, f),
            None => (rest, ""),
        };
        let secs: u64 = secs.parse().unwrap();
        mins * 60000 + secs * 1000 + parse_fraction(frac)
    }

    #[test]
    fn exhaustive_round_trip_over_one_hour() {
        for ms in 0..=3_600_000u64 {
            let st = secs_text(ms);
            assert_eq!(parse_secs_text(&st), ms, "secs_text round trip at {ms}");
            let ct = clock_text(ms);
            assert_eq!(parse_clock_text(&ct), ms, "clock_text round trip at {ms}");
        }
    }
}
