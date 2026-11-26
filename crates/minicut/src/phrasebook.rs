//! The phrasebook: the request vocabulary MiniCut is taught and documents
//! (docs/plans/2026-11-25-phrasebook.md). One table drives three things:
//! the keep/remove classifier below, the tests that every template and every
//! frozen eval prompt follow it, and `minicut --help` / docs/phrasebook.md.
//!
//! Keep vs remove, the only wording rule that decides between two tools:
//!   1. a phrase that names the RESULT keeps it (trim);
//!   2. a phrase that names a PART removes it (cut_out);
//!   3. no part and no times: ask for them (clarify, missing start/end);
//!   4. something MiniCut cannot do: ask, offering what it can (UNSUPPORTED).
//!
//! The classifier is a documentation and test oracle over our own wording,
//! not a parser of user input: the model reads requests.

use crate::templates::{vocab, UNSUPPORTED};

/// Words that say the result is what stays (rule 1).
pub const KEEP_WORDS: &[&str] = &["keep", "only", "just", "isolate"];
/// Multi-word keep markers (rule 1).
pub const KEEP_PHRASES: &[&str] = &["give me"];
/// After "trim X" / "cut X", these say the result follows (rule 1):
/// "trim X to the first 10 seconds", "cut X down to 5-15", "trim X from 5 to 15".
pub const KEEP_AFTER_FILE: &[&str] = &["to", "down", "from"];
/// Verbs that name a part to take away (rule 2).
pub const REMOVE_WORDS: &[&str] = &["cut", "snip", "remove", "delete", "drop", "take", "without"];

/// How the phrasebook reads a keep/remove request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reading {
    /// rule 1: trim
    Keep,
    /// rule 2: cut_out
    Remove,
    /// rule 3: no times given; clarify with missing start/end
    MissingTimes,
    /// not a phrasebook wording (e.g. "trim the first 10 seconds of X"):
    /// never taught, so it is never a training label
    NotCovered,
}

fn words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|w| {
            w.trim_matches(|c: char| ",.;:!?()\"'".contains(c))
                .to_lowercase()
        })
        .filter(|w| !w.is_empty())
        .collect()
}

/// A time or number: starts with a digit (5, 7.5, 0:12, 2-8, 3–9). File
/// names such as clip2.mp4 contain digits but do not start with one.
fn is_number(w: &str) -> bool {
    w.starts_with(|c: char| c.is_ascii_digit())
}

fn is_file(w: &str) -> bool {
    w.contains('.') && !is_number(w)
}

/// Classify a keep/remove request by the phrasebook rules.
pub fn reading(text: &str) -> Reading {
    let w = words(text);
    if !w.iter().any(|x| is_number(x)) {
        return Reading::MissingTimes;
    }
    let lower = w.join(" ");
    if w.iter().any(|x| KEEP_WORDS.contains(&x.as_str()))
        || KEEP_PHRASES.iter().any(|p| lower.contains(p))
    {
        return Reading::Keep;
    }
    // "trim X to …", "cut X down to …", "cut X to …", "trim X from a to b"
    for i in 0..w.len().saturating_sub(2) {
        if (w[i] == "trim" || w[i] == "cut")
            && is_file(&w[i + 1])
            && KEEP_AFTER_FILE.contains(&w[i + 2].as_str())
        {
            return Reading::Keep;
        }
    }
    if w.iter().any(|x| REMOVE_WORDS.contains(&x.as_str())) {
        return Reading::Remove;
    }
    Reading::NotCovered
}

/// Keywords for the tools without a keep/remove choice (help text; each
/// row's wording is covered by templates of that tool).
pub const TOOL_KEYWORDS: &[(&str, &str, &str)] = &[
    (
        "join",
        "join, concatenate, merge, append, put N copies together, loop, ... in a row",
        "join clip.mp4 to itself",
    ),
    (
        "resize",
        "make X WxH, resize X to W by H, scale X up/down to WxH",
        "make clip.mp4 1280x720",
    ),
    (
        "speed",
        "run/play X at Nx speed, make X N times as fast, speed X up by N%, slow X to N% speed",
        "run clip.mp4 at 2x speed",
    ),
    (
        "convert",
        "convert / re-encode / save X ... CONTAINER ... CODEC",
        "convert clip.mp4 to mkv with libx264",
    ),
    (
        "audio",
        "extract / save / pull / get the audio (soundtrack, mp3) from X as FORMAT",
        "extract the audio from clip.mp4 as mp3",
    ),
    (
        "mute",
        "mute, silence, remove/drop/strip the audio from X, make X silent",
        "mute clip.mp4",
    ),
];

/// Plain-text phrasebook for `minicut --help` (and the Python CLI epilog).
pub fn help_text() -> String {
    let mut s = String::from(
        "How to ask (the MiniCut phrasebook)\n\n\
         Keep or remove part of a clip: say what stays, or say what goes.\n\
         \x20 KEEP   (the result)  keep only 2 to 8 seconds of clip.mp4\n\
         \x20                      trim clip.mp4 to the first 10 seconds\n\
         \x20                      cut clip.mp4 down to 5-15 seconds\n\
         \x20                      give me just the first 5 seconds of clip.mp4\n\
         \x20 REMOVE (a part)      cut the first 10 seconds off clip.mp4\n\
         \x20                      remove 2 to 8 seconds from clip.mp4\n\
         \x20                      drop seconds 2 through 8 of clip.mp4\n\n",
    );
    s += &format!(
        "  Keep words: {}; or \"to\" / \"down to\" / \"from\" right after \"trim X\" or \"cut X\".\n",
        [KEEP_WORDS, KEEP_PHRASES].concat().join(", ")
    );
    s += &format!(
        "  Remove words: {} (\"cut ... of\" and \"cut ... off\" mean the same).\n",
        REMOVE_WORDS.join(", ")
    );
    s += "  Without times (\"trim clip.mp4\"), MiniCut asks for them.\n\
          \x20 Give times in seconds (7.5) or as m:ss (1:05). Percentages and \"the last N\n\
          \x20 seconds\" are not supported yet: give the start and end.\n\nOther edits:\n";
    for (name, words, example) in TOOL_KEYWORDS {
        s += &format!("  {name:<8} {words}\n  {:<8} e.g. \"{example}\"\n", "");
    }
    s += "\nNot supported (MiniCut asks and offers the nearest edit):\n";
    for u in UNSUPPORTED {
        let offer = vocab(u.offer).map_or("", |t| t.text);
        s += &format!(
            "  \"{}\": cannot {}; offers \"{}\"\n",
            u.text.replace("{file}", "X"),
            u.what,
            offer.replace("{file}", "X")
        );
    }
    s
}

/// docs/phrasebook.md (a test keeps the committed copy in sync).
pub fn markdown() -> String {
    format!(
        "# MiniCut phrasebook (generated)\n\n\
         Generated by `minicut phrasebook --markdown` from `crates/minicut/src/phrasebook.rs`; \
         `minicut --help` prints the same text. The rule and its reasons: \
         `docs/plans/2026-11-25-phrasebook.md`.\n\n```text\n{}```\n",
        help_text()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clarify::REQUESTS;
    use crate::templates::{IntentKind as K, EXCLUDED, TEMPLATES};
    use minicut_core::Kind;

    fn fill(text: &str) -> String {
        text.replace("{file}", "clip.mp4")
            .replace("{a}", "2")
            .replace("{b}", "8")
            .replace("{pct}", "25%")
    }

    #[test]
    fn every_time_template_follows_the_rule() {
        let mut n = 0;
        for t in TEMPLATES {
            let want = match t.kind {
                K::KeepRange | K::KeepFirst | K::KeepLast => Reading::Keep,
                K::RemoveRange | K::RemoveFirst | K::RemoveLast => Reading::Remove,
                _ => continue,
            };
            assert_eq!(reading(&fill(t.text)), want, "{}: {}", t.id, t.text);
            n += 1;
        }
        assert!(n >= 25, "{n}");
    }

    #[test]
    fn time_requests_without_times_ask_for_them() {
        for r in REQUESTS {
            if r.kinds
                .iter()
                .any(|k| matches!(k, Kind::Trim | Kind::CutOut))
            {
                assert_eq!(reading(&fill(r.text)), Reading::MissingTimes, "{}", r.id);
            }
        }
    }

    #[test]
    fn untaught_trim_the_first_wording_is_not_covered() {
        // phrasebook Q1: "trim the first/last ... of X" is not taught
        for (id, text, _) in EXCLUDED
            .iter()
            .filter(|(id, _, _)| id.starts_with("v3_amb_trim"))
        {
            let s = fill(text).replace("half", "10 seconds");
            assert_eq!(reading(&s), Reading::NotCovered, "{id}: {s}");
        }
    }

    #[test]
    fn of_and_off_read_the_same() {
        assert_eq!(
            reading("cut the first 10 seconds of clip.mp4"),
            Reading::Remove
        );
        assert_eq!(
            reading("cut the first 10 seconds off clip.mp4"),
            Reading::Remove
        );
    }

    #[test]
    fn the_frozen_eval_already_speaks_the_phrasebook() {
        let files = [
            include_str!("../../../eval/eval_seen.jsonl"),
            include_str!("../../../eval/eval_unseen_combo.jsonl"),
            include_str!("../../../eval/eval_unseen_fixture.jsonl"),
            include_str!("../../../eval/eval_unseen_paraphrase.jsonl"),
        ];
        let mut n = 0;
        for line in files.iter().flat_map(|f| f.lines()) {
            let v: serde_json::Value = serde_json::from_str(line).unwrap();
            let prompt = v["prompt"].as_str().unwrap();
            let want = match v["expected"]["tool"].as_str().unwrap() {
                "trim" => Reading::Keep,
                "cut_out" => Reading::Remove,
                "clarify" if prompt.contains("trim") || prompt.contains("cut") => {
                    Reading::MissingTimes
                }
                _ => continue,
            };
            assert_eq!(reading(prompt), want, "{}: {prompt}", v["id"]);
            n += 1;
        }
        assert_eq!(n, 68, "eval keep/remove prompts checked");
    }

    #[test]
    fn help_lists_every_keyword_and_unsupported_request() {
        let h = help_text();
        for w in KEEP_WORDS.iter().chain(REMOVE_WORDS) {
            assert!(h.contains(w), "{w}");
        }
        for u in UNSUPPORTED {
            assert!(h.contains(&u.text.replace("{file}", "X")), "{}", u.id);
        }
        // every help example follows its own rule
        assert_eq!(
            reading("give me just the first 5 seconds of clip.mp4"),
            Reading::Keep
        );
        assert_eq!(
            reading("drop seconds 2 through 8 of clip.mp4"),
            Reading::Remove
        );
    }
}
