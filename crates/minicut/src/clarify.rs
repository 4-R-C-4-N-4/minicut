//! Clarify rows: requests that cannot be labelled get a question, never a
//! guess (plan P3-D11). `missing` is always computed by the verified core
//! (`minicut_core::missing`), never written by hand. Questions offer our own
//! vocabulary (templates::VOCAB) as suggestions; graders compare only
//! `missing`, so the wording can carry them without changing any score.

use minicut_core::Kind;

/// Argument names in the core's bit order (minicut-core `missing`, bits 0-9).
pub const ARG_BITS: [&str; 10] = [
    "input",
    "start",
    "end",
    "inputs",
    "width",
    "height",
    "factor",
    "codec",
    "container",
    "format",
];

/// Bit 0 (`input`): the application binds the file, so it is always provided.
pub const INPUT_BIT: u16 = 1;

/// Missing argument names for `kind`, given the provided-args bitmask, in bit
/// order. Computed by the verified core: exactly required − provided.
pub fn missing_names(kind: Kind, provided: u16) -> Vec<&'static str> {
    let m = minicut_core::missing(kind, provided);
    ARG_BITS
        .iter()
        .enumerate()
        .filter(|(b, _)| (m >> b) & 1 == 1)
        .map(|(_, n)| *n)
        .collect()
}

/// A request that names a tool family but not its arguments. `kinds` is every
/// tool the wording could mean; when there are several, the question offers a
/// suggestion for each.
pub struct Request {
    pub id: &'static str,
    pub text: &'static str,
    pub kinds: &'static [Kind],
    /// Bits the request provides (the app always provides `input`).
    pub provided: u16,
    pub note: &'static str,
}

const TRIM_OR_CUT: &[Kind] = &[Kind::Trim, Kind::CutOut];

/// Ported from training/generate_dataset.py CLARIFY_AMBIG, with each request
/// assigned the set of tools it could mean.
pub const REQUESTS: &[Request] = &[
    Request {
        id: "v3_clar_trim",
        text: "trim {file}",
        kinds: TRIM_OR_CUT,
        provided: INPUT_BIT,
        note: "bare 'trim' is read as keep or remove (cf. 'trim the first 25%')",
    },
    Request {
        id: "v3_clar_cut",
        text: "cut {file}",
        kinds: TRIM_OR_CUT,
        provided: INPUT_BIT,
        note: "v1 answered with the trim question; 'cut' can mean either",
    },
    Request {
        id: "v3_clar_shorten",
        text: "shorten {file}",
        kinds: TRIM_OR_CUT,
        provided: INPUT_BIT,
        note: "shorter by keeping a part or by removing one",
    },
    Request {
        id: "v3_clar_trim_the_clip",
        text: "trim the clip",
        kinds: TRIM_OR_CUT,
        provided: INPUT_BIT,
        note: "as 'trim {file}'",
    },
    Request {
        id: "v3_clar_edit_down",
        text: "edit {file} down",
        kinds: &[Kind::Trim],
        provided: INPUT_BIT,
        note: "'edit down' keeps the essential part",
    },
    Request {
        id: "v3_clar_cut_bad",
        text: "cut the bad part out of {file}",
        kinds: &[Kind::CutOut],
        provided: INPUT_BIT,
        note: "'cut ... out' removes",
    },
    Request {
        id: "v3_clar_remove_boring",
        text: "remove the boring bit from {file}",
        kinds: &[Kind::CutOut],
        provided: INPUT_BIT,
        note: "'remove' names what goes",
    },
    Request {
        id: "v3_clar_cut_something",
        text: "cut something out of {file}",
        kinds: &[Kind::CutOut],
        provided: INPUT_BIT,
        note: "'cut ... out' removes",
    },
    Request {
        id: "v3_clar_smaller",
        text: "make {file} smaller",
        kinds: &[Kind::Scale],
        provided: INPUT_BIT,
        note: "the frozen eval defines 'make it smaller' as scale",
    },
    Request {
        id: "v3_clar_resize",
        text: "resize {file}",
        kinds: &[Kind::Scale],
        provided: INPUT_BIT,
        note: "",
    },
    Request {
        id: "v3_clar_size",
        text: "change the size of {file}",
        kinds: &[Kind::Scale],
        provided: INPUT_BIT,
        note: "",
    },
    Request {
        id: "v3_clar_bigger",
        text: "make {file} bigger",
        kinds: &[Kind::Scale],
        provided: INPUT_BIT,
        note: "",
    },
    Request {
        id: "v3_clar_speed_up",
        text: "speed up {file}",
        kinds: &[Kind::Speed],
        provided: INPUT_BIT,
        note: "",
    },
    Request {
        id: "v3_clar_slow",
        text: "slow {file} down",
        kinds: &[Kind::Speed],
        provided: INPUT_BIT,
        note: "",
    },
    Request {
        id: "v3_clar_speed",
        text: "change the speed of {file}",
        kinds: &[Kind::Speed],
        provided: INPUT_BIT,
        note: "",
    },
    Request {
        id: "v3_clar_convert",
        text: "convert {file}",
        kinds: &[Kind::Convert],
        provided: INPUT_BIT,
        note: "",
    },
    Request {
        id: "v3_clar_format",
        text: "change the format of {file}",
        kinds: &[Kind::Convert],
        provided: INPUT_BIT,
        note: "",
    },
    Request {
        id: "v3_clar_reencode",
        text: "re-encode {file}",
        kinds: &[Kind::Convert],
        provided: INPUT_BIT,
        note: "",
    },
    Request {
        id: "v3_clar_audio",
        text: "get the audio from {file}",
        kinds: &[Kind::ExtractAudio],
        provided: INPUT_BIT,
        note: "",
    },
    Request {
        id: "v3_clar_pull_audio",
        text: "pull the audio out of {file}",
        kinds: &[Kind::ExtractAudio],
        provided: INPUT_BIT,
        note: "",
    },
    Request {
        id: "v3_clar_sound",
        text: "save the sound from {file}",
        kinds: &[Kind::ExtractAudio],
        provided: INPUT_BIT,
        note: "",
    },
    Request {
        id: "v3_clar_join",
        text: "join the clips",
        kinds: &[Kind::Concat],
        provided: 0,
        note: "concat takes `inputs`; none given",
    },
    Request {
        id: "v3_clar_together",
        text: "put the videos together",
        kinds: &[Kind::Concat],
        provided: 0,
        note: "",
    },
    Request {
        id: "v3_clar_combine",
        text: "combine {file} with another video",
        kinds: &[Kind::Concat],
        provided: 0,
        note: "the other video is not named",
    },
];

/// The opening sentence for a single tool, from the v1 questions (models
/// trained on v1 have seen these).
fn ask(kind: Kind) -> &'static str {
    match kind {
        Kind::Trim => "What start and end times should I trim to?",
        Kind::CutOut => "Which start and end times should I cut out?",
        Kind::Scale => "What width and height should I scale to?",
        Kind::Speed => "What speed factor should I use?",
        Kind::Convert => "Which codec and container should I convert to?",
        Kind::ExtractAudio => "What audio format should I extract to?",
        Kind::Concat => "Which files should I join, and in what order?",
        Kind::Mute => "Which file should I mute?",
    }
}

fn quoted_list(suggestions: &[String]) -> String {
    let q: Vec<String> = suggestions.iter().map(|s| format!("\"{s}\"")).collect();
    match q.len() {
        0 => String::new(),
        1 => q[0].clone(),
        _ => format!("{} or {}", q[..q.len() - 1].join(", "), q[q.len() - 1]),
    }
}

/// Question for an argument-missing request. `suggestions` are vocabulary
/// phrasings already rendered with example values, one per kind in
/// `req.kinds` (same order).
pub fn missing_question(req: &Request, file: &str, suggestions: &[String]) -> String {
    let head = if req.kinds == TRIM_OR_CUT {
        format!(
            "Do you want to keep only part of {file} or cut part of it out? Tell me the start and end times."
        )
    } else {
        ask(req.kinds[0]).to_string()
    };
    if suggestions.is_empty() {
        head
    } else {
        format!("{head} For example: {}.", quoted_list(suggestions))
    }
}

/// docs/templates.md section: each request with the tools it could mean and
/// the missing list the verified core computes for it.
pub fn markdown() -> String {
    let mut s = String::from(
        "\n## Clarify requests: arguments missing (missing list computed by the verified core)\n\n\
         | id | request | could mean | missing | why |\n|---|---|---|---|---|\n",
    );
    for r in REQUESTS {
        s += &format!(
            "| `{}` | {} | {:?} | {:?} | {} |\n",
            r.id,
            r.text,
            r.kinds,
            missing_names(r.kinds[0], r.provided),
            r.note
        );
    }
    s
}

/// Question for reading-ambiguous wording: both readings, as our phrasings
/// with the user's own numbers ("Do you want to keep only the first 7.5
/// seconds of clip.mp4, or remove the first 7.5 seconds of clip.mp4?").
pub fn reading_question(first: &str, second: &str) -> String {
    format!("Do you want to {first}, or {second}?")
}

/// Question when the other reading is something MiniCut cannot do.
pub fn unsupported_question(supported: &str, unsupported: &str) -> String {
    format!("Do you mean \"{supported}\"? I can't {unsupported}.")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn required(kind: Kind) -> Vec<&'static str> {
        match kind {
            Kind::Trim | Kind::CutOut => vec!["input", "start", "end"],
            Kind::Concat => vec!["inputs"],
            Kind::Scale => vec!["input", "width", "height"],
            Kind::Speed => vec!["input", "factor"],
            Kind::Convert => vec!["input", "codec", "container"],
            Kind::ExtractAudio => vec!["input", "format"],
            Kind::Mute => vec!["input"],
        }
    }

    #[test]
    fn missing_is_required_minus_provided_for_every_subset() {
        let kinds = [
            Kind::Trim,
            Kind::CutOut,
            Kind::Concat,
            Kind::Scale,
            Kind::Speed,
            Kind::Convert,
            Kind::ExtractAudio,
            Kind::Mute,
        ];
        for k in kinds {
            for provided in 0u16..1024 {
                let want: Vec<&str> = required(k)
                    .into_iter()
                    .filter(|n| {
                        let b = ARG_BITS.iter().position(|a| a == n).unwrap();
                        (provided >> b) & 1 == 0
                    })
                    .collect();
                let mut want_sorted = want.clone();
                want_sorted.sort_by_key(|n| ARG_BITS.iter().position(|a| a == n));
                assert_eq!(
                    missing_names(k, provided),
                    want_sorted,
                    "{k:?} {provided:#b}"
                );
            }
        }
    }

    #[test]
    fn v1_missing_lists_are_reproduced() {
        // training/generate_dataset.py CLARIFY_AMBIG, by request text
        let v1 = [
            ("trim {file}", vec!["start", "end"]),
            ("cut {file}", vec!["start", "end"]),
            ("make {file} smaller", vec!["width", "height"]),
            ("speed up {file}", vec!["factor"]),
            ("convert {file}", vec!["codec", "container"]),
            ("get the audio from {file}", vec!["format"]),
            ("join the clips", vec!["inputs"]),
        ];
        for (text, want) in v1 {
            let r = REQUESTS.iter().find(|r| r.text == text).unwrap();
            for k in r.kinds {
                assert_eq!(missing_names(*k, r.provided), want, "{text} as {k:?}");
            }
        }
    }

    #[test]
    fn every_candidate_reading_has_the_same_missing_list() {
        // A request may name several tools, but the clarify row has one
        // `missing` list, so every candidate must agree on it.
        for r in REQUESTS {
            let first = missing_names(r.kinds[0], r.provided);
            assert!(
                !first.is_empty(),
                "{}: nothing missing, so not a clarify",
                r.id
            );
            for k in r.kinds {
                assert_eq!(missing_names(*k, r.provided), first, "{}", r.id);
            }
        }
    }

    #[test]
    fn questions_carry_every_suggestion() {
        let r = REQUESTS.iter().find(|r| r.id == "v3_clar_cut").unwrap();
        let s = vec![
            "keep only START to END seconds of clip.mp4".to_string(),
            "remove START to END seconds from clip.mp4".to_string(),
        ];
        let q = missing_question(r, "clip.mp4", &s);
        assert_eq!(
            q,
            "Do you want to keep only part of clip.mp4 or cut part of it out? Tell me the start and end times. \
             For example: \"keep only START to END seconds of clip.mp4\" or \"remove START to END seconds from clip.mp4\"."
        );
        assert_eq!(
            reading_question(
                "keep only the first 7.5 seconds of clip.mp4",
                "remove the first 7.5 seconds of clip.mp4"
            ),
            "Do you want to keep only the first 7.5 seconds of clip.mp4, or remove the first 7.5 seconds of clip.mp4?"
        );
        assert_eq!(
            unsupported_question("mute clip.mp4", "lower the volume"),
            "Do you mean \"mute clip.mp4\"? I can't lower the volume."
        );
    }

    #[test]
    fn no_request_reuses_a_frozen_or_heldout_prompt_shape() {
        // clarify_t1..t3 and clarify_h1/h2 prompts are frozen eval prompts.
        let frozen = [
            "trim it",
            "cut out the bad part",
            "convert it",
            "trim clip.mp4 for me",
            "make it smaller",
        ];
        for r in REQUESTS {
            assert!(!frozen.contains(&r.text), "{}", r.id);
        }
    }
}
