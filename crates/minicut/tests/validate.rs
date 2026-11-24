// `minicut validate`: the wire boundary rules, pinned case by case.
use std::process::Command;

const PROBE: &str = r#"{"duration_s": 30.0, "width": 1280, "height": 720, "has_audio": true}"#;

fn validate(call: &str) -> (i32, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_minicut"))
        .args(["validate", "--call", call, "--probe", PROBE])
        .output()
        .unwrap();
    (
        out.status.code().unwrap(),
        String::from_utf8(out.stdout).unwrap(),
    )
}

/// Accepted (exit 0) or rejected (exit 1). Anything else, e.g. a usage
/// error, fails the test, so rejection cases cannot pass vacuously.
fn accepts(call: &str) -> bool {
    let (code, out) = validate(call);
    assert!(
        code == 0 || code == 1,
        "exit {code}, expected a verdict: {out}"
    );
    assert_eq!(
        code == 0,
        out.contains(r#""ok":true"#),
        "exit code and JSON disagree: {out}"
    );
    code == 0
}

#[test]
fn accepts_valid_calls() {
    assert!(accepts(
        r#"{"tool":"trim","input":"clip.mp4","start":12,"end":20.5}"#
    ));
    assert!(accepts(
        r#"{"tool":"cut_out","input":"clip.mp4","start":0.0,"end":10.0}"#
    ));
    assert!(accepts(r#"{"tool":"concat","inputs":["a.mp4","b.mp4"]}"#));
    assert!(accepts(
        r#"{"tool":"scale","input":"clip.mp4","width":640,"height":360}"#
    ));
    assert!(accepts(
        r#"{"tool":"speed","input":"clip.mp4","factor":2.0}"#
    ));
    assert!(accepts(
        r#"{"tool":"convert","input":"clip.mp4","codec":"vp9","container":"webm"}"#
    ));
    assert!(accepts(
        r#"{"tool":"extract_audio","input":"clip.mp4","format":"aac"}"#
    ));
    assert!(accepts(
        r#"{"tool":"mute","input":"clip.mp4","output":"x.mp4"}"#
    ));
    assert!(accepts(
        r#"{"tool":"clarify","question":"Which times?","missing":["start","end"]}"#
    ));
}

#[test]
fn clarify_reports_question_and_missing() {
    let (code, out) =
        validate(r#"{"tool":"clarify","question":"Which times?","missing":["start","end"]}"#);
    assert_eq!(code, 0);
    assert!(out.contains(r#""clarify":"Which times?""#), "{out}");
    assert!(out.contains(r#""missing":["start","end"]"#), "{out}");
}

#[test]
fn rejects_no_op_edits() {
    assert!(!accepts(
        r#"{"tool":"trim","input":"clip.mp4","start":0,"end":30}"#
    ));
    assert!(!accepts(
        r#"{"tool":"scale","input":"clip.mp4","width":1280,"height":720}"#
    ));
    assert!(!accepts(
        r#"{"tool":"speed","input":"clip.mp4","factor":1.0}"#
    ));
}

#[test]
fn rejects_wrong_shapes_and_types() {
    // the v1 failures: right tool, wrong field name
    assert!(!accepts(
        r#"{"tool":"speed","input":"clip.mp4","speed":2.0}"#
    ));
    assert!(!accepts(r#"{"tool":"concat","input":["a.mp4","b.mp4"]}"#));
    assert!(!accepts(r#"{"tool":"concat","inputs":["a.mp4"]}"#));
    assert!(!accepts(
        r#"{"tool":"scale","input":"clip.mp4","width":640.0,"height":360}"#
    ));
    assert!(!accepts(
        r#"{"tool":"convert","input":"clip.mp4","codec":"h264","container":"mp4"}"#
    ));
    assert!(!accepts(
        r#"{"tool":"convert","input":"clip.mp4","codec":"vp9","container":"mp4","crf":52}"#
    ));
    assert!(!accepts(
        r#"{"tool":"trim","input":"clip.mp4","start":true,"end":5}"#
    ));
    assert!(!accepts(
        r#"{"tool":"trim","input":"clip.mp4","start":-1,"end":5}"#
    ));
    assert!(!accepts(r#"{"tool":"rotate","input":"clip.mp4"}"#));
    assert!(!accepts(r#"{"tool":"clarify","question":"   "}"#));
}

#[test]
fn rejects_duplicate_keys() {
    // Python's json.loads keeps the last value; Rust refuses the call. The
    // stricter side is intended (docs/rust-port.md, open question 5).
    assert!(!accepts(
        r#"{"tool":"mute","input":"a.mp4","input":"b.mp4"}"#
    ));
}

#[test]
fn rejects_unsafe_paths() {
    assert!(!accepts(r#"{"tool":"mute","input":"/etc/passwd"}"#));
    assert!(!accepts(r#"{"tool":"mute","input":"../clip.mp4"}"#));
    assert!(!accepts(r#"{"tool":"mute","input":""}"#));
    assert!(!accepts(
        r#"{"tool":"mute","input":"clip.mp4","output":"../x.mp4"}"#
    ));
}

#[test]
fn ranges_follow_the_probe() {
    assert!(!accepts(
        r#"{"tool":"trim","input":"clip.mp4","start":5,"end":30.001}"#
    ));
    assert!(!accepts(
        r#"{"tool":"cut_out","input":"clip.mp4","start":10,"end":10}"#
    ));
    assert!(!accepts(
        r#"{"tool":"speed","input":"clip.mp4","factor":4.001}"#
    ));
    assert!(accepts(
        r#"{"tool":"speed","input":"clip.mp4","factor":0.25}"#
    ));
}

#[test]
fn usage_errors_exit_2() {
    let out = Command::new(env!("CARGO_BIN_EXE_minicut"))
        .arg("validate")
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));
}
