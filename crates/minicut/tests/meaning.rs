//! `minicut meaning`: kept source intervals for an accepted edit.
use std::process::Command;

fn run(args: &[&str]) -> (std::process::Output, serde_json::Value) {
    let out = Command::new(env!("CARGO_BIN_EXE_minicut"))
        .args(args)
        .output()
        .unwrap();
    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).unwrap_or(serde_json::Value::Null);
    (out, v)
}

const PROBE_30S: &str = r#"{"duration_s": 30, "width": 1280, "height": 720, "has_audio": true}"#;

#[test]
fn trim_keeps_the_trimmed_span() {
    let call = r#"{"tool":"trim","input":"a.mp4","start":5,"end":12}"#;
    let (out, v) = run(&["meaning", "--call", call, "--probe", PROBE_30S]);
    assert!(out.status.success());
    assert_eq!(v["kept"], serde_json::json!([[5000, 12000]]));
}

#[test]
fn cut_out_front_keeps_the_remainder() {
    let call = r#"{"tool":"cut_out","input":"a.mp4","start":0,"end":5}"#;
    let (out, v) = run(&["meaning", "--call", call, "--probe", PROBE_30S]);
    assert!(out.status.success());
    assert_eq!(v["kept"], serde_json::json!([[5000, 30000]]));
}

#[test]
fn cut_out_middle_keeps_both_sides() {
    let call = r#"{"tool":"cut_out","input":"a.mp4","start":10,"end":20}"#;
    let (out, v) = run(&["meaning", "--call", call, "--probe", PROBE_30S]);
    assert!(out.status.success());
    assert_eq!(v["kept"], serde_json::json!([[0, 10000], [20000, 30000]]));
}

#[test]
fn mute_keeps_the_whole_clip() {
    let call = r#"{"tool":"mute","input":"a.mp4"}"#;
    let (out, v) = run(&["meaning", "--call", call, "--probe", PROBE_30S]);
    assert!(out.status.success());
    assert_eq!(v["kept"], serde_json::json!([[0, 30000]]));
}

#[test]
fn whole_clip_trim_is_a_noop_not_accepted() {
    let call = r#"{"tool":"trim","input":"a.mp4","start":0,"end":30}"#;
    let (out, v) = run(&["meaning", "--call", call, "--probe", PROBE_30S]);
    assert_eq!(out.status.code(), Some(1));
    assert_eq!(v["ok"], serde_json::json!(false));
}

#[test]
fn clarify_has_no_kept_intervals() {
    let call = r#"{"tool":"clarify","question":"start and end?"}"#;
    let (out, v) = run(&["meaning", "--call", call, "--probe", PROBE_30S]);
    assert_eq!(out.status.code(), Some(1));
    assert_eq!(v["ok"], serde_json::json!(false));
}

#[test]
fn missing_flags_is_usage_error() {
    let out = Command::new(env!("CARGO_BIN_EXE_minicut"))
        .args(["meaning"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));
}
