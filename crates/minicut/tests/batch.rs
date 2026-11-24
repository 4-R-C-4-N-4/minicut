// `minicut validate --batch`: one verdict per input row, never a crash.
use std::io::Write;
use std::process::{Command, Stdio};

#[test]
fn malformed_rows_are_rejections_not_crashes() {
    let good = r#"{"call": "{\"tool\":\"mute\",\"input\":\"a.mp4\"}", "probe": {"duration_s": 30, "width": 1280, "height": 720, "has_audio": true}}"#;
    let rows = [
        good,
        "not json at all",
        r#"{"call": 42, "probe": {"duration_s": 30, "width": 1280, "height": 720, "has_audio": true}}"#,
        r#"{"call": "{\"tool\":\"mute\",\"input\":\"a.mp4\"}", "probe": "nope"}"#,
        good,
    ];
    let mut child = Command::new(env!("CARGO_BIN_EXE_minicut"))
        .args(["validate", "--batch"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all((rows.join("\n") + "\n").as_bytes())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success(), "batch run died: {:?}", out.status);
    let lines: Vec<serde_json::Value> = String::from_utf8(out.stdout)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(lines.len(), rows.len(), "one verdict per row");
    let oks: Vec<bool> = lines.iter().map(|v| v["ok"].as_bool().unwrap()).collect();
    assert_eq!(oks, [true, false, false, false, true]);
    for bad in &lines[1..4] {
        assert!(
            bad["error"].as_str().unwrap().starts_with("bad batch row"),
            "{bad}"
        );
    }
}
