// docs/templates.md is generated from templates.rs and must stay in sync.
use std::process::Command;

#[test]
fn committed_doc_matches_generator() {
    let out = Command::new(env!("CARGO_BIN_EXE_minicut"))
        .args(["templates", "--markdown"])
        .output()
        .unwrap();
    assert!(out.status.success());
    let doc = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../docs/templates.md"
    ))
    .expect("docs/templates.md exists");
    assert_eq!(
        String::from_utf8(out.stdout).unwrap(),
        doc,
        "regenerate: target/debug/minicut templates --markdown > docs/templates.md"
    );
}
