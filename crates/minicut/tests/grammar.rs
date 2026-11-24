// The emitted grammar's rules must equal the llama.cpp-validated golden file.
use std::process::Command;

fn rules(s: &str) -> Vec<String> {
    s.lines()
        .filter(|l| !l.starts_with('#'))
        .map(String::from)
        .collect()
}

#[test]
fn grammar_rules_match_golden() {
    let out = Command::new(env!("CARGO_BIN_EXE_minicut"))
        .arg("grammar")
        .output()
        .unwrap();
    assert!(out.status.success());
    let got = String::from_utf8(out.stdout).unwrap();
    let golden = include_str!("golden/minicut.gbnf");
    assert_eq!(rules(&got), rules(golden));
}
