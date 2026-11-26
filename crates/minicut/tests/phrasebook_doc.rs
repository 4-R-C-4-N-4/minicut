// docs/phrasebook.md and minicut/phrasebook.txt (the Python CLI's --help
// epilog) are generated from phrasebook.rs and must stay in sync.
use std::process::Command;

fn run(args: &[&str]) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_minicut"))
        .args(args)
        .output()
        .unwrap();
    assert!(out.status.success());
    String::from_utf8(out.stdout).unwrap()
}

fn committed(path: &str) -> String {
    std::fs::read_to_string(format!("{}/../../{path}", env!("CARGO_MANIFEST_DIR")))
        .unwrap_or_else(|_| panic!("{path} exists"))
}

#[test]
fn committed_phrasebook_files_match_generator() {
    assert_eq!(
        run(&["phrasebook", "--markdown"]),
        committed("docs/phrasebook.md"),
        "regenerate: target/debug/minicut phrasebook --markdown > docs/phrasebook.md"
    );
    assert_eq!(
        run(&["phrasebook"]),
        committed("minicut/phrasebook.txt"),
        "regenerate: target/debug/minicut phrasebook > minicut/phrasebook.txt"
    );
}

#[test]
fn help_prints_usage_and_phrasebook() {
    for flag in ["--help", "-h", "help"] {
        let h = run(&[flag]);
        assert!(h.starts_with("usage: minicut"), "{flag}");
        assert!(h.contains("How to ask (the MiniCut phrasebook)"), "{flag}");
    }
}

#[test]
fn check_reports_the_reading() {
    assert_eq!(
        run(&[
            "phrasebook",
            "--check",
            "cut the first 10 seconds of clip.mp4"
        ]),
        "Remove\n"
    );
    assert_eq!(
        run(&[
            "phrasebook",
            "--check",
            "trim clip.mp4 to the first 10 seconds"
        ]),
        "Keep\n"
    );
    assert_eq!(run(&["phrasebook", "--check", "trim it"]), "MissingTimes\n");
}
