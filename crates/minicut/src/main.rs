#[cfg_attr(not(test), allow(dead_code))] // used by the label subcommand (Task 3.17)
mod clarify;
mod grammar;
mod templates;
mod wire;

use std::process::ExitCode;

const USAGE: &str = "usage: minicut grammar | minicut templates --markdown | minicut validate --call JSON --probe JSON | minicut validate --batch < rows.jsonl";

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(|s| s.as_str()) {
        Some("templates") if args.iter().any(|a| a == "--markdown") => {
            print!("{}", templates::markdown());
            ExitCode::SUCCESS
        }
        Some("grammar") => {
            print!("{}", grammar::gbnf());
            ExitCode::SUCCESS
        }
        Some("validate") if args.iter().any(|a| a == "--batch") => {
            // JSONL on stdin: {"call": <raw call string>, "probe": {...}}.
            // One verdict per row; a malformed row is a rejection, not a crash.
            use std::io::BufRead;
            for line in std::io::stdin().lock().lines() {
                let verdict = match line.map_err(|e| e.to_string()).and_then(|l| batch_row(&l)) {
                    Ok(Ok(())) => serde_json::json!({"ok": true}),
                    Ok(Err(e)) => serde_json::json!({"ok": false, "error": e}),
                    Err(e) => {
                        serde_json::json!({"ok": false, "error": format!("bad batch row: {e}")})
                    }
                };
                println!("{verdict}");
            }
            ExitCode::SUCCESS
        }
        Some("validate") => {
            let (Some(call), Some(probe)) = (flag(&args, "--call"), flag(&args, "--probe")) else {
                eprintln!("{USAGE}");
                return ExitCode::from(2);
            };
            let probe: wire::WireProbe = match serde_json::from_str(probe) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("bad --probe: {e}");
                    return ExitCode::from(2);
                }
            };
            let result = wire::validate(call, &probe);
            let out = match &result {
                Ok(wire::Parsed::Edit(_)) => serde_json::json!({"ok": true}),
                Ok(wire::Parsed::Clarify { question, missing }) => {
                    serde_json::json!({"ok": true, "clarify": question, "missing": missing})
                }
                Err(e) => serde_json::json!({"ok": false, "error": e}),
            };
            println!("{out}");
            if result.is_ok() {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(1)
            }
        }
        _ => {
            eprintln!("{USAGE}");
            ExitCode::from(2)
        }
    }
}

/// Outer error: the row itself is malformed. Inner result: the validator's verdict.
fn batch_row(line: &str) -> Result<Result<(), String>, String> {
    let row: serde_json::Value = serde_json::from_str(line).map_err(|e| e.to_string())?;
    let call = row["call"].as_str().ok_or("`call` must be a JSON string")?;
    let probe: wire::WireProbe =
        serde_json::from_value(row["probe"].clone()).map_err(|e| format!("probe: {e}"))?;
    Ok(wire::validate(call, &probe).map(|_| ()))
}
