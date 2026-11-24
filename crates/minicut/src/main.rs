mod grammar;
mod wire;

use std::process::ExitCode;

const USAGE: &str = "usage: minicut grammar | minicut validate --call JSON --probe JSON | minicut validate --batch < rows.jsonl";

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(|s| s.as_str()) {
        Some("grammar") => {
            print!("{}", grammar::gbnf());
            ExitCode::SUCCESS
        }
        Some("validate") if args.iter().any(|a| a == "--batch") => {
            // JSONL on stdin: {"call": <raw call string>, "probe": {...}} -> {"ok": bool}
            use std::io::BufRead;
            for line in std::io::stdin().lock().lines() {
                let line = line.expect("stdin");
                let row: serde_json::Value = serde_json::from_str(&line).expect("batch row");
                let probe: wire::WireProbe =
                    serde_json::from_value(row["probe"].clone()).expect("probe");
                let ok = wire::validate(row["call"].as_str().expect("call string"), &probe).is_ok();
                println!("{}", serde_json::json!({"ok": ok}));
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
