mod clarify;
mod grammar;
mod label;
mod naming;
mod prompts;
mod render;
mod rng;
mod templates;
mod wire;

use std::process::ExitCode;

const USAGE: &str = "usage: minicut grammar | minicut templates --markdown | minicut validate --call JSON --probe JSON | minicut validate --batch < rows.jsonl | minicut meaning --call JSON --probe JSON | minicut system-prompt | minicut label --seed N --out ROWS --meta META [--oversample PCT]";

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(|s| s.as_str()) {
        Some("label") => {
            let seed = flag(&args, "--seed").and_then(|s| s.parse::<u64>().ok());
            let (Some(seed), Some(out), Some(meta)) =
                (seed, flag(&args, "--out"), flag(&args, "--meta"))
            else {
                eprintln!("{USAGE}");
                return ExitCode::from(2);
            };
            let mut g = label::Generator::new(seed);
            let over = flag(&args, "--oversample").map_or(Some(0), |s| s.parse::<usize>().ok());
            let Some(over) = over else {
                eprintln!("--oversample takes a percentage, e.g. 50");
                return ExitCode::from(2);
            };
            let rows = g.run(&label::Counts::pilot_matched().oversampled(over));
            let lines: String = rows.iter().map(|r| r.line() + "\n").collect();
            let metas: String = rows.iter().map(|r| label::dumps(&r.meta) + "\n").collect();
            if let Err(e) = std::fs::write(out, lines).and_then(|_| std::fs::write(meta, metas)) {
                eprintln!("cannot write: {e}");
                return ExitCode::from(1);
            }
            eprintln!(
                "{} rows ({} resampled draws) -> {out}, {meta}",
                rows.len(),
                g.resampled
            );
            ExitCode::SUCCESS
        }
        Some("templates") if args.iter().any(|a| a == "--markdown") => {
            print!("{}{}", templates::markdown(), clarify::markdown());
            ExitCode::SUCCESS
        }
        Some("grammar") => {
            print!("{}", grammar::gbnf());
            ExitCode::SUCCESS
        }
        Some("system-prompt") => {
            print!("{}", prompts::SYSTEM_PROMPT);
            ExitCode::SUCCESS
        }
        Some("meaning") => {
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
            let result = wire::parse(call).and_then(|c| wire::to_parsed(&c));
            match result {
                Ok(wire::Parsed::Edit(edit)) => {
                    match minicut_core::kept_intervals(edit, wire::clip(&probe)) {
                        Some((a, b)) => {
                            let kept: Vec<[u64; 2]> =
                                [a, b].into_iter().flatten().map(|(x, y)| [x, y]).collect();
                            println!("{}", serde_json::json!({ "kept": kept }));
                            ExitCode::SUCCESS
                        }
                        None => {
                            println!(
                                "{}",
                                serde_json::json!({"ok": false, "error": "edit not accepted"})
                            );
                            ExitCode::from(1)
                        }
                    }
                }
                Ok(wire::Parsed::Clarify { .. }) => {
                    println!(
                        "{}",
                        serde_json::json!({"ok": false, "error": "clarify has no kept intervals"})
                    );
                    ExitCode::from(1)
                }
                Err(e) => {
                    println!("{}", serde_json::json!({"ok": false, "error": e}));
                    ExitCode::from(1)
                }
            }
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
