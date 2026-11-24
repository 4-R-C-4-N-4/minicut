mod grammar;

use std::process::ExitCode;

const USAGE: &str = "usage: minicut grammar";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(|s| s.as_str()) {
        Some("grammar") => {
            print!("{}", grammar::gbnf());
            ExitCode::SUCCESS
        }
        _ => {
            eprintln!("{USAGE}");
            ExitCode::from(2)
        }
    }
}
