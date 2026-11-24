//! Port of `minicut.build.default_output` (Python), using Rust path
//! semantics that mirror `pathlib.Path.stem` / `.suffix` exactly.

/// Mirrors Python's `Path(name).stem`: the filename with the last
/// extension removed, UNLESS the filename has no dot after position 0
/// (a leading dot, as in dotfiles, does not count as an extension).
fn stem(name: &str) -> &str {
    match name.rfind('.') {
        Some(0) => name,       // ".hidden" -> stem is ".hidden"
        Some(i) => &name[..i], // "a.b.mp4" -> "a.b"
        None => name,          // "noext" -> "noext"
    }
}

/// Mirrors Python's `Path(name).suffix`: everything from (and including)
/// the last dot, or "" if there is none / it's a leading dot.
fn suffix(name: &str) -> &str {
    match name.rfind('.') {
        Some(0) => "",
        Some(i) => &name[i..],
        None => "",
    }
}

pub fn default_output(
    tool: &str,
    input: Option<&str>,
    container: Option<&str>,
    format: Option<&str>,
) -> String {
    if tool == "concat" {
        return "concat_output.mp4".to_string();
    }
    let input = input.expect("default_output: this tool requires `input`");
    if tool == "convert" {
        let container = container.expect("default_output: convert requires `container`");
        return format!("{}_converted.{}", stem(input), container);
    }
    if tool == "extract_audio" {
        let format = format.expect("default_output: extract_audio requires `format`");
        let ext = if format == "aac" { "m4a" } else { format };
        return format!("{}.{}", stem(input), ext);
    }
    let suf = suffix(input).trim_start_matches('.');
    let suf = if suf.is_empty() { "mp4" } else { suf };
    format!("{}_{}.{}", stem(input), tool, suf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn golden_matches_python() {
        let golden = include_str!("../tests/golden/default_output.json");
        let rows: Vec<serde_json::Value> = serde_json::from_str(golden).unwrap();
        assert!(!rows.is_empty());
        for row in &rows {
            let tool = row["tool"].as_str().unwrap();
            let input = row["input"].as_str();
            let container = row["container"].as_str();
            let format = row["format"].as_str();
            let expected = row["expected"].as_str().unwrap();
            let got = default_output(tool, input, container, format);
            assert_eq!(got, expected, "row: {row}");
        }
    }
}
