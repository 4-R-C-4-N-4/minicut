"""Generate minicut/grammar/minicut.gbnf from config/tools.yaml.

The dictionary (formal/dafny/minicut.dfy) pins down what each tool
*means*; this generator pins down what the model may *emit*. Both are
derived from the same source, config/tools.yaml, so the three copies
of the tool contract can no longer drift apart by hand:

  tools.yaml  --generate-->  minicut.gbnf  (shape: field names, types,
                                            enum values, required fields)
  tools.yaml  --modelled-->  formal/dafny/minicut.dfy (meaning: validity,
                                            expected outputs, invariants)
  tools.yaml  --checked--->  minicut/validate.py (runtime: ranges and the
                                            ffprobe context the grammar
                                            cannot see)

Rules of the generated grammar, per tool, in canonical field order
(the order of the few-shot examples in minicut/llm/prompts.py):

  * the tool name is pinned inside its own alternative, so the fields
    that follow belong to that tool only. A wrong field name cannot be
    emitted at all -- that was the v1 baseline failure (55.0%
    verification, arg accuracy 48%: right tool, wrong field name,
    e.g. speed/"speed" for "factor", concat/"input" for "inputs").
  * every field the wrapper requires is required in the grammar, except
    fields with a default in tools.yaml (crf, missing) and "output",
    which the wrapper owns and ignores when the model emits it.
  * value types are pinned: enum fields only accept allowlisted values,
    ints are non-negative digit strings, floats are JSON numbers,
    list[str] fields are string arrays (inputs needs >= 2 items).
  * numeric ranges, start < end, even dims and everything needing the
    probe still live in validate.py. The grammar cannot see the clip.

Usage:
  python minicut/grammar/generate_grammar.py            # rewrite the .gbnf
  python minicut/grammar/generate_grammar.py --check    # exit 1 if it differs

tests/test_grammar.py runs the generator and fails if the committed
.gbnf does not match, so the file cannot drift from tools.yaml.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
TOOLS_YAML = ROOT / "config" / "tools.yaml"
GRAMMAR_PATH = Path(__file__).resolve().parent / "minicut.gbnf"

# File plumbing the model emits but tools.yaml does not list as args:
# every edit tool takes an input file, and may emit an output name the
# wrapper then ignores (it owns output names). concat takes inputs
# instead of input; clarify takes neither.
INPUT_FIELD = ("input", "string", True)
OUTPUT_FIELD = ("output", "string", False)

HEADER = """# MiniCut v1 GBNF grammar -- llama.cpp
# GENERATED from config/tools.yaml by minicut/grammar/generate_grammar.py.
# Do not edit by hand: run `python minicut/grammar/generate_grammar.py`.
# Enforces: exactly one JSON object, tool from the v1 enum (+ clarify),
# and, per tool, exactly the field names in tools.yaml, in canonical
# order, with the right JSON types and enum values. Required fields are
# required; `output`, `crf` and `missing` are optional (defaults in
# tools.yaml). Numeric ranges, start < end, even dims and the ffprobe
# checks live in minicut/validate.py, which has the context the grammar
# cannot see. The meaning of each tool is pinned in
# formal/dafny/minicut.dfy; the three agree by construction (see
# tests/test_grammar.py) and by test (tests/test_formal_dafny.py).
"""


def _value_rule(spec: dict) -> str:
    """GBNF rule name for one tools.yaml arg spec."""
    t = spec["type"]
    if t == "float":
        return "number"
    if t == "int":
        return "int"
    if t == "str":
        return "string"
    if t == "list[str]":
        return "str-list-2" if spec.get("min_items", 0) >= 2 else "str-list"
    if t == "enum":
        alts = " | ".join(_lit(v) for v in spec["values"])
        return f"({alts})"
    raise ValueError(f"unknown arg type: {t!r}")


def _lit(s: str) -> str:
    """A JSON string literal as a GBNF literal, e.g. "input" -> "\"input\""."""
    return '"\\"' + s + '\\""'


def _tool_fields(tool: str, spec: dict) -> list[tuple[str, str, bool]]:
    """(field name, GBNF value rule, required) in canonical order."""
    args = spec.get("args") or {}
    fields: list[tuple[str, str, bool]] = []
    if tool == "concat":
        fields.append(("inputs", _value_rule(args["inputs"]), True))
    elif tool == "clarify":
        fields.append(("question", "string", True))
        fields.append(("missing", _value_rule(args["missing"]),
                       "default" not in args.get("missing", {})))
    else:
        fields.append(INPUT_FIELD)
        for name, arg in args.items():
            fields.append((name, _value_rule(arg), "default" not in arg))
    if tool != "clarify":
        fields.append(OUTPUT_FIELD)
    return fields


def _field_fragment(name: str, value_rule: str, required: bool) -> str:
    frag = f'ws "," ws {_lit(name)} ws ":" ws {value_rule}'
    return frag if required else f"({frag})?"


def generate(tools_yaml: Path = TOOLS_YAML) -> str:
    with tools_yaml.open() as f:
        registry = yaml.safe_load(f)
    tools = registry["tools"]
    lines = [HEADER.rstrip("\n")]
    call_rules = []
    for tool, spec in tools.items():
        # Rule names use hyphens, not underscores: llama.cpp's GBNF
        # parser only accepts [a-zA-Z0-9-] in a rule name, and an
        # underscore ends the name mid-rule (the server then fails to
        # parse the grammar). Tool names inside string literals keep
        # their underscores.
        rule_name = tool.replace("_", "-") + "-call"
        fields = _tool_fields(tool, spec)
        parts = [f'"{{" ws {_lit("tool")} ws ":" ws {_lit(tool)}']
        parts.extend(_field_fragment(n, r, req) for n, r, req in fields)
        parts.append('ws "}"')
        call_rules.append(f"{rule_name} ::= " + " ".join(parts))
    lines.append("root ::= " + " | ".join(
        f"{t.replace('_', '-')}-call" for t in tools))
    lines.extend(call_rules)
    lines.append('string ::= "\\"" ([^"\\\\] | "\\\\" .)* "\\""')
    lines.append('number ::= "-"? [0-9]+ ("." [0-9]+)?')
    lines.append("int ::= [0-9]+")
    lines.append('str-list ::= "[" ws (string (ws "," ws string)*)? ws "]"')
    lines.append('str-list-2 ::= "[" ws string ws "," ws string (ws "," ws string)* ws "]"')
    lines.append("ws ::= [ \\t\\n]*")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="exit 1 if the committed grammar differs")
    args = parser.parse_args(argv)
    generated = generate()
    if args.check:
        current = GRAMMAR_PATH.read_text()
        if current != generated:
            print(f"{GRAMMAR_PATH} is out of date: run "
                  "python minicut/grammar/generate_grammar.py", file=sys.stderr)
            return 1
        return 0
    GRAMMAR_PATH.write_text(generated)
    print(f"wrote {GRAMMAR_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
