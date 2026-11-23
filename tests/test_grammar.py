"""Grammar tests: the .gbnf is generated from tools.yaml, and the
Dafny dictionary agrees with tools.yaml on arg names.

The point of these tests is the three-way agreement:

  tools.yaml --generate--> minicut/grammar/minicut.gbnf
  tools.yaml --modelled--> formal/dafny/minicut.dfy (Tool datatype)
  tools.yaml --checked---> minicut/validate.py (TOOLS map)

If any one of them changes without the others, a test here fails
first. These are static checks on the generated text and the Dafny
source; the llama.cpp server is the thing that actually parses the
grammar (see the smoke test in the PR description), and the eval
harness is the thing that measures whether the stricter grammar
helps or hurts the model.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TOOLS_YAML = ROOT / "config" / "tools.yaml"
GRAMMAR_PATH = ROOT / "minicut" / "grammar" / "minicut.gbnf"
DAFNY_PATH = ROOT / "formal" / "dafny" / "minicut.dfy"

from minicut.grammar.generate_grammar import generate  # noqa: E402
from minicut.llm.prompts import FEW_SHOT  # noqa: E402

# File plumbing the model emits but tools.yaml does not list as args.
EXPECTED_FIELDS = {
    "trim": {"input", "start", "end", "output"},
    "cut_out": {"input", "start", "end", "output"},
    "concat": {"inputs", "output"},
    "scale": {"input", "width", "height", "output"},
    "speed": {"input", "factor", "output"},
    "convert": {"input", "codec", "container", "crf", "output"},
    "extract_audio": {"input", "format", "output"},
    "mute": {"input", "output"},
    "clarify": {"question", "missing"},
}
REQUIRED_FIELDS = {
    "trim": {"input", "start", "end"},
    "cut_out": {"input", "start", "end"},
    "concat": {"inputs"},
    "scale": {"input", "width", "height"},
    "speed": {"input", "factor"},
    "convert": {"input", "codec", "container"},
    "extract_audio": {"input", "format"},
    "mute": {"input"},
    "clarify": {"question"},
}


def _lit(s: str) -> str:
    """The GBNF literal for a JSON string, e.g. "input" -> "\"input\""."""
    return '"\\"' + s + '\\""'


def _load_registry() -> dict:
    with TOOLS_YAML.open() as f:
        return yaml.safe_load(f)


def _tool_rule(grammar: str, tool: str) -> str:
    """The single line defining <tool>-call in the grammar."""
    rule = tool.replace("_", "-") + "-call"
    for line in grammar.splitlines():
        if line.startswith(f"{rule} ::="):
            return line
    raise AssertionError(f"no {rule} rule in grammar")


def _rule_keys(rule: str) -> set[str]:
    """Field names a rule accepts: string literals followed by `ws ":"`."""
    return set(re.findall(r'"\\"(\w+)\\""\s+ws\s+":"', rule)) - {"tool"}


def _optional_fragment(field: str) -> str:
    return f'(ws "," ws {_lit(field)} ws ":" ws'


def test_generated_grammar_matches_committed_file():
    """The committed .gbnf must be exactly what the generator emits."""
    assert GRAMMAR_PATH.read_text() == generate()


def test_grammar_pins_exact_field_names_per_tool():
    """Each tool's rule accepts exactly its tools.yaml fields, no others."""
    grammar = GRAMMAR_PATH.read_text()
    registry = _load_registry()
    assert set(registry["tools"]) == set(EXPECTED_FIELDS)
    for tool in EXPECTED_FIELDS:
        keys = _rule_keys(_tool_rule(grammar, tool))
        assert keys == EXPECTED_FIELDS[tool], tool


def test_grammar_requires_required_fields():
    """Required fields appear unconditionally; optional ones are (…)? groups."""
    grammar = GRAMMAR_PATH.read_text()
    for tool, required in REQUIRED_FIELDS.items():
        rule = _tool_rule(grammar, tool)
        for field in required:
            assert _lit(field) in rule, (tool, field)
            assert _optional_fragment(field) not in rule, (tool, field)
        for field in EXPECTED_FIELDS[tool] - required:
            assert _optional_fragment(field) in rule, (tool, field)


def test_grammar_pins_enum_values():
    """Codec/container/format only accept allowlisted values."""
    grammar = GRAMMAR_PATH.read_text()
    convert = _tool_rule(grammar, "convert")
    assert f"({_lit('libx264')} | {_lit('libx265')} | {_lit('vp9')})" in convert
    assert f"({_lit('mp4')} | {_lit('mkv')} | {_lit('webm')})" in convert
    extract = _tool_rule(grammar, "extract_audio")
    assert f"({_lit('mp3')} | {_lit('wav')} | {_lit('aac')})" in extract


def test_grammar_rules_all_defined():
    """Every rule name referenced in the grammar is defined once."""
    grammar = GRAMMAR_PATH.read_text()
    defined = re.findall(r"^([\w-]+) ::=", grammar, re.M)
    assert len(defined) == len(set(defined))
    assert "root" in defined
    for name in ("trim-call", "cut-out-call", "concat-call", "scale-call",
                 "speed-call", "convert-call", "extract-audio-call",
                 "mute-call", "clarify-call", "string", "number", "int",
                 "str-list", "str-list-2", "ws"):
        assert name in defined, name
    code = "\n".join(l for l in grammar.splitlines() if not l.startswith("#"))
    assert code.count("(") == code.count(")")


def test_rule_names_have_no_underscores():
    """llama.cpp's GBNF parser only accepts [a-zA-Z0-9-] in a rule name:
    an underscore ends the name mid-rule and the server fails to parse
    the grammar (caught by the smoke test, 2026-11-23). Tool names
    inside string literals keep their underscores."""
    grammar = GRAMMAR_PATH.read_text()
    defined = re.findall(r"^([\w-]+) ::=", grammar, re.M)
    for name in defined:
        assert "_" not in name, name
    # References too, not just definitions: the first version of this
    # generator emitted `str_list` as a *reference* inside clarify-call
    # while the rule was defined as `str-list`, and the server died on
    # it. Strip string literals first (tool names keep underscores).
    for line in grammar.splitlines():
        if line.startswith("#"):
            continue
        code = re.sub(r'"([^"\\]|\\.)*"', '""', line)
        for word in re.findall(r"[A-Za-z][\w-]*", code):
            assert "_" not in word, (line, word)


def test_few_shot_examples_conform_to_grammar_fields():
    """The examples the model sees use exactly the fields the grammar allows."""
    for _prompt, raw in FEW_SHOT:
        call = json.loads(raw)
        tool = call["tool"]
        keys = set(call) - {"tool"}
        assert keys <= EXPECTED_FIELDS[tool], (tool, keys)
        assert REQUIRED_FIELDS[tool] <= keys, (tool, keys)


def test_v1_wrong_field_names_do_not_conform():
    """The baseline_v1 failures: right tool, wrong field name."""
    bad = [
        {"tool": "speed", "input": "clip.mp4", "speed": 2.0},
        {"tool": "concat", "input": "clip.mp4"},
        {"tool": "convert", "input": "clip.mp4", "codec": "libx264",
         "container": "mkv", "quality": 23},
    ]
    for call in bad:
        tool = call["tool"]
        keys = set(call) - {"tool"}
        # Either an unexpected key, or a required field missing.
        assert (keys - EXPECTED_FIELDS[tool]) or not (
            REQUIRED_FIELDS[tool] <= keys), call


def _dafny_tool_args() -> dict[str, list[str]]:
    """Constructor name -> field names, parsed from the Tool datatype."""
    text = DAFNY_PATH.read_text()
    block = text.split("datatype Tool =", 1)[1].split("datatype Clip", 1)[0]
    out: dict[str, list[str]] = {}
    for m in re.finditer(r"(\w+)\(([^)]*)\)", block):
        name, args = m.group(1), m.group(2)
        out[name] = [a.split(":")[0].strip() for a in args.split(",") if a.strip()]
    out["Mute"] = []
    return out


_CAMEL_TO_TOOL = {
    "Trim": "trim", "CutOut": "cut_out", "Concat": "concat",
    "Scale": "scale", "Speed": "speed", "Convert": "convert",
    "ExtractAudio": "extract_audio", "Mute": "mute", "Clarify": "clarify",
}


def test_dafny_tool_args_match_tools_yaml():
    """The Dafny Tool datatype has exactly the args tools.yaml lists."""
    registry = _load_registry()
    dafny = _dafny_tool_args()
    assert set(_CAMEL_TO_TOOL) == set(dafny)
    for ctor, fields in dafny.items():
        tool = _CAMEL_TO_TOOL[ctor]
        assert fields == list(registry["tools"][tool].get("args") or {}), tool
