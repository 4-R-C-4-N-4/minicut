import json
import stat
import sys

from scripts import accept_dataset
from minicut.llm.prompts import SYSTEM_PROMPT


def _row(prompt, label):
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": json.dumps(label)},
    ]}


def _meta(template_id="trim_t1", fixture="base_30s", kind="tool", tool="trim"):
    return {"kind": kind, "tool": tool, "fixture": fixture, "template_id": template_id,
            "intent": "x", "seed": 1, "generator": "test"}


GOOD = [
    ("trim clip.mp4 to the first 5 seconds",
     {"tool": "trim", "input": "clip.mp4", "start": 0.0, "end": 5.0}, _meta()),
    ("mute clip.mp4",
     {"tool": "mute", "input": "clip.mp4"}, _meta(template_id="mute_t1", tool="mute")),
    ("scale clip.mp4 to 640x480",
     {"tool": "scale", "input": "clip.mp4", "width": 640, "height": 480},
     _meta(template_id="scale_t1", tool="scale")),
]


def _write_rows_meta(tmp_path, cases):
    rows_path = tmp_path / "rows.jsonl"
    meta_path = tmp_path / "meta.jsonl"
    rows_path.write_text("".join(json.dumps(_row(p, l)) + "\n" for p, l, _ in cases))
    meta_path.write_text("".join(json.dumps(m) + "\n" for _, _, m in cases))
    return rows_path, meta_path


def test_pilot_filters_and_validator_pass_on_clean_rows(tmp_path):
    rows_path, meta_path = _write_rows_meta(tmp_path, GOOD)
    cases = accept_dataset._rows_and_meta(rows_path, meta_path)
    ok, detail = accept_dataset.check_pilot_filters(cases)
    assert ok, detail
    ok, detail = accept_dataset.check_validator(cases)
    assert ok, detail


def test_validator_fails_on_bad_label(tmp_path):
    bad = [("trim clip.mp4 badly",
            {"tool": "trim", "input": "clip.mp4", "start": 999.0, "end": 1000.0}, _meta())]
    rows_path, meta_path = _write_rows_meta(tmp_path, bad)
    cases = accept_dataset._rows_and_meta(rows_path, meta_path)
    ok, detail = accept_dataset.check_validator(cases)
    assert not ok


def test_pilot_filters_fails_on_heldout_template(tmp_path):
    bad = [("trim clip.mp4 to the first 5 seconds",
            {"tool": "trim", "input": "clip.mp4", "start": 0.0, "end": 5.0},
            _meta(template_id="train_clarify_invalid", tool="trim"))]
    rows_path, meta_path = _write_rows_meta(tmp_path, bad)
    cases = accept_dataset._rows_and_meta(rows_path, meta_path)
    ok, detail = accept_dataset.check_pilot_filters(cases)
    assert not ok


def test_main_skips_grammar_and_preflight_and_release_without_flags(tmp_path, capsys):
    rows_path, meta_path = _write_rows_meta(tmp_path, GOOD)
    argv = sys.argv
    sys.argv = ["accept_dataset.py", "--rows", str(rows_path), "--meta", str(meta_path)]
    try:
        accept_dataset.main()
    except SystemExit as exc:
        assert exc.code in (0, None)
    finally:
        sys.argv = argv
    out = capsys.readouterr().out
    assert "SKIP grammar" in out
    assert "SKIP preflight" in out
    assert "SKIP release" in out
    assert "PASS pilot_filters" in out
    assert "PASS validator" in out


FAKE_BIN = """#!/usr/bin/env python3
import json, sys
call = json.loads(sys.argv[sys.argv.index('--call') + 1])
probe = json.loads(sys.argv[sys.argv.index('--probe') + 1])
if call['input'] == 'bad.mp4':
    print(json.dumps({"kept": [[0, 1000]]}))
else:
    start_ms, end_ms = int(call['start'] * 1000), int(call['end'] * 1000)
    if call['tool'] == 'trim':
        print(json.dumps({"kept": [[start_ms, end_ms]]}))
    else:
        dur_ms = int(probe['duration_s'] * 1000)
        kept = []
        if start_ms > 0:
            kept.append([0, start_ms])
        if end_ms < dur_ms:
            kept.append([end_ms, dur_ms])
        print(json.dumps({"kept": kept}))
"""


def test_release_check_with_fake_bin(tmp_path):
    fake = tmp_path / "fakebin.py"
    fake.write_text(FAKE_BIN)
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    ok, detail = accept_dataset.check_release(str(fake))
    # The fake binary answers correctly for genuine trim/cut_out release
    # cases but will mis-answer any case whose input is literally bad.mp4
    # (none of the frozen cases use that name, so this run should pass).
    assert ok, detail


def test_release_check_detects_mismatch(tmp_path, monkeypatch):
    fake = tmp_path / "fakebin.py"
    fake.write_text(FAKE_BIN)
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)

    import json as _json
    from eval.release import bench

    real_read = accept_dataset._read_jsonl

    def broken_read(path):
        rows = real_read(path)
        if path.name == "cases.jsonl":
            orig_input = rows[0]["expected"]["call"]["input"]
            fixture_id = rows[0]["files"].pop(orig_input)
            rows[0]["expected"]["call"] = dict(rows[0]["expected"]["call"], input="bad.mp4")
            rows[0]["files"]["bad.mp4"] = fixture_id
        return rows

    monkeypatch.setattr(accept_dataset, "_read_jsonl", broken_read)
    ok, detail = accept_dataset.check_release(str(fake))
    assert not ok
