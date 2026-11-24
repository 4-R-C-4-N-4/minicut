import json

from training import build_verified
from minicut.llm.prompts import SYSTEM_PROMPT

EVAL_COLLISION_PROMPT = "cut the first 10 seconds off clip.mp4"
EVAL_COLLISION_LABEL = {"tool": "cut_out", "input": "clip.mp4", "start": 0.0, "end": 10.0}


def _row(prompt, label):
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": json.dumps(label)},
    ]}


def _meta(template_id="trim_t1", fixture="base_30s", kind="tool", tool="trim"):
    return {"kind": kind, "tool": tool, "fixture": fixture, "template_id": template_id,
            "intent": "x", "seed": 1, "generator": "test"}


def _write(tmp_path, rows_meta):
    rows_path = tmp_path / "rows.jsonl"
    meta_path = tmp_path / "meta.jsonl"
    rows_path.write_text("".join(json.dumps(r) + "\n" for r, _ in rows_meta))
    meta_path.write_text("".join(json.dumps(m) + "\n" for _, m in rows_meta))
    return rows_path, meta_path


def test_eval_prompt_collision_dropped(tmp_path):
    rows_meta = [(_row(EVAL_COLLISION_PROMPT, EVAL_COLLISION_LABEL), _meta(template_id="cut_t1", tool="cut_out"))]
    rows_path, meta_path = _write(tmp_path, rows_meta)
    kept_rows, kept_meta, stats = build_verified.build(rows_path, meta_path)
    assert kept_rows == []
    assert stats["dropped_by_reason"].get("eval_prompt_canon_collision") == 1


def test_heldout_template_dropped(tmp_path):
    rows_meta = [(_row("trim clip.mp4 to 5-15 seconds",
                        {"tool": "trim", "input": "clip.mp4", "start": 5.0, "end": 15.0}),
                  _meta(template_id="trim_h1"))]
    rows_path, meta_path = _write(tmp_path, rows_meta)
    kept_rows, kept_meta, stats = build_verified.build(rows_path, meta_path)
    assert kept_rows == []
    assert stats["dropped_by_reason"].get("heldout_template") == 1


def test_reserved_fixture_dropped(tmp_path):
    rows_meta = [(_row("trim clip.mp4 to 5-15 seconds",
                        {"tool": "trim", "input": "clip.mp4", "start": 5.0, "end": 15.0}),
                  _meta(template_id="trim_t1", fixture="long_60s_1080p"))]
    rows_path, meta_path = _write(tmp_path, rows_meta)
    kept_rows, kept_meta, stats = build_verified.build(rows_path, meta_path)
    assert kept_rows == []
    assert stats["dropped_by_reason"].get("reserved_fixture") == 1


def test_duplicate_dropped(tmp_path):
    row = _row("trim some other unique prompt clip.mp4",
                {"tool": "trim", "input": "clip.mp4", "start": 3.0, "end": 17.0})
    rows_meta = [(row, _meta()), (row, _meta())]
    rows_path, meta_path = _write(tmp_path, rows_meta)
    kept_rows, kept_meta, stats = build_verified.build(rows_path, meta_path)
    assert len(kept_rows) == 1
    assert stats["dropped_by_reason"].get("duplicate") == 1


def test_malformed_dropped(tmp_path):
    bad_row = {"messages": [{"role": "system", "content": "WRONG"},
                             {"role": "user", "content": "x"},
                             {"role": "assistant", "content": "{}"}]}
    rows_path, meta_path = _write(tmp_path, [(bad_row, _meta())])
    kept_rows, kept_meta, stats = build_verified.build(rows_path, meta_path)
    assert kept_rows == []
    assert stats["dropped_by_reason"].get("malformed") == 1


def test_clean_row_kept_byte_identical(tmp_path):
    row = _row("trim some other unique prompt clip.mp4",
                {"tool": "trim", "input": "clip.mp4", "start": 3.0, "end": 17.0})
    rows_meta = [(row, _meta())]
    rows_path, meta_path = _write(tmp_path, rows_meta)
    kept_rows, kept_meta, stats = build_verified.build(rows_path, meta_path)
    assert kept_rows == [rows_path.read_text().splitlines()[0]]
    assert kept_meta == [meta_path.read_text().splitlines()[0]]


def test_stats_counts(tmp_path):
    good = _row("trim some other unique prompt clip.mp4",
                 {"tool": "trim", "input": "clip.mp4", "start": 3.0, "end": 17.0})
    rows_meta = [
        (good, _meta(template_id="trim_t1")),
        (_row(EVAL_COLLISION_PROMPT, EVAL_COLLISION_LABEL), _meta(template_id="cut_t1", tool="cut_out")),
    ]
    rows_path, meta_path = _write(tmp_path, rows_meta)
    kept_rows, kept_meta, stats = build_verified.build(rows_path, meta_path)
    assert stats["input"] == 2
    assert stats["kept"] == 1
    assert stats["kept_by_tool"] == {"trim": 1}
