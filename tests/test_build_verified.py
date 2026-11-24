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


def test_combo_arg_pool_dropped(tmp_path):
    # eval/freeze.json rule: "Combo arg pools must never be used in training data."
    rows_meta = [
        (_row("speed up train.mp4 by 1.5x", {"tool": "speed", "input": "train.mp4", "factor": 1.5}),
         _meta(template_id="v3_test", tool="speed")),
        (_row("convert train.mp4 to mkv with libx265",
              {"tool": "convert", "input": "train.mp4", "codec": "libx265", "container": "mkv", "crf": 18}),
         _meta(template_id="v3_test", tool="convert")),
        (_row("extract the audio from train.mp4 as wav",
              {"tool": "extract_audio", "input": "train.mp4", "format": "wav"}),
         _meta(template_id="v3_test", tool="extract_audio")),
        (_row("keep only 8 to 22 seconds of train.mp4",
              {"tool": "trim", "input": "train.mp4", "start": 8.0, "end": 22.0}),
         _meta(template_id="v3_test", tool="trim")),
        (_row("make train.mp4 854x480", {"tool": "scale", "input": "train.mp4", "width": 854, "height": 480}),
         _meta(template_id="v3_test", tool="scale")),
        (_row("speed up train.mp4 by 1.25x", {"tool": "speed", "input": "train.mp4", "factor": 1.25}),
         _meta(template_id="v3_test", tool="speed")),
    ]
    rows_path, meta_path = _write(tmp_path, rows_meta)
    kept_rows, kept_meta, stats = build_verified.build(rows_path, meta_path)
    assert stats["dropped_by_reason"].get("combo_arg_pool") == 5
    assert len(kept_rows) == 1 and "1.25" in kept_rows[0]


def test_targets_keep_first_n_per_tool_and_fail_when_short(tmp_path):
    rows_meta = []
    for i in range(5):
        rows_meta.append((_row(f"mute target{i}.mp4", {"tool": "mute", "input": f"target{i}.mp4"}),
                          _meta(template_id="v3_test", tool="mute")))
    for i in range(3):
        rows_meta.append((_row(f"ask {i}", {"tool": "clarify", "question": f"q{i}", "missing": []}),
                          _meta(template_id="v3_amb_test", tool="clarify", kind="clarify_reading", fixture=None)))
    rows_path, meta_path = _write(tmp_path, rows_meta)
    kept_rows, kept_meta, stats = build_verified.build(
        rows_path, meta_path, targets={"mute": 2, "clarify_reading": 3})
    assert [json.loads(r)["messages"][1]["content"] for r in kept_rows] == [
        "mute target0.mp4", "mute target1.mp4", "ask 0", "ask 1", "ask 2"]
    assert stats["dropped_by_reason"]["over_target"] == 3
    assert stats["short_of_target"] == {}
    _, _, stats = build_verified.build(rows_path, meta_path, targets={"mute": 9})
    assert stats["short_of_target"] == {"mute": {"target": 9, "kept": 5}}
