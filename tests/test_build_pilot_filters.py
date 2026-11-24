from training.build_pilot import pilot_drop_reason

COMBOS = [["vp9", "webm", 23], ["libx265", "mkv", 18], ["vp9", "mkv", 23]]


def meta(kind="tool", template_id="trim_t1"):
    return {"kind": kind, "template_id": template_id}


def test_general_row_dropped():
    assert pilot_drop_reason("x", {"tool": "trim", "start": 1, "end": 2},
                              meta(kind="general"), COMBOS) == "general"


def test_clarify_invalid_dropped():
    assert pilot_drop_reason("x", {"tool": "clarify", "question": "q", "missing": []},
                              meta(template_id="train_clarify_invalid"), COMBOS) == "clarify_invalid"


def test_percent_dropped():
    assert pilot_drop_reason("cut 50% of clip.mp4", {"tool": "trim", "start": 0, "end": 5},
                              meta(), COMBOS) == "percent_or_half"


def test_half_dropped():
    assert pilot_drop_reason("cut the clip in half", {"tool": "cut_out", "start": 0, "end": 5},
                              meta(), COMBOS) == "percent_or_half"


def test_mmss_fraction_dropped():
    assert pilot_drop_reason("trim from 0:05 to 0:10", {"tool": "trim", "start": 5.3, "end": 10.0},
                              meta(), COMBOS) == "mmss_fraction"


def test_scale_dims_not_in_prompt_dropped():
    assert pilot_drop_reason("scale clip.mp4 down", {"tool": "scale", "width": 640, "height": 480},
                              meta(template_id="scale_t1"), COMBOS) == "scale_dims_not_in_prompt"


def test_scale_dims_in_prompt_kept():
    assert pilot_drop_reason("scale clip.mp4 to 640x480", {"tool": "scale", "width": 640, "height": 480},
                              meta(template_id="scale_t1"), COMBOS) is None


def test_concat_three_unstated_dropped():
    call = {"tool": "concat", "inputs": ["a.mp4", "b.mp4", "c.mp4"]}
    assert pilot_drop_reason("combine a.mp4, b.mp4 and c.mp4", call,
                              meta(template_id="concat_t1"), COMBOS) == "concat_three_unstated"


def test_concat_three_stated_kept():
    call = {"tool": "concat", "inputs": ["a.mp4", "b.mp4", "c.mp4"]}
    assert pilot_drop_reason("combine three clips a.mp4, b.mp4 and c.mp4", call,
                              meta(template_id="concat_t1"), COMBOS) is None


def test_convert_combo_pool_dropped():
    call = {"tool": "convert", "codec": "vp9", "container": "webm", "crf": 23}
    assert pilot_drop_reason("convert clip.mp4 to webm", call,
                              meta(template_id="convert_t1"), COMBOS) == "convert_combo_pool"


def test_extract_mismatch_dropped():
    call = {"tool": "extract_audio", "input": "clip.mp4", "format": "aac"}
    assert pilot_drop_reason("pull the sound from clip.mp4", call,
                              meta(template_id="extract_t1"), COMBOS) == "extract_mismatch"


def test_extract_match_kept():
    call = {"tool": "extract_audio", "input": "clip.mp4", "format": "aac"}
    assert pilot_drop_reason("pull the aac audio from clip.mp4", call,
                              meta(template_id="extract_t1"), COMBOS) is None


def test_clean_trim_row_kept():
    call = {"tool": "trim", "start": 5.0, "end": 15.0}
    assert pilot_drop_reason("trim clip.mp4 from 5 to 15 seconds", call,
                              meta(template_id="trim_t1"), COMBOS) is None
