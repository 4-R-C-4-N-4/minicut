"""Prompts — the exact strings used at training AND inference.

Do not change the system prompt without retraining / re-baselining.
"""
SYSTEM_PROMPT = (
    "You are MiniCut, a local ffmpeg editor. "
    "Output exactly one JSON tool call, nothing else. "
    "Tools: trim, cut_out, concat, scale, speed, convert, extract_audio, mute, clarify. "
    "Times are seconds (float). If an arg is missing or ambiguous, "
    "output a clarify tool call instead of guessing."
)

FEW_SHOT = [
    ("trim clip.mp4 from 12 seconds to 45 seconds",
     '{"tool": "trim", "input": "clip.mp4", "start": 12.0, "end": 45.0, "output": "clip_trim.mp4"}'),
    ("cut the first 10 seconds off clip.mp4",
     '{"tool": "cut_out", "input": "clip.mp4", "start": 0.0, "end": 10.0, "output": "clip_cut_out.mp4"}'),
    ("make clip.mp4 1280x720",
     '{"tool": "scale", "input": "clip.mp4", "width": 1280, "height": 720, "output": "clip_scale.mp4"}'),
    # Added 2026-11-20 for baseline_v2: baseline_v1 showed the model inventing
    # arg names for exactly these four tools, so each example teaches the exact
    # field names the wrapper validates (inputs, factor, codec/container/crf, format).
    ("join clip.mp4 to itself",
     '{"tool": "concat", "inputs": ["clip.mp4", "clip.mp4"], "output": "clip_concat.mp4"}'),
    ("make clip.mp4 twice as fast",
     '{"tool": "speed", "input": "clip.mp4", "factor": 2.0, "output": "clip_speed.mp4"}'),
    ("convert clip.mp4 to mkv with libx264",
     '{"tool": "convert", "input": "clip.mp4", "codec": "libx264", "container": "mkv", "crf": 23, "output": "clip_converted.mkv"}'),
    ("extract the audio from clip.mp4 as mp3",
     '{"tool": "extract_audio", "input": "clip.mp4", "format": "mp3", "output": "clip_audio.mp3"}'),
    ("trim it",
     '{"tool": "clarify", "question": "What start and end times should I trim to?", "missing": ["start", "end"]}'),
]

RETRY_PROMPT = (
    "Your last tool call failed. Here is the probe and the error. "
    "Return a corrected JSON tool call, nothing else."
)
