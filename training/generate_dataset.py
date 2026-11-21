"""Generate MiniCut Phase 4 training dataset v1 — training/train_v1.jsonl.

Label first, prompt derived from it, never the other way round
(guide §7.1). Every label is validated against its fixture's probe
with minicut.validate before it is written; a label that fails is
dropped and counted, never written.

Freeze compliance (eval/freeze.json — read it, do not work around it):
  * Only train template ids are used, and only with arg values that do
    not appear in any eval case. Arg pools are disjoint from the eval
    seen AND combo pools wherever a disjoint pool exists (times on a
    0.5s grid, fresh scale pairs, fresh speed factors, fresh convert
    triples). Where disjointness is impossible (extract_audio formats:
    combo pool is {aac, wav} and seen pool is {mp3, wav}), training
    uses only mp3 — the combo formats are never trained.
  * The reserved fixture long_60s_1080p is never used.
  * Mute / concat / clarify train templates are NOT reused: their
    eval prompts exhaust the train pool (every mute_tN and clarify_tN
    prompt appears in eval), so those tools get fresh paraphrases
    written for this dataset, each checked against the eval prompt set.
  * Every candidate (prompt, label) pair is checked against every eval
    case (prompt, label) and every (template_id, label) pair before
    it is accepted. A collision is dropped and counted.
  * The cut_out "remove this range" phrasings are fresh too: the
    held-out template cut_h3 ("without the part from ...") is never
    used, and neither is its wording.

Unit variety (guide §7.1.3): seconds, mm:ss, short "12s" units,
percentages and halves resolved against the fixture duration at
generation time, resolutions as WxH / named p-values / relative
("half size", "quarter size"), speeds as factors and percentages.
Labels are always canonical (absolute seconds, WxH, factor).

Contrast pairs: trim vs cut_out with the same times, generated as
pairs ("cut the first 10 seconds off" vs "trim to the first 10
seconds", "remove 5 to 15 seconds" vs "keep only 5 to 15 seconds").

Clarify (10-15%): ambiguous tool-only prompts with the EXACT minimal
missing list (the baseline_v3 failure was over-broad missing lists),
plus a small number of invalid-against-manifest prompts (end >
duration, out-of-range speed, odd scale dimensions) -> clarify.
Note the limitation: the model does not see the probe at inference,
so invalid-against-manifest clarify only works where the duration is
implied by the prompt's own numbers being impossible for a 30s clip;
kept to a small share for that reason.

General mix (~12%): hand-written short Q&A (source: the dataset
author), templated question/answer variants, plain-text assistant
replies (no tool call). Recorded in training/README.md.

Output: training/train_v1.jsonl, one {"messages": [...]} per line,
plus training/train_v1_meta.jsonl (line-aligned provenance: kind,
tool, fixture, template_id) for auditing and for the wrapper sample
check. The meta file is not training data.

Usage:
    python -m training.generate_dataset --out training --seed 20261121
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.generate_eval import (  # noqa: E402
    CLARIFY_TRAIN,  # noqa: F401  (imported to prove the pool is known; not reused)
    CONCAT_TRAIN,
    CONVERT_TRAIN,
    CUT_TRAIN,
    EXTRACT_TRAIN,
    FIXTURE_DURATIONS,
    MUTE_TRAIN,
    SCALE_TRAIN,
    SPEED_TRAIN,
    TRIM_TRAIN,
    _fmt_secs,
    _mmss,
    _silent_ok,
)
from minicut.build import default_output  # noqa: E402
from minicut.llm.prompts import SYSTEM_PROMPT  # noqa: E402
from minicut.probe import probe_file  # noqa: E402
from minicut.validate import validate  # noqa: E402

SEED = 20261121
FILE = "clip.mp4"
FIXTURES_DIR = ROOT / "factory" / "fixtures"
EVAL_DIR = ROOT / "eval"

TRAIN_FIXTURES = ["base_30s", "short_10s", "silent_30s", "portrait_30s"]
FIXTURE_DIMS = {
    "base_30s": (1280, 720),
    "short_10s": (1280, 720),
    "silent_30s": (1280, 720),
    "portrait_30s": (720, 1280),
}
RESERVED_FIXTURE = "long_60s_1080p"

# Target counts — total 3000. Weak spots from baseline_v2/v3 are
# over-represented: cut_out (inversion), speed / convert /
# extract_audio (arg names), clarify (exact missing lists).
COUNTS = {
    "trim": 360,
    "cut_out": 400,
    "concat": 270,
    "scale": 270,
    "speed": 300,
    "convert": 300,
    "extract_audio": 270,
    "mute": 120,
    "clarify": 350,
    "general": 360,
}
CLARIFY_AMBIGUOUS = 310
CLARIFY_INVALID = 40
CONTRAST_PAIRS = 40  # 40 trim + 40 cut_out, inside the trim/cut_out totals

# --- Arg pools: disjoint from eval seen AND combo pools -------------
# Eval pools (from freeze.json): trim seen {(2,8),(5,15),(12,27),(0,5),
# (20,28),(3,9)}, combo {(5,15),(8,22),(0,10),(15,25)}; cut seen
# {(10,20),(0,5),(5,10),(15,25)}, combo {(10,20),(0,10),(5,25),(2,8)};
# scale seen {(640,360),(1280,720),(854,480)}, combo {(854,480),
# (1920,1080),(720,1280)}; speed seen {2.0,0.5,1.5}, combo
# {0.25,4.0,1.5}; convert seen {(libx264,mp4,23),(libx265,mkv,23),
# (libx264,mkv,28)}, combo {(vp9,webm,23),(libx265,mkv,18),(vp9,mkv,23)};
# extract seen {mp3,wav}, combo {aac,wav}.
TRIM_FORBIDDEN = {(2.0, 8.0), (5.0, 15.0), (12.0, 27.0), (0.0, 5.0),
                  (20.0, 28.0), (3.0, 9.0), (8.0, 22.0), (0.0, 10.0), (15.0, 25.0)}
CUT_FORBIDDEN = {(10.0, 20.0), (0.0, 5.0), (5.0, 10.0), (15.0, 25.0),
                 (0.0, 10.0), (5.0, 25.0), (2.0, 8.0)}
SCALE_PAIRS = [(960, 540), (800, 450), (1024, 576), (320, 180), (480, 270),
               (1440, 810), (640, 480), (360, 640), (426, 240)]
SPEED_FACTORS = [0.6, 0.75, 0.8, 1.2, 1.25, 1.75, 2.25, 2.5, 3.0, 3.5]
CONVERT_CRFS = [18, 23, 28]
CONVERT_FORBIDDEN = {("libx264", "mp4", 23), ("libx265", "mkv", 23),
                     ("libx264", "mkv", 28), ("vp9", "webm", 23),
                     ("libx265", "mkv", 18), ("vp9", "mkv", 23)}
EXTRACT_FORMATS = ["mp3"]  # combo pool {aac, wav} is forbidden; wav is also in the seen pool
SPEED_PCT = {1.2: 20, 1.25: 25, 1.75: 75, 2.25: 125, 2.5: 150, 3.0: 200,
             3.5: 250, 0.6: 60, 0.75: 75, 0.8: 80}


def _canon(expected: dict):
    """Canonical form of a label for eval-collision checks (input and
    output excluded — the harness ignores them, and the wrapper derives
    output)."""
    tool = expected["tool"]
    if tool == "clarify":
        return ("clarify", expected["question"], tuple(sorted(expected.get("missing", []))))
    if tool == "concat":
        return ("concat", len(expected["inputs"]))
    args = {k: v for k, v in expected.items() if k not in ("tool", "input", "output")}
    return (tool, tuple(sorted((k, json.dumps(v)) for k, v in args.items())))


def _load_eval_collisions() -> tuple[set, set, set]:
    """(prompt, canon) pairs, (template_id, canon) pairs, and bare
    prompt strings from all four frozen eval splits."""
    prompt_canon, tid_canon, prompts = set(), set(), set()
    for path in sorted(EVAL_DIR.glob("eval_*.jsonl")):
        for line in path.read_text().splitlines():
            c = json.loads(line)
            canon = _canon(c["expected"])
            prompt_canon.add((c["prompt"], canon))
            tid_canon.add((c["template_id"], canon))
            prompts.add(c["prompt"])
    return prompt_canon, tid_canon, prompts


FILENAMES = ["clip.mp4", "video.mp4", "my_video.mp4", "input.mp4",
             "clip2.mp4", "footage.mp4", "a.mp4", "b.mp4", "test.mp4",
             "myclip.mp4", "vid.mp4", "movie.mp4", "recording.mp4",
             "clip_01.mp4", "final.mp4", "edit.mp4", "source.mp4",
             "cam.mp4", "take1.mp4", "scene.mp4", "clip3.mp4",
             "raw.mp4", "cut.mp4", "part1.mp4", "v1.mp4", "demo.mp4",
             "file.mp4", "media.mp4", "video1.mp4", "output1.mp4"]


def _label(tool: str, args: dict, input_file: str = FILE) -> dict:
    """Ground-truth JSON for one tool call, in few-shot key order:
    tool, input, args..., output (canonical, from build.default_output).
    Clarify has no input/output; concat has inputs but no input key
    (matching eval's _expected and the few-shot concat example)."""
    if tool == "clarify":
        return {"tool": "clarify", "question": args["question"],
                "missing": list(args["missing"])}
    call = {"tool": tool}
    if tool == "concat":
        call["inputs"] = list(args["inputs"])
    else:
        call["input"] = input_file
        call.update(args)
    call["output"] = default_output(call)
    return call


# --- Fresh paraphrases (not in eval) ----------------------------------
# cut_out "remove this range": deliberately avoids the held-out cut_h3
# wording ("without the part from ...") — these are different strings
# with the same semantics, which is exactly what the LoRA must learn
# (baseline_v3 inverted these). Concat / mute / clarify templates are
# all fresh because the eval prompts exhaust their train pools.
CUT_EXTRA = [
    ("train_cut_x1", lambda a, f: f"take out {_fmt_secs(a['start'])} to {_fmt_secs(a['end'])} seconds from {FILE}"),
    ("train_cut_x2", lambda a, f: f"delete seconds {_fmt_secs(a['start'])} through {_fmt_secs(a['end'])} from {FILE}"),
    ("train_cut_x3", lambda a, f: f"cut the section from {_fmt_secs(a['start'])} to {_fmt_secs(a['end'])} seconds out of {FILE}"),
    ("train_cut_x4", lambda a, f: f"remove the middle {_fmt_secs(a['start'])} to {_fmt_secs(a['end'])} seconds from {FILE}"),
]
TRIM_EXTRA = [
    ("train_trim_x1", lambda a, f: f"trim {FILE} from {_fmt_secs(a['start'])}s to {_fmt_secs(a['end'])}s"),
    ("train_trim_x2", lambda a, f: f"cut {FILE} to {_fmt_secs(a['start'])}-{_fmt_secs(a['end'])}"),
    ("train_trim_x3", lambda a, f: f"keep the {_fmt_secs(a['start'])} to {_fmt_secs(a['end'])} second part of {FILE}"),
]
CONCAT_NEW = [
    ("train_concat_x1", lambda a, f: f"put {'two' if len(a['inputs']) == 2 else 'three'} copies of {FILE} together"),
    ("train_concat_x2", lambda a, f: f"merge {FILE} with itself"),
    ("train_concat_x3", lambda a, f: f"append {FILE} to itself"),
    ("train_concat_x4", lambda a, f: f"loop {FILE} {'twice' if len(a['inputs']) == 2 else 'three times'}"),
    ("train_concat_x5", lambda a, f: f"make a video with {FILE} {'twice' if len(a['inputs']) == 2 else 'three times'} in a row"),
]
MUTE_NEW = [
    ("train_mute_x1", lambda a, f: f"silence the audio in {FILE}"),
    ("train_mute_x2", lambda a, f: f"make {FILE} silent"),
    ("train_mute_x3", lambda a, f: f"turn off the sound in {FILE}"),
    ("train_mute_x4", lambda a, f: f"remove all sound from {FILE}"),
    ("train_mute_x5", lambda a, f: f"drop the sound from {FILE}"),
    ("train_mute_x6", lambda a, f: f"make {FILE} quiet"),
    ("train_mute_x7", lambda a, f: f"strip the audio track from {FILE}"),
    ("train_mute_x8", lambda a, f: f"remove the audio track from {FILE}"),
    ("train_mute_x9", lambda a, f: f"kill the audio in {FILE}"),
    ("train_mute_x10", lambda a, f: f"drop all audio from {FILE}"),
]
EXTRACT_EXTRA = [
    ("train_extract_x1", lambda a, f: f"get the mp3 from {FILE}"),
    ("train_extract_x2", lambda a, f: f"rip the mp3 out of {FILE}"),
    ("train_extract_x3", lambda a, f: f"save {FILE}'s audio as mp3"),
    ("train_extract_x4", lambda a, f: f"export the soundtrack as mp3"),
    ("train_extract_x5", lambda a, f: f"make an mp3 from {FILE}"),
    ("train_extract_x6", lambda a, f: f"pull the mp3 from {FILE}"),
    ("train_extract_x7", lambda a, f: f"audio only from {FILE} as mp3"),
    ("train_extract_x8", lambda a, f: f"isolate the audio in {FILE} as mp3"),
    ("train_extract_x9", lambda a, f: f"grab the audio from {FILE} in mp3"),
    ("train_extract_x10", lambda a, f: f"turn {FILE} into an mp3"),
]
SCALE_EXTRA = [
    ("train_scale_x1", lambda a, f: f"scale {FILE} down to {a['width']}x{a['height']}" if (a['width'], a['height']) < FIXTURE_DIMS[f] else f"scale {FILE} up to {a['width']}x{a['height']}"),
    ("train_scale_x2", lambda a, f: f"make {FILE} 540p" if (a['width'], a['height']) == (960, 540) else f"make {FILE} 270p" if (a['width'], a['height']) == (480, 270) else f"resize {FILE} to {a['width']}x{a['height']}"),
]
SPEED_EXTRA = [
    ("train_speed_x1", lambda a, f: f"make {FILE} {SPEED_PCT[a['factor']]}% faster" if a['factor'] > 1 else f"slow {FILE} to {SPEED_PCT[a['factor']]}% speed"),
    ("train_speed_x2", lambda a, f: f"change {FILE} to play at {a['factor']}x"),
]
CONVERT_EXTRA = [
    ("train_convert_x1", lambda a, f: f"save {FILE} as a .{a['container']} file with {a['codec']}"),
    ("train_convert_x2", lambda a, f: f"change {FILE} to {a['container']} format using {a['codec']}"),
]

# Clarify: ambiguous tool-only prompts. Each question names the exact
# minimal missing list — over-broad missing lists were 4 of the 9
# baseline_v3 failures, so these are taught deliberately.
# (prompt, question, missing)
CLARIFY_AMBIG = [
    ("trim clip.mp4", "What start and end times should I trim to?", ["start", "end"]),
    ("cut clip.mp4", "What start and end times should I trim to?", ["start", "end"]),
    ("shorten clip.mp4", "What start and end times should I trim to?", ["start", "end"]),
    ("trim the clip", "What start and end times should I trim to?", ["start", "end"]),
    ("edit clip.mp4 down", "What start and end times should I trim to?", ["start", "end"]),
    ("cut the bad part out of clip.mp4", "Which start and end times should I cut out?", ["start", "end"]),
    ("remove the boring bit from clip.mp4", "Which start and end times should I cut out?", ["start", "end"]),
    ("cut something out of clip.mp4", "Which start and end times should I cut out?", ["start", "end"]),
    ("make clip.mp4 smaller", "What width and height should I scale to?", ["width", "height"]),
    ("resize clip.mp4", "What width and height should I scale to?", ["width", "height"]),
    ("change the size of clip.mp4", "What width and height should I scale to?", ["width", "height"]),
    ("make clip.mp4 bigger", "What width and height should I scale to?", ["width", "height"]),
    ("speed up clip.mp4", "What speed factor should I use?", ["factor"]),
    ("slow clip.mp4 down", "What speed factor should I use?", ["factor"]),
    ("change the speed of clip.mp4", "What speed factor should I use?", ["factor"]),
    ("convert clip.mp4", "Which codec and container should I convert to?", ["codec", "container"]),
    ("change the format of clip.mp4", "Which codec and container should I convert to?", ["codec", "container"]),
    ("re-encode clip.mp4", "Which codec and container should I convert to?", ["codec", "container"]),
    ("get the audio from clip.mp4", "What audio format should I extract to?", ["format"]),
    ("pull the audio out of clip.mp4", "What audio format should I extract to?", ["format"]),
    ("save the sound from clip.mp4", "What audio format should I extract to?", ["format"]),
    ("join the clips", "Which files should I join, and in what order?", ["inputs"]),
    ("put the videos together", "Which files should I join, and in what order?", ["inputs"]),
    ("combine clip.mp4 with another video", "Which files should I join, and in what order?", ["inputs"]),
]

# General mix: hand-written short Q&A (dataset author), question and
# answer variants. Plain-text assistant replies — no tool call.
# (questions, answers)
GENERAL_BASES = [
    (["hi", "hello", "hey", "hey minicut", "hello there"],
     ["Hi! Tell me an edit, like 'trim clip.mp4 from 12 to 45 seconds'.",
      "Hello! What should I do with your video?"]),
    (["what can you do?", "what do you do?", "what are you for?", "what edits can you make?", "what can you help with?"],
     ["I edit videos: trim, cut out sections, join clips, resize, change speed, convert formats, extract audio, and mute.",
      "I take plain-English video edits and turn them into ffmpeg commands — trim, cut, concat, scale, speed, convert, extract audio, mute."]),
    (["who are you?", "what are you?", "are you an ai?", "what is minicut?"],
     ["I'm MiniCut, a small local video editor. You describe the edit, I do it with ffmpeg.",
      "MiniCut — a tiny specialist that turns video edit requests into ffmpeg edits, running locally."]),
    (["thanks", "thank you", "thanks!", "cheers", "nice, thanks"],
     ["You're welcome!", "Any time."]),
    (["bye", "goodbye", "see you later", "that's all for now"],
     ["Bye!", "See you next edit."]),
    (["help", "help me", "how do i use this?", "how does this work?", "i need help"],
     ["Just describe the edit, like 'trim clip.mp4 from 12 to 45 seconds' or 'make clip.mp4 twice as fast'. If something's missing I'll ask.",
      "Tell me what to do with the video in plain words. I ask when I need times, sizes, or formats."]),
    (["how do i trim a video?", "how can i cut a video?", "how do i cut out part of a video?"],
     ["Say which part to keep or cut, with times — for example 'trim clip.mp4 from 12 to 45 seconds'.",
      "Give me the times, like 'cut the first 10 seconds off clip.mp4'."]),
    (["how do i make a video faster?", "how do i slow a video down?", "can you change the speed?"],
     ["Say how much faster or slower, like 'make clip.mp4 twice as fast' or 'slow clip.mp4 to 0.5x'."]),
    (["how do i extract audio?", "can you get the audio out?", "how do i save just the sound?"],
     ["Say which format — for example 'extract the audio from clip.mp4 as mp3'."]),
    (["how do i convert a video?", "how do i change a video format?", "can you convert mp4 to mkv?"],
     ["Say the codec and container, like 'convert clip.mp4 to mkv with libx264'."]),
    (["can you crop videos?", "can you rotate a video?", "can you add fades?", "can you add text to a video?"],
     ["Not yet — crop, rotate, and fades are planned for v2. Right now I do trim, cut, concat, scale, speed, convert, extract audio, and mute.",
      "Those aren't in v1. I do trim, cut out, join, resize, speed, convert, extract audio, and mute."]),
    (["what formats can you convert to?", "what video formats do you support?", "which containers work?"],
     ["mp4, mkv, and webm, with libx264, libx265, or vp9 video.",
      "I can output mp4, mkv, or webm."]),
    (["what audio formats?", "which audio formats can you extract?", "can you extract wav?"],
     ["mp3, wav, and aac.",
      "mp3, wav, or aac."]),
    (["is my video private?", "do you upload my video anywhere?", "where does my video go?"],
     ["Everything runs locally — your video never leaves your machine.",
      "It's all local. No uploads, no cloud."]),
    (["what is ffmpeg?", "what does ffmpeg do?", "why ffmpeg?"],
     ["ffmpeg is the command-line tool that does the actual video work. I just translate your words into its commands.",
      "ffmpeg is the video tool under the hood — I turn your request into the right ffmpeg command."]),
    (["what is a codec?", "what's the difference between libx264 and libx265?", "what is crf?"],
     ["A codec is how the video is compressed. libx264 is the safe default; libx265 is smaller but slower; crf sets the quality, lower is better.",
      "The codec compresses the video. libx264 is the compatible default, libx265 compresses more, and crf controls quality."]),
    (["how long does an edit take?", "is this fast?", "will this take a while?"],
     ["Usually seconds for a short clip — it depends on the length and whether it needs re-encoding.",
      "Short clips take a few seconds."]),
    (["can you join two videos?", "can you combine clips?", "can you merge videos?"],
     ["Yes — say which ones, like 'join clip.mp4 to itself'.",
      "Yes, I can concatenate clips in order."]),
    (["can you make a video silent?", "how do i remove audio?", "can you mute a video?"],
     ["Yes — just say 'mute clip.mp4'.",
      "Say 'mute clip.mp4' and I'll drop the audio track."]),
    (["can you resize a video?", "how do i make a video smaller?", "can you change the resolution?"],
     ["Say the size, like 'make clip.mp4 1280x720' or 'shrink clip.mp4 to half size'."]),
    (["what happens if i don't give a time?", "what if i forget the times?", "what if my request is unclear?"],
     ["I'll ask — I clarify instead of guessing.",
      "I ask you for the missing piece rather than guessing."]),
    (["can you edit more than one video at once?", "can you do two edits?", "can you trim and speed up at the same time?"],
     ["One edit per request for now — do them one at a time.",
      "One edit at a time in v1."]),
    (["is this free?", "does this cost anything?", "do i need an account?"],
     ["It's local and free — no account, no subscription.",
      "Free, local, no account needed."]),
    (["who made you?", "who built minicut?", "who is behind this?"],
     ["MiniCut is a small open project built around MiniCPM5-2B and ffmpeg.",
      "It's a hobby-scale project: a fine-tuned MiniCPM5-2B driving ffmpeg."]),
    (["what is a lora?", "what does fine-tuning mean?", "how were you trained?"],
     ["A LoRA is a small adapter trained on top of the base model, teaching it MiniCut's exact tool format.",
      "I was fine-tuned with a small LoRA adapter on examples of video-edit requests and their tool calls."]),
    (["good morning", "good afternoon", "good evening"],
     ["Good morning! What are we editing today?",
      "Hello! What edit do you need?"]),
    (["ok", "okay", "got it", "sounds good", "cool", "nice"],
     ["Great — what's next?",
      "Sounds good."]),
    (["that was wrong", "that's not what i wanted", "you got it wrong"],
     ["Sorry about that — tell me what it should have been and I'll redo it.",
      "My mistake. Describe the edit again and I'll fix it."]),
    (["can you undo that?", "undo", "can you revert?", "go back"],
     ["I never modify the original — every edit writes a new file, so the original is still there.",
      "No need to undo: the original file is untouched, each edit makes a new one."]),
    (["where is my file?", "where did the output go?", "what was the output called?"],
     ["Each edit writes a new file next to the original, named after the edit — for example clip_trim.mp4.",
      "Outputs land next to the input, named after the edit that made them."]),
    (["can you handle long videos?", "what about a 2 hour video?", "is there a size limit?"],
     ["Length isn't a problem in principle, but long videos take longer to re-encode.",
      "It works on long videos too — just slower, since re-encoding takes time."]),
    (["do you work offline?", "do you need internet?", "can i use this without wifi?"],
     ["Yes — the model and ffmpeg both run locally, no internet needed.",
      "Fully offline once it's set up."]),
    (["what is 720p?", "what does 1080p mean?", "what is resolution?"],
     ["720p means 1280 by 720 pixels. 1080p is 1920 by 1080.",
      "Resolution is the frame size in pixels — 720p is 1280x720."]),
    (["why did my edit fail?", "the edit didn't work", "it gave me an error"],
     ["Usually a missing detail or a time outside the video. Tell me the exact request and I'll check it.",
      "Most failures are a time beyond the video's length or a missing format. Paste the request and I'll look."]),
    (["can you make a gif?", "can you make a thumbnail?", "can you take a screenshot of a video?"],
     ["Not in v1 — those are possible later, but the current tools are trim, cut, concat, scale, speed, convert, extract, and mute.",
      "Not yet."]),
    (["how accurate are you?", "do you make mistakes?", "are you reliable?"],
     ["I get most straightforward edits right, and I clarify instead of guessing when something's unclear. Check the output — the original is never touched.",
      "Pretty good on clear requests; I ask when I'm unsure."]),
    (["i love this", "this is great", "this is cool", "awesome"],
     ["Glad to hear it!",
      "Thank you!"]),
    (["what's new?", "any updates?", "what changed?"],
     ["The latest version adds edit history in the file's metadata, so you can see what was done to a clip.",
      "Recent work: few-shot examples for the tricky tools, and edit history stored in metadata."]),
    (["can you edit audio?", "can you cut audio?", "can you change audio volume?"],
     ["I can extract audio or drop it, but volume and audio cutting aren't in v1.",
      "Extract and mute only, for now."]),
    (["what's the best format for youtube?", "what format should i use?", "which codec is best?"],
     ["libx264 in mp4 is the safest bet for compatibility.",
      "For sharing, libx264 mp4 is the most compatible."]),
    (["what's the difference between trim and cut?", "trim vs cut out — what's the difference?", "how is trim different from cut out?"],
     ["Trim keeps only the part you name; cut out removes the part you name and keeps the rest.",
      "Trim = keep this range. Cut out = remove this range."]),
    (["can you speed up only part of a video?", "can i change speed for just one section?"],
     ["Not in one step — trim the section out first, then speed it up.",
      "One edit at a time: cut the section, then change its speed."]),
    (["can i use a video from my phone?", "does it work with phone videos?", "can i use an iphone video?"],
     ["Yes — if it's an mp4 or mov file, it should work. Convert it first if it's an odd format.",
      "Phone videos usually work fine."]),
    (["do you support 4k?", "can you handle 4k video?", "what about high resolution?"],
     ["Yes, though 4k takes longer to re-encode.",
      "4k works, just slower."]),
    (["can you add music?", "can you add a soundtrack?", "can you add background music?"],
     ["Not in v1 — I can extract or remove audio, but not mix new audio in.",
      "No audio mixing yet."]),
    (["can you make a video shorter?", "how do i shorten a video?", "make this video shorter"],
     ["Say what to keep or cut, with times — like 'trim clip.mp4 to the first 30 seconds'."]),
    (["what is the output called?", "what will the new file be named?"],
     ["It's named after the edit — for example clip_trim.mp4 — next to the original.",
      "The output is named for the edit, like clip_speed.mp4."]),
    (["can you process multiple files?", "can you do a batch of videos?", "can you edit a folder of videos?"],
     ["One file per request for now.",
      "Not yet — one video at a time."]),
    (["how do i install this?", "how do i set it up?", "how do i get started?"],
     ["You need the model, llama.cpp, and ffmpeg — the README has the setup steps.",
      "Install ffmpeg and llama.cpp, download the model, and you're set."]),
    (["is there a gui?", "is there an app?", "do you have a website?"],
     ["No — it's a command-line tool for now.",
      "No interface yet, just chat and the command line."]),
    (["can i run this on my laptop?", "what hardware do i need?", "do i need a gpu?"],
     ["A GPU helps a lot, but it runs on CPU too — just slower.",
      "It runs on CPU, faster on a GPU."]),
    (["how do i report a bug?", "where do i report problems?", "i found a bug"],
     ["Tell me what you asked for and what happened, and I'll look into it.",
      "Describe the request and the result — that helps me fix it."]),
    (["can you translate?", "do you speak other languages?", "can you work in spanish?"],
     ["I understand English best; other languages may work but aren't tested.",
      "English is the safe bet for now."]),
    (["what is the minicut_chain?", "what is the metadata in my file?", "how do i see what was done to a video?"],
     ["Each edit writes a minicut_chain tag into the file's metadata listing the edits, like trim:12-45;speed:2.0.",
      "The edit history lives in the file's metadata tag — read it with ffprobe."]),
    (["can you flip a video?", "can you mirror a video?", "can you reverse a video?"],
     ["Not in v1.",
      "Flipping and reversing aren't in the current toolset."]),
    (["can you compress a video?", "how do i make a video file smaller?", "can you reduce file size?"],
     ["Convert with a higher crf or a smaller resolution, like 'convert clip.mp4 to mkv with libx265' or 'make clip.mp4 640x480'.",
      "Re-encode with libx265 or scale down — that shrinks the file."]),
    (["what does crf 23 mean?", "is lower crf better?", "what crf should i use?"],
     ["crf 23 is the default. Lower means better quality and bigger files; higher means smaller files.",
      "23 is a good default. Go lower for quality, higher to save space."]),
    (["can you split a video in two?", "how do i split a video?", "can you cut a video into parts?"],
     ["Trim the first part, then trim the second part — two edits.",
      "Use two trims: one for each half."]),
    (["do you remember my edits?", "do you keep history?", "can you see what i did before?"],
     ["The history is in each file's metadata, not in my memory — I read it from the file when you edit it again.",
      "Each file carries its own edit history in metadata."]),
]


# --- Arg sampling -------------------------------------------------------
# Filename variation: prompts mostly say clip.mp4 (the eval convention),
# but low-entropy tools (concat / mute / extract / clarify) would
# otherwise have almost no distinct (prompt, label) pairs at all —
# concat has 5 templates x 2 counts = 10. Varying the filename teaches
# the model to copy the filename from the prompt into the input field,
# which is what real users need, and multiplies the unique pairs.
# Arg-bearing tools keep clip.mp4 half the time.
def _pick_fixture(rng: random.Random, tool: str) -> str:
    """Weighted fixture pick; silent_30s only for tools that make sense
    there (no extract_audio / mute — no audio to extract or drop)."""
    while True:
        f = rng.choices(TRAIN_FIXTURES, weights=[50, 15, 15, 20])[0]
        if f == "silent_30s" and not _silent_ok(tool):
            continue
        return f


def _sample_times(rng: random.Random, dur: float, kind: str) -> tuple[float, float]:
    """Sample (start, end) on a 0.5s grid, disjoint from the eval pools.
    kind 'trim' needs >= 2s of output; 'cut_out' needs >= 1s left.
    Sometimes force a percentage / half pair so those render styles
    actually occur instead of always falling back to train templates."""
    forbidden = TRIM_FORBIDDEN if kind == "trim" else CUT_FORBIDDEN
    roll = rng.random()
    if roll < 0.10:
        pct = rng.choice([10, 20, 25, 50])
        where = rng.choice(["first", "last"])
        s, e = _pct_times(dur, pct, where)
        if (s, e) not in forbidden:
            if kind == "trim" and e - s >= 2.0:
                return s, e
            if kind == "cut_out" and dur - (e - s) >= 1.0:
                return s, e
    if roll < 0.14:
        s, e = (0.0, dur / 2) if rng.random() < 0.5 else (dur / 2, dur)
        if (s, e) not in forbidden:
            if kind == "trim" and e - s >= 2.0:
                return s, e
            if kind == "cut_out" and dur - (e - s) >= 1.0:
                return s, e
    for _ in range(200):
        s2 = rng.randrange(0, int(dur * 2) - 1)  # leave room for e > s
        e2 = rng.randrange(s2 + 2, int(dur * 2) + 1)
        s, e = s2 / 2, e2 / 2
        if (s, e) in forbidden:
            continue
        if kind == "trim" and e - s < 2.0:
            continue
        if kind == "cut_out" and dur - (e - s) < 1.0:
            continue
        return s, e
    raise RuntimeError(f"no valid {kind} times for duration {dur}")


def _sample_convert(rng: random.Random) -> dict:
    for _ in range(200):
        codec = rng.choice(["libx264", "libx265", "vp9"])
        container = rng.choice(["mp4", "mkv", "webm"])
        if container == "webm" and codec != "vp9":
            continue  # WebM only allows VP8/VP9/AV1 video
        crf = rng.choice(CONVERT_CRFS)
        if (codec, container, crf) in CONVERT_FORBIDDEN:
            continue
        return {"codec": codec, "container": container, "crf": crf}
    raise RuntimeError("no valid convert combo")


def _pct_times(dur: float, pct: int, where: str) -> tuple[float, float]:
    """Resolve a percentage against the fixture duration (clean halves
    only: 10/20/25/50% of 10s or 30s are all on the 0.5s grid)."""
    span = dur * pct / 100
    return (0.0, span) if where == "first" else (dur - span, dur)


def _render_times(tool: str, rng: random.Random, args: dict, fixture: str):
    """Pick a render style for trim/cut_out and return (template_id,
    prompt). Styles: train templates, mm:ss, short units, percentages,
    halves, extra remove/keep phrasings."""
    dur = FIXTURE_DURATIONS[fixture]
    s, e = args["start"], args["end"]
    train = TRIM_TRAIN if tool == "trim" else CUT_TRAIN
    style = rng.choices(
        ["train", "mmss", "short", "pct", "half", "extra"],
        weights=[50, 12, 10, 10, 4, 14])[0]
    if style == "train":
        tid, render = rng.choice(train)
        return tid, render(args, fixture)
    if style == "mmss":
        if tool == "trim":
            return "train_trim_x_mmss", f"trim {FILE} from {_mmss(s)} to {_mmss(e)}"
        return "train_cut_x_mmss", f"remove {_mmss(s)} to {_mmss(e)} from {FILE}"
    if style == "short":
        if tool == "trim":
            return "train_trim_x1", f"trim {FILE} from {_fmt_secs(s)}s to {_fmt_secs(e)}s"
        return "train_cut_x1", f"take out {_fmt_secs(s)} to {_fmt_secs(e)} seconds from {FILE}"
    if style == "pct":
        for pct in (10, 20, 25, 50):
            first = _pct_times(dur, pct, "first")
            last = _pct_times(dur, pct, "last")
            if (s, e) == first or (s, e) == last:
                where = "first" if (s, e) == first else "last"
                if tool == "trim":
                    return "train_trim_x_pct", f"trim the {where} {pct}% of {FILE}"
                return "train_cut_x_pct", f"cut the {where} {pct}% off {FILE}"
        tid, render = rng.choice(train)
        return tid, render(args, fixture)
    if style == "half":
        if (s, e) == (0.0, dur / 2):
            return ("train_trim_x_half", f"trim the first half of {FILE}") if tool == "trim" \
                else ("train_cut_x_half", f"cut the first half off {FILE}")
        if (s, e) == (dur / 2, dur):
            return ("train_trim_x_half", f"trim the last half of {FILE}") if tool == "trim" \
                else ("train_cut_x_half", f"cut the last half off {FILE}")
        tid, render = rng.choice(train)
        return tid, render(args, fixture)
    # extra
    pool = TRIM_EXTRA if tool == "trim" else CUT_EXTRA
    tid, render = rng.choice(pool)
    return tid, render(args, fixture)


def _render_scale(rng: random.Random, args: dict, fixture: str):
    style = rng.choices(["train", "dir", "named", "relative"],
                        weights=[55, 20, 10, 15])[0]
    if style == "train":
        tid, render = rng.choice(SCALE_TRAIN)
        return tid, render(args, fixture)
    if style == "dir":
        tid, render = rng.choice(SCALE_EXTRA[:1])
        return tid, render(args, fixture)
    if style == "named":
        if (args["width"], args["height"]) == (960, 540):
            return "train_scale_x2", f"make {FILE} 540p"
        if (args["width"], args["height"]) == (480, 270):
            return "train_scale_x2", f"make {FILE} 270p"
        tid, render = rng.choice(SCALE_TRAIN)
        return tid, render(args, fixture)
    # relative: half / quarter of the fixture size
    fw, fh = FIXTURE_DIMS[fixture]
    if (args["width"], args["height"]) == (fw // 2, fh // 2):
        return "train_scale_x_rel", f"shrink {FILE} to half size"
    if (args["width"], args["height"]) == (fw // 4, fh // 4):
        return "train_scale_x_rel", f"shrink {FILE} to a quarter of its size"
    tid, render = rng.choice(SCALE_TRAIN)
    return tid, render(args, fixture)


def _render_speed(rng: random.Random, args: dict, fixture: str):
    style = rng.choices(["train", "pct", "extra"], weights=[60, 25, 15])[0]
    if style == "train":
        tid, render = rng.choice(SPEED_TRAIN)
        return tid, render(args, fixture)
    if style == "pct":
        tid, render = rng.choice(SPEED_EXTRA[:1])
        return tid, render(args, fixture)
    tid, render = rng.choice(SPEED_EXTRA[1:])
    return tid, render(args, fixture)


# --- Generation ---------------------------------------------------------
def _sample_args(rng: random.Random, tool: str, fixture: str, fname: str = FILE) -> dict:
    dur = FIXTURE_DURATIONS[fixture]
    if tool == "trim":
        s, e = _sample_times(rng, dur, "trim")
        return {"start": s, "end": e}
    if tool == "cut_out":
        s, e = _sample_times(rng, dur, "cut_out")
        return {"start": s, "end": e}
    if tool == "concat":
        return {"inputs": [fname] * rng.choice([2, 2, 3])}
    if tool == "scale":
        # Relative sizes: quarter of the fixture, or half on portrait
        # (360x640). Never (640,360): that is the base fixture's half
        # size and it is in the eval pool.
        if rng.random() < 0.12:
            fw, fh = FIXTURE_DIMS[fixture]
            if fixture == "portrait_30s" and rng.random() < 0.5:
                return {"width": 360, "height": 640}
            return {"width": fw // 4, "height": fh // 4}
        w, h = rng.choice(SCALE_PAIRS)
        return {"width": w, "height": h}
    if tool == "speed":
        return {"factor": rng.choice(SPEED_FACTORS)}
    if tool == "convert":
        return _sample_convert(rng)
    if tool == "extract_audio":
        return {"format": "mp3"}
    if tool == "mute":
        return {}
    raise ValueError(tool)


def _render(rng: random.Random, tool: str, args: dict, fixture: str,
            fname: str = FILE):
    """(template_id, prompt) for a tool + args + fixture + filename.
    Every renderer is written against clip.mp4; the filename is swapped
    in here so the label's input field matches the prompt."""
    if tool in ("trim", "cut_out"):
        tid, prompt = _render_times(tool, rng, args, fixture)
    elif tool == "concat":
        tid, render = rng.choice(CONCAT_NEW)
        tid, prompt = tid, render(args, fixture)
    elif tool == "scale":
        tid, prompt = _render_scale(rng, args, fixture)
    elif tool == "speed":
        tid, prompt = _render_speed(rng, args, fixture)
    elif tool == "convert":
        style = rng.choices(["train", "extra"], weights=[70, 30])[0]
        pool = CONVERT_TRAIN if style == "train" else CONVERT_EXTRA
        tid, render = rng.choice(pool)
        tid, prompt = tid, render(args, fixture)
    elif tool == "extract_audio":
        tid, render = rng.choice(EXTRACT_TRAIN + EXTRACT_EXTRA)
        tid, prompt = tid, render(args, fixture)
    elif tool == "mute":
        tid, render = rng.choice(MUTE_NEW)
        tid, prompt = tid, render(args, fixture)
    else:
        raise ValueError(tool)
    return tid, prompt.replace(FILE, fname)


def _contrast_examples(rng: random.Random, n_pairs: int):
    """Trim vs cut_out with the SAME times — the pair the model must
    learn to tell apart. Returns (tool, args, fixture, tid, prompt,
    fname) tuples; the filename varies so the (0, X) pairs are not
    limited to three distinct prompts."""
    out = []
    for _ in range(n_pairs):
        fixture = rng.choice(["base_30s", "base_30s", "portrait_30s", "silent_30s"])
        dur = FIXTURE_DURATIONS[fixture]
        fname = rng.choice(FILENAMES)
        if rng.random() < 0.5:
            # (0, 5) and (0, 10) are eval-pool pairs — excluded here so
            # contrast times stay disjoint from eval arg values.
            x = rng.choice([6.0, 7.0, 12.0, 15.0, 17.0])
            s, e = 0.0, x
            trim_prompt = f"trim {fname} to the first {_fmt_secs(x)} seconds"
            cut_prompt = f"cut the first {_fmt_secs(x)} seconds off {fname}"
            trim_tid, cut_tid = "trim_t5", "cut_t1"
        else:
            s, e = _sample_times(rng, dur, "trim")
            trim_prompt = f"keep only {_fmt_secs(s)} to {_fmt_secs(e)} seconds of {fname}"
            cut_prompt = f"remove {_fmt_secs(s)} to {_fmt_secs(e)} seconds from {fname}"
            trim_tid, cut_tid = "trim_t3", "cut_t2"
        out.append(("trim", {"start": s, "end": e}, fixture, trim_tid, trim_prompt, fname))
        out.append(("cut_out", {"start": s, "end": e}, fixture, cut_tid, cut_prompt, fname))
    return out


def _invalid_clarify(rng: random.Random):
    """Invalid-against-manifest prompts -> clarify. The model cannot
    see the probe, so these only teach 'impossible numbers -> ask'."""
    kind = rng.choice(["trim_end", "cut_end", "speed", "scale_odd"])
    if kind == "trim_end":
        s = rng.choice([20.0, 25.0, 28.0])
        e = s + rng.choice([5.0, 10.0, 15.0, 17.0])
        prompt = f"trim {FILE} from {_fmt_secs(s)} to {_fmt_secs(e)} seconds"
        return (prompt, "The clip is only 30 seconds long — what start and end times should I trim to?",
                ["start", "end"], "base_30s")
    if kind == "cut_end":
        s = rng.choice([25.0, 28.0])
        e = s + rng.choice([5.0, 10.0])
        prompt = f"remove {_fmt_secs(s)} to {_fmt_secs(e)} seconds from {FILE}"
        return (prompt, "The clip is only 30 seconds long — which start and end times should I cut out?",
                ["start", "end"], "base_30s")
    if kind == "speed":
        f = rng.choice([5.0, 9.0, 0.1])
        prompt = f"make {FILE} {f} times as fast" if f > 1 else f"slow {FILE} to {f}x"
        return (prompt, "Speed must be between 0.25x and 4x — what speed factor should I use?",
                ["factor"], "base_30s")
    w = rng.choice([641, 643, 1281])
    h = 360
    prompt = f"make {FILE} {w}x{h}"
    return (prompt, "Width and height must be even numbers — what size should I scale to?",
            ["width", "height"], "base_30s")


def generate(rng: random.Random):
    """Build all examples. Returns (examples, meta, stats) where each
    example is {"messages": [...]}, meta is the provenance record, and
    stats counts drops and collisions."""
    prompt_canon, tid_canon, eval_prompts = _load_eval_collisions()
    probes = {f: probe_file(FIXTURES_DIR / f / "input.mp4") for f in TRAIN_FIXTURES}
    seen: set[tuple] = set()
    examples: list[dict] = []
    meta: list[dict] = []
    stats = {"validate_failed": 0, "eval_collision": 0, "duplicate": 0,
             "validate_failures": []}

    def add(tool: str, args: dict, fixture: str | None, template_id: str,
            prompt: str, kind: str, input_file: str = FILE) -> bool:
        label = _label(tool, args, input_file) if tool != "general" else None
        if tool == "general":
            assistant = args["answer"]
            key = ("general", prompt, assistant)
            if key in seen:
                stats["duplicate"] += 1
                return False
            if prompt in eval_prompts:
                stats["eval_collision"] += 1
                return False
            seen.add(key)
            examples.append({"messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": assistant}]})
            meta.append({"kind": "general", "tool": None, "fixture": None,
                         "template_id": template_id})
            return True
        # Validate the label against the fixture probe (clarify: no probe).
        probe = probes[fixture] if fixture else None
        errs = validate(label, probe, Path("."))
        if errs:
            stats["validate_failed"] += 1
            stats["validate_failures"].append((prompt, label, errs))
            return False
        canon = _canon(label)
        if (prompt, canon) in prompt_canon or (template_id, canon) in tid_canon:
            stats["eval_collision"] += 1
            return False
        if prompt in eval_prompts:
            stats["eval_collision"] += 1
            return False
        key = (prompt, json.dumps(label, sort_keys=True))
        if key in seen:
            stats["duplicate"] += 1
            return False
        seen.add(key)
        examples.append({"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": json.dumps(label)}]})
        meta.append({"kind": "clarify" if tool == "clarify" else "tool",
                     "tool": tool, "fixture": fixture, "template_id": template_id})
        return True

    def sample_tool(tool: str, count: int, guard_extra: int = 800):
        made, guard = 0, 0
        while made < count and guard < count + guard_extra:
            guard += 1
            fixture = _pick_fixture(rng, tool)
            # Low-entropy tools need filename variation for uniqueness;
            # arg-bearing tools keep clip.mp4 half the time (eval style).
            fname = rng.choice(FILENAMES) if tool in ("concat", "mute", "extract_audio") \
                else ("clip.mp4" if rng.random() < 0.5 else rng.choice(FILENAMES))
            args = _sample_args(rng, tool, fixture, fname)
            tid, prompt = _render(rng, tool, args, fixture, fname)
            if add(tool, args, fixture, tid, prompt, "tool", input_file=fname):
                made += 1
        if made < count:
            raise RuntimeError(f"only made {made}/{count} {tool} examples")

    # Contrast pairs first (inside the trim/cut_out totals).
    contrast = _contrast_examples(rng, CONTRAST_PAIRS)
    n_trim_contrast = n_cut_contrast = 0
    for tool, args, fixture, tid, prompt, fname in contrast:
        if add(tool, args, fixture, tid, prompt, "tool", input_file=fname):
            if tool == "trim":
                n_trim_contrast += 1
            else:
                n_cut_contrast += 1
    stats["contrast_trim"] = n_trim_contrast
    stats["contrast_cut_out"] = n_cut_contrast

    sample_tool("trim", COUNTS["trim"] - n_trim_contrast)
    sample_tool("cut_out", COUNTS["cut_out"] - n_cut_contrast)
    for tool in ("concat", "scale", "speed", "convert", "extract_audio", "mute"):
        sample_tool(tool, COUNTS[tool])

    # Clarify: ambiguous prompts, then invalid-against-manifest.
    # Filenames vary so the 24 ambiguous prompts yield 310 unique
    # (prompt, label) pairs instead of 24.
    made, guard = 0, 0
    while made < CLARIFY_AMBIGUOUS and guard < CLARIFY_AMBIGUOUS + 800:
        guard += 1
        prompt, question, missing = rng.choice(CLARIFY_AMBIG)
        prompt = prompt.replace(FILE, rng.choice(FILENAMES))
        if add("clarify", {"question": question, "missing": missing},
               None, "train_clarify_x", prompt, "clarify"):
            made += 1
    if made < CLARIFY_AMBIGUOUS:
        raise RuntimeError(f"only made {made}/{CLARIFY_AMBIGUOUS} ambiguous clarify")
    made, guard = 0, 0
    while made < CLARIFY_INVALID and guard < CLARIFY_INVALID + 200:
        guard += 1
        prompt, question, missing, fixture = _invalid_clarify(rng)
        prompt = prompt.replace(FILE, rng.choice(FILENAMES))
        if add("clarify", {"question": question, "missing": missing},
               fixture, "train_clarify_invalid", prompt, "clarify"):
            made += 1
    if made < CLARIFY_INVALID:
        raise RuntimeError(f"only made {made}/{CLARIFY_INVALID} invalid clarify")

    # General mix.
    pool = [(q, a, f"general_g{i:02d}")
            for i, (qs, ans) in enumerate(GENERAL_BASES)
            for q in qs for a in ans]
    rng.shuffle(pool)
    made = 0
    for q, a, tid in pool:
        if made >= COUNTS["general"]:
            break
        if add("general", {"answer": a}, None, tid, q, "general"):
            made += 1
    if made < COUNTS["general"]:
        raise RuntimeError(f"only made {made}/{COUNTS['general']} general examples")

    # Shuffle examples and meta together with one permutation so the
    # two files stay line-aligned.
    order = list(range(len(examples)))
    rng.shuffle(order)
    examples = [examples[i] for i in order]
    meta = [meta[i] for i in order]
    return examples, meta, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="training")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    out = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    examples, meta, stats = generate(rng)
    assert len(examples) == len(meta) == sum(COUNTS.values()), \
        f"{len(examples)} examples, expected {sum(COUNTS.values())}"
    data_path = out / "train_v1.jsonl"
    meta_path = out / "train_v1_meta.jsonl"
    data_path.write_text("".join(json.dumps(e) + "\n" for e in examples))
    meta_path.write_text("".join(json.dumps(m) + "\n" for m in meta))
    counts: dict[str, int] = {}
    for m in meta:
        counts[m["kind"] if m["kind"] == "general" else m["tool"]] = \
            counts.get(m["kind"] if m["kind"] == "general" else m["tool"], 0) + 1
    print(f"wrote {len(examples)} examples -> {data_path}")
    print(f"meta -> {meta_path}")
    print("counts:", json.dumps(counts, sort_keys=True))
    print(f"validate_failed (dropped): {stats['validate_failed']}")
    for prompt, label, errs in stats["validate_failures"][:10]:
        print(f"  FAIL {prompt!r} {label} -> {errs}")
    print(f"eval collisions (dropped): {stats['eval_collision']}")
    print(f"duplicates (dropped): {stats['duplicate']}")
    print(f"contrast pairs: {stats['contrast_trim']} trim + {stats['contrast_cut_out']} cut_out")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
