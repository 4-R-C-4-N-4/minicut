# Plan: the MiniCut phrasebook (a user grammar for requests)

Status: proposal, not yet implemented. Ivy's direction (2026-11-25): don't deliberately confuse the model; "cut the first N seconds of X" means the same as "… off X"; define a user grammar and show it in `minicut --help`.

## Why

Phase 3 treated some wordings as ambiguous and trained the model to ask about them. "Cut the first N seconds **of** X" became a clarify row while "… **off** X" stayed a cut. The model can't learn a one-letter difference in 100 updates. Every weak verified pilot turned the dev case "cut the first 10 seconds off clip.mp4" (4 dev cases) into a question (`eval/PILOT_BOARD.md`, v2 and v3 sweeps). Pairing the two wordings made it no better. The ambiguity came from our own table, not from users.

A phrasebook replaces "is this wording ambiguous?" judgments with one written rule, which we:
- teach consistently;
- check every template against;
- show to users.

Name: *phrasebook*, not "grammar", because `minicut grammar` already prints the GBNF output grammar.

## The rule (keep vs remove)

1. **A phrase that names the result keeps it** (trim).
   - Markers: *keep*, *only*, *just*, *isolate*, *give me*, and *to* / *down to* after the verb.
   - Examples: "trim X to the first 10 seconds", "cut X down to 5–15 seconds", "cut X to 5-15", "trim X from 5 to 15", "keep only 2 to 8 seconds of X", "give me just the first 5 seconds of X".
2. **A phrase that names a part removes it** (cut_out).
   - Verbs: *cut*, *cut out*, *snip out*, *remove*, *delete*, *drop*, *take … out (of)*, *X without …*, each followed by the part.
   - Examples: "cut the first 10 seconds off X", "cut the first 10 seconds of X", "remove the last 5 seconds", "drop seconds 2 through 8", "cut out 2-8 from X".
3. **No part and no times means ask** (clarify listing the missing arguments).
   - Examples: "trim it", "cut X", "shorten X", "cut out the bad part".
4. **Things MiniCut can't do mean ask, offering what it can.**
   - Example: "make X quiet" → 'Do you mean "mute X"? I can't lower the volume.'

The frozen eval already follows rules 1–3. A prototype classifier over all 68 eval trim / cut_out / trim-or-cut clarify prompts found no contradiction. Its only 6 misses were removal verbs missing from its list (*without*, *snip out*, *take … out of*), now added above.

The rule has no "flip" ambiguity. The reading-ambiguous clarify class (`v3_amb_*`, 60 rows) goes away, except the unsupported-request rows (rule 4).

## Other tools (keywords already used consistently by the templates and eval)

| tool | keywords | numbers the user gives |
|---|---|---|
| concat | join, concatenate, combine, merge, append, stick … together, put N copies together, loop, … in a row | the files, in order |
| scale | make X WxH, resize X to W by H / WxH, scale X up/down to WxH, fit X into a WxH frame | width and height |
| speed | run/play X at Nx speed, speed X up by N%, make X N times as fast, slow X down to Nx / N% speed | the factor |
| convert | convert / re-encode / save / transcode X … CONTAINER … CODEC | container and codec |
| extract_audio | extract / save / pull / rip / get the audio (soundtrack, mp3) from X as FORMAT | the format |
| mute | mute, silence, remove/drop/strip the audio (sound) from X, make X silent, X with no sound | none |

Named sizes ("720p") and percentages ("the first 25%") need the clip's size or duration, which the model never sees (P3-D12). The phrasebook lists them as unsupported for now and suggests exact numbers.

## Work, by effort

| # | Change | Effort |
|---|---|---|
| 1 | `crates/minicut/src/phrasebook.rs`: the keyword table, plus a classifier `meaning_of_wording(text) -> Keep / Remove / Ask` for the rule above. It is data and a few string checks, with no model involved. | 2 h |
| 2 | Tests: every `TEMPLATES` row classifies to its declared kind; every frozen-eval trim/cut_out/clarify prompt classifies to its expected tool. That second test proves the eval already speaks the phrasebook. | 1 h |
| 3 | Templates: move "cut the first N seconds of X" to a `RemoveFirst` template. Delete the flip rows (`v3_amb_trim_*`, `v3_amb_cut_first`) and the near-miss machinery they needed. Keep `v3_amb_quiet` as an unsupported-request clarify. | 1 h |
| 4 | `minicut --help` (Rust) and the Python CLI's `--help` epilog print the phrasebook, generated from the one table. A sync test keeps `docs/phrasebook.md` current. | 1 h |
| 5 | Regenerate as `train_verified_v4` (pilot-matched, same gate), run acceptance, then sweep seeds 1–3 (about 25 min GPU) and add the results to the board. | 1 h + GPU |

Total about one working day. The verified core is unchanged: labels still come only from `try_label` and its siblings.

## Open questions

- **Q1. "trim the first/last N (of X)".** Video editors usually mean remove. Under rule 2 this phrase names a part, so it removes. But it sits next to the eval's "trim X **to** the first 10 seconds" (keep), the same kind of near-pair that just failed. Options:
  - (a) remove, per the rule;
  - (b) leave it out of training entirely and list only canonical phrasings in the phrasebook.
  - Recommendation: (b). Teach only phrasings whose meaning is carried by a clear keyword.
- **Q2. System prompt.** Should the system prompt carry a one-line version of the rule? That would change the prompt used in training and inference (parity tests cover both). Recommendation: not yet; measure the data change alone first.
