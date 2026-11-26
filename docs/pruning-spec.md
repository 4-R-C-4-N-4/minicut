# MiniCut pruning spec: finding the smallest model that can still do the job

Status: proposal for review — doc only, no code, no training, no eval changes.
Date: 2026-11-26
Base: `main` at `d79051a` (Merge PR #8, near-miss sweep)
Author: Muse, for Ivy — for review by the local agent.
Scope: MiniCut only. Lilguy is out of scope here except as a contrast in §2.

## 0. The question (Q1)

MiniCut is a family-of-experts idea in miniature: one narrow task,
one tiny model tuned for exactly that task, with the wrapper and the
verified core doing everything deterministic. The question this spec
asks is:

> Can we find the theoretical floor of the entropy required by
> MiniCut's input/output grammar, use that floor as a tuning mechanism,
> and trim MiniCPM5-2B down to literally the smallest size that can
> still operate — and if so, how, concretely, with this base model?

Short answer, up front: the *exact* floor is not computable, but a
*usable* floor is measurable. This spec defines what "usable" means,
what to measure, and the experiment that finds it. It does not claim
a number in advance.

## 1. What MiniCut is now (ground truth for this spec)

- Base model: `openbmb/MiniCPM5-2B`, dense `LlamaForCausalLM`,
  2,516,756,480 parameters total / 1,981,982,720 non-embedding,
  42 layers, hidden 2048, intermediate 6144, GQA 16 Q / 2 KV heads,
  head_dim 128, vocab 130,560, untied embeddings, context 131,072.
  See the model card: https://huggingface.co/openbmb/MiniCPM5-2B
  and `spec.md` §2. The final RL+OPD checkpoint is the fine-tune
  base, not SFT-only / Base / Midtrain.
- Task: v1 is 8 tools + `clarify` — trim, cut_out, concat, scale,
  speed, convert, extract_audio, mute. Crop, rotate, fades are v2.
  Source of truth: `config/tools.yaml`. See `spec.md` §5.
- Output: exactly one JSON tool call, GBNF-constrained in llama.cpp.
  The wrapper validates against ffprobe, builds deterministic ffmpeg
  argv (no shell), runs it, and verifies the output. The model never
  emits raw ffmpeg/bash and never chooses paths. See `README.md`
  and `spec.md` §3–§4.
- Input side, current state: the Rust port (PR #4, merged) makes the
  tool contract a single Rust enum in `crates/minicut-core`, which
  generates the grammar, validates calls, and labels training data.
  See `docs/rust-port.md`. The phrasebook is a *proposal*, not yet
  implemented (`docs/plans/2026-11-25-phrasebook.md`): it would
  replace "is this wording ambiguous?" judgements with one written
  keep-vs-remove rule, taught consistently and shown in `--help`.
  As of the base commit the `v3_amb_*` flip rows are still in the
  templates, and the near-miss sweep (PR #8) did *not* stabilise
  them — see Appendix A. Do not read this spec as claiming the
  phrasebook already exists.
- Eval, frozen: Phase 3 eval, 4 splits × 50 cases, seed 20261120,
  oracle sanity 200/200. Two scoring systems exist and must not
  be conflated (see §1.1): the historical verification-only
  baselines (v2 94.5%, v3 95.5%, 191/200) and the current
  `intent_v2` / `task_pass` scoring, on which stock is 167/200
  and the best tuned pilot is 191/200 on one seed. Do not touch
  the frozen eval inputs in any experiment under this spec.
  See `eval/freeze.json`, `eval/PILOT_BOARD.md`, `PROGRESS.md`.
- Training data: generated from templates, with labels from the
  verified core — but *not yet* from a single source. As of the
  base commit, `training/generate_dataset.py` and
  `eval/generate_eval.py` still have separate hand-written
  templates and do not read `config/tools.yaml`; the
  dictionary-as-generator work that would make them one source
  is not yet built (see §7.3 and Appendix A). The claim in §6
  that distillation data is "free, exact, and unlimited"
  therefore depends on that unbuilt work plus the freeze rules
  in `eval/README.md` (held-out template ids, reserved fixture,
  combo pools, no eval sampling into training). Until the
  generator is single-source, every dataset version must be
  recorded by generator commit + seed + row counts, and
  template drift must be checked explicitly — it has already
  bitten this project twice (`isolate`, `720p`; see Appendix A).

### 1.1 Teacher identity and scoring — unresolved, must be named before any run

This is the biggest open item in the spec, and it was underspecified
in v1 of this doc. There is no single settled "fine-tuned 2B teacher"
yet, and the two baseline numbers in circulation measure different
things:

- **Verification-only baselines (historical, not comparable):**
  v2 94.5%, v3 95.5% (191/200). These scored whether the *executed
  edit verified* against the fixture, not whether the model chose
  the intended tool and args. `eval/README.md` states this
  explicitly: "baseline_v1–baseline_v3 are historical and their
  legacy verification rates are not comparable to `task_pass`."
  Do not use 95.5% as a teacher score or as an acceptance bar.
- **`intent_v2` / `task_pass` (current):** `task_pass` requires the
  intended action and canonical arguments plus verified media, or
  an accepted clarification with exactly the expected missing
  fields. The grader does *not* judge the wording of clarification
  questions. On this scoring, stock MiniCPM5-2B is **167/200** in
  every pilot-board run, and the tuned pilots are:
  - first LoRA pilot (100 updates, rank 16, seed 42): tuned
    197/200 on the then-current grader, re-graded 191/200 on
    today's grader; gate **failed** on one new keep/remove
    inversion (`isolate seconds 3–9` read as removing that range).
  - verified-label pilot (`train_verified_v1`, same recipe/seed):
    tuned 191/200, 0 new inversions, gate **passed** — once.
  - seed sweep, verified v2/v3 data: tuned 175–191 across seeds
    1–3; one seed collapsed to 175 through over-clarifying
    (18 unnecessary clarifies). Verified data introduces no new
    inversions in 7 runs, but it does not reliably hold 191.
  - Stock stays the default. See `eval/PILOT_BOARD.md` and
    Appendix A for the full table.

Consequence for this spec: **the teacher must be named in the
experiment log before the ladder starts**, with all of: base
checkpoint (final RL+OPD, F16 GGUF), adapter if any (which data
version, which seed, merged or not), server and decoding (§6.3),
and its frozen-eval score on `intent_v2` measured on the same
grader as the students. The candidates are (a) stock F16
(167/200), (b) the verified-v1 LoRA adapter merged (191/200,
one seed, gate passed once), or (c) a fresh teacher trained on
the current verified data with a seed sweep, so its score is
not a single-seed artefact. The spec does not pick between
these — that is open question §8.1 — but no student result is
interpretable until the pick is recorded. If the teacher is
(b) or (c), the LoRA adapter must be *merged* into the base
before pruning: pruning a LoRA-tuned model, or pruning the
base and keeping an adapter trained for 42 layers, invalidates
the adapter. Order is: prune the base, then distill-train the
student (see §6.3).

Non-goals for this spec: no change to `config/tools.yaml`, the GBNF
grammar, the wrapper, the frozen eval, or the release benchmark;
no MoE; no general-benchmark chasing (MMLU, GSM8K, etc. are not
MiniCut metrics); no training from scratch as the default route
(see §6 for why).

## 2. The entropy floor, in three parts

Do not use one entropy. MiniCut has three, and only one of them
needs a model:

1. `H(raw English)` — huge and open-ended: synonyms, paraphrase,
   clock vs seconds vs percent, "cut the first N seconds of X"
   vs "… off X".
2. `H(canonical intent)` — tiny. 8 tools + clarify, each with a
   small typed arg set. That is the actual decision.
3. `H(output | canonical intent)` — near zero when the grammar is
   deterministic. Once the intent is `trim:12–45`, the JSON shape,
   field names, and arg types are forced by the grammar and the
   verified core.

A model is only needed for the gap between (1) and (2): noisy
English → canonical intent. Everything after that can be Rust,
proved in Verus, with zero model parameters. That is the
dictionary-as-enum idea: the dictionary *is* the canonicalisation.

Two consequences follow, and both are testable:

- **Synonyms do not raise the floor.** If "cut out", "remove",
  and "delete the middle" all parse to the same canonical
  `cut_out`, they collapse to one equivalence class. Input entropy
  rises; canonical entropy does not. Capacity needed grows with
  the number of *distinct meanings* and with *ambiguity* — inputs
  that genuinely map to two intents — not with the number of ways
  to say the same thing.
- **Ambiguity is where the model earns its keep, and clarify is
  the honest answer there.** An input with `H(intent | input) > 0`
  should produce `clarify` with the exact missing list (computed
  by the verified core), not a guess. The phrasebook's flip rows
  (`v3_amb_*`) are the training-set form of this: the same span
  read as keep vs remove. The spec's position is that ambiguity
  should be *measured* (how often, on which templates) and
  *priced* (clarify costs a turn, a wrong edit costs a bad file),
  not silently absorbed by a bigger model.

### 2.1 Why the exact floor is not computable

The exact floor is, roughly, the Kolmogorov complexity of the
function `raw English → canonical intent` on MiniCut's task
distribution: the length of the smallest program that implements
it. Kolmogorov complexity is uncomputable in general, and the
function here is not even fully specified — "raw English" is
unbounded, and the task distribution is whatever users actually
type. Counting distinct outputs does not rescue it: discretised
to 0.1 s, MiniCut's output space is tens of millions of combos,
which is only ~25 bits. The hard part is not storing outputs,
it is *generalising* the mapping from unseen phrasings. A
counting bound is therefore weak by orders of magnitude, and
this spec does not use one as a target.

What *is* computable is an empirical floor, defined in §5:
the smallest student size whose frozen-eval accuracy stays
above an agreed threshold, with the scaling law fitted to
that task. The true theoretical floor lies below that number
by an unknown training-inefficiency gap. This spec treats the
empirical floor as the shippable answer and the theoretical
floor as the direction of travel, not a number to claim.

### 2.2 The scaling-law form to fit

For general language modelling, loss follows a power law in
parameters `N`, data `D`, and compute `C`, with an irreducible
floor `E` — the entropy of text itself:

- Kaplan et al. (2020): `L(N) = (Nc/N)^αN`, `αN ≈ 0.076`,
  and the joint form `L(N,D)` with overfitting depending on
  `N^0.74 / D`. https://arxiv.org/abs/2001.08361
- Hoffmann et al. (2022, Chinchilla): `L(N,D) = E + A/N^α + B/D^β`,
  with fitted `E = 1.69`, `α ≈ 0.34`, `β ≈ 0.28`, and the
  compute-optimal rule `D ≈ 20·N`. https://arxiv.org/abs/2203.15556
  Note the audit caveat in the literature: the published
  parametric fit has been criticised and the exact constants are
  less certain than the headline; the qualitative conclusion
  (scale data with the model) survives.

For MiniCut, fit the *task* version of the same form:

```
L_task(N) = E_task + A_task / N^α_task
```

`E_task` is MiniCut's irreducible loss, not general text entropy.
It is estimated from the teacher: the fine-tuned 2B model's
cross-entropy / error rate on held-out task data, at the point
where more parameters stop helping. `A_task` and `α_task` are
fitted from the student ladder in §6, not borrowed from
Chinchilla — a narrow task's exponents will differ from general
text, and borrowing them would be the error this spec is
designed to avoid. Definition of `N` for the fit: **non-embedding
parameters** throughout (see §5.1), because vocab trimming
moves total parameters a lot while moving non-embedding
parameters not at all; total parameters and GGUF bytes are
reported alongside, never substituted for `N`. `E_task` itself
must be measured on held-out data the teacher was *not*
trained on (frozen eval prompts are held out by the freeze
rules, but the teacher's training corpus is drawn from the
same template pools, so also report teacher loss on the
unseen-paraphrase split alone, where the templates were never
in training). A teacher loss measured on data the teacher
trained on is an overfit number, not `E_task`.

The curve to expect: smooth, diminishing returns, no cliff.
That is the entropy curve in usable form. The knee — where
accuracy falls off faster than parameters are saved — is the
floor for a given acceptance threshold.

Evidence that the floor moves with data, not just size:
MobileLLM-R1-950M matches or surpasses Qwen3-0.6B on reasoning
benchmarks while training on only 11.7% of Qwen3's tokens,
because the data was curated and resampled for reasoning
(AIME 15.5 vs 0.6 for OLMo-2-1.48B). Zhao et al., ICLR 2026,
https://arxiv.org/abs/2509.24945 — latest version 27 Feb 2026.
Data quality, distillation, and test-time compute (grammar,
wrapper verification, clarify) all shift the curve left.
A small model plus a good wrapper can beat a bigger bare model
on a narrow task. That is the whole bet of this spec.

## 3. MiniCPM5-2B anatomy: where the parameters are

All figures from the model card / `config.json`
(https://huggingface.co/openbmb/MiniCPM5-2B, accessed 2026-11-26;
also `spec.md` §2):

| Part | Size | Notes |
|---|---|---|
| Total | 2,516,756,480 | BF16, 5.03 GB |
| Non-embedding | 1,981,982,720 | |
| Embeddings (input + lm_head, untied) | 534,773,760 | 130,560 × 2,048 × 2; 21% of total |
| Layers | 42 | `LlamaForCausalLM`, RMSNorm, SwiGLU, RoPE θ=5e6 |
| Hidden / intermediate | 2,048 / 6,144 | |
| Attention | 16 Q heads / 2 KV heads, head_dim 128 | GQA 8:1 |

Per-layer arithmetic (from the config, before any training):

- Non-embedding per layer: 1,981,982,720 / 42 ≈ 47.2M.
- Attention per layer: q 2,048×2,048 = 4.19M, k 2,048×256 = 0.52M,
  v 0.52M, o 2,048×2,048 = 4.19M; total ≈ 9.4M.
- MLP per layer: gate 2,048×6,144 = 12.58M, up 12.58M,
  down 6,144×2,048 = 12.58M; total ≈ 37.7M (~80% of the layer).
- Check: 9.4M + 37.7M ≈ 47.1M, matching the 47.2M above.

Implications, in order of payoff:

1. **Vocabulary is the biggest free win — with the §4.1 caveat.**
   535M params, and the lm_head is also the decode bottleneck:
   every generated token reads all 267M lm_head parameters in a
   memory-bandwidth-bound decode loop. MiniCut's *tool-call-only*
   vocabulary is tiny — tool names, numbers, punctuation, a few
   enums — on the order of 200 tokens. The *actual* output
   vocabulary as trained is larger, because clarify questions
   are free-text English and 12% of training is general
   plain-text replies (see §4.1). The vocab saving is real
   either way; its size depends on the clarify decision in §4.1.
   See §4.1.
2. **MLP width is next.** The MLP is ~80% of every layer, so
   cutting intermediate 6144 → 3072 halves the bulk of the model.
3. **Depth is next.** Each layer is ~47M non-embedding. 42 → 24
   layers saves ~850M non-embedding params before any other change.
4. **Quantisation is orthogonal.** F16 (5.03 GB) → Q8 → Q4 halves
   bytes per step with no architecture change. MiniCut already
   runs the F16 GGUF in llama.cpp; Q4_K_M is the usual local
   form. BitNet-style ternary weights (1.58 bits/param) are a
   further step, but they require training from scratch with
   quantisation in the loop, not post-training quantisation —
   see §4.4 and the BitNet reference below.

## 4. Trimming levers, in order

### 4.1 Vocabulary trimming (do this first in the analysis)

Vocabulary trimming deletes tokens that never occur in the
target task, shrinking both the input embedding and the lm_head.
Ushio & Zhou show it can retain the original performance of a
multilingual model while being considerably smaller, because
the embedding matrix can dominate a small model's parameters.
https://arxiv.org/abs/2305.15020
For small models generally, the embedding share grows as the
model shrinks (3% at 70B, 13% at 8B, 34% at 1.8B in the
COLING 2025 study), and reducing vocab to 8k cuts ~1B params
from Llama-3-8B; fine-tuning recovers much of the lost
performance. https://aclanthology.org/2025.coling-industry.64.pdf

COMPACT (NeurIPS 2025 workshop) does this jointly with FFN
pruning: prune rare vocabulary to shrink embedding/lm_head,
then prune FFN channels weighted by the *remaining* common
tokens, keeping a standard transformer layout.
https://openreview.net/pdf?id=0iXlSGtiH8
That joint order matters — prune the vocab first, then score
FFN channels on the tokens that survive.

For MiniCut, split the two matrices, because they have
different jobs:

- **lm_head (output):** prune to the tokens that can appear in
  a valid tool call. That set is enumerable from
  `config/tools.yaml` + the GBNF grammar: tool names, digits,
  `.`, `-`, `:`, `,`, `[`, `]`, `{`, `}`, `"`, enum values
  (`libx264`, `mp4`, `mp3`, …). **But that enumeration is only
  the whole story if clarify questions stop being free text.**
  Today `clarify.question` is a free-text `str`
  (`config/tools.yaml`), the model writes the question in
  English, and the grader scores only the `missing` list, not
  the wording (`eval/README.md`, `crates/minicut/src/clarify.rs`).
  The training mix is also 12% general plain-text replies
  (360/3000 in `train_v1.jsonl`, `training/README.md`), which
  need a general English output head. So the "~200 tokens"
  figure in v1 of this spec was wrong as stated: it is the
  tool-call-only token set, not the actual output token set
  of the model as trained. Two honest options, and the spec
  must pick one before a vocab-trim experiment:
  (a) **Templated clarify:** the question is generated by the
  verified core from the `missing` list and the phrasebook
  vocabulary (`templates::VOCAB`), and the model outputs only
  the tool name + `missing` — a design change to the tool
  contract and the grammar, not a measurement; or
  (b) **English output head kept:** the lm_head is trimmed
  only to English + tool-call tokens actually seen in training
  outputs, which will be thousands of tokens, not ~200, and
  the saving is correspondingly smaller.
  Either way, measure the exact output token set from the
  training outputs (tokenise every assistant reply in the
  distillation set, count distinct token ids, and report the
  cumulative-frequency curve) before claiming a number.
  Do not guess it.
- **Input embedding:** prune to tokens actually seen in the
  training/eval corpora plus the phrasebook synonyms, plus
  byte-fallback tokens. Keep a frequency count from the
  grammar-generated corpus and cut below a threshold, then
  measure the out-of-vocabulary rate on held-out paraphrases.
  The risk is a real user word that was never in the corpus —
  that failure mode must be measured, not assumed away.
  Note what must survive any trim, because it is not in the
  task corpus alone: the system prompt and few-shot examples
  (identical at train and inference, `minicut/llm/prompts.py`),
  the GBNF/JSON punctuation, and BPE subwords for arbitrary
  filenames and numbers. Filenames are copied from the prompt
  into the `input` field (30 filenames in `train_v1`, and the
  eval convention is `clip.mp4`), and numbers are tokenised
  unpredictably by the MiniCPM tokenizer — `12.5`, `1m30s`,
  `720p` do not decompose into a tidy digit set. Measure the
  token set on tokenised prompts, not on words.

Open question for the reviewer: does llama.cpp / GGUF support
a trimmed vocab cleanly for this model (retokenised tokenizer,
remapped ids, lm_head rows dropped), or does vocab trimming
force a custom export? Answer that before scheduling any
vocab-trim experiment. If GGUF makes it painful, the
vocab-trim saving is still real for a future runtime, but it
is not the first experiment.

### 4.2 Structured pruning: depth and width (Minitron method)

The reference method is Muralidharan et al. (NVIDIA, 2024):
prune an existing LLM and retrain with <3% of the original
training data, combining depth, width, attention, and MLP
pruning with knowledge-distillation retraining. Compressing
Nemotron-4 15B by 2–4× needed up to 40× fewer training tokens
per model than training from scratch, saved 1.8× compute for
the full family, and improved MMLU by up to 16% vs
training from scratch.
https://arxiv.org/abs/2407.14679
The follow-up practice report applies the same method to
Llama-3.1 8B → 4B and Mistral NeMo 12B → 8B, comparing depth
pruning against joint hidden/attention/MLP (width) pruning,
and finds width-pruned variants outperform depth-pruned ones
at the same size.
https://arxiv.org/abs/2408.11796

Applied to MiniCPM5-2B:

- **Depth pruning:** remove whole layers. Easiest to implement
  in transformers/llama.cpp terms, biggest latency win (fewer
  sequential dispatches), but coarsest — accuracy can drop
  abruptly. Rank layers by importance first (ablation: drop one
  layer, measure frozen-eval drop; or activation/gradient
  magnitude on task data), do not drop uniformly, and record
  *which* layers were dropped: dropping the last N layers,
  dropping evenly spaced layers, and dropping the
  importance-ranked lowest layers are three different
  experiments with different results. The ladder in §6.2
  assumes importance-ranked dropping; if the reviewer picks a
  different rule, the ladder's sizes stay the same but its
  results will not be comparable to this spec's predictions.
  Prune the *base* model, then distill-train — never prune a
  model that already carries a LoRA adapter trained for the
  unpruned shape (see §1.1).
- **Width pruning:** remove attention heads, MLP intermediate
  channels, and/or hidden channels. Finer-grained, usually
  better accuracy at the same parameter count, but needs
  surgery on every matrix that touches the pruned dimension
  and a retraining pass to recover.
- **Combine, in Minitron order:** importance-rank → trim to
  target size → distill-retrain (teacher = fine-tuned 2B,
  see §6) → measure → repeat one axis at a time so the
  experiment attributes the drop.

### 4.3 Knowledge distillation (the retraining step)

Distillation trains a smaller student to match a larger
teacher, not just hard labels but the teacher's logits /
intermediate states. Hinton et al. (2015):
https://arxiv.org/abs/1503.02531
For MiniCut the teacher is already fine-tuned and the
distillation data is grammar-generated, so the student
never sees unlabelled general text — it sees only task
inputs with teacher labels, which is exactly the distribution
it will serve. One practical point the v1 spec glossed over:
**teacher logits do not come out of llama.cpp.** The
`llama-server` used for eval serves completions, not a
logit distribution over the vocabulary. Logit distillation
therefore needs the teacher in transformers/PyTorch (HF
`openbmb/MiniCPM5-2B`, LoRA adapter merged first — see §1.1),
run offline over the distillation prompts, with the logits
saved to disk. If that pipeline is not built, the first
distillation arm is hard-label-only by necessity, and the
spec's "KL is the default" is aspirational. Say which arm
was actually run in the log. Two styles to distinguish in the experiment:

- **Logit distillation (KL on the teacher's distribution):**
  the student learns the teacher's uncertainty, including
  near-miss cases (keep vs remove). This is the Minitron
  retraining style and the right default.
- **Hard-label fine-tuning on teacher outputs:** simpler, and
  sufficient if the teacher is near-deterministic on the task
  (greedy tool calls). Use it as a control, not the default,
  because it throws away the teacher's uncertainty on
  ambiguous inputs — the cases where clarify matters.

Teacher correction (Minitron practice report, §1): if the
teacher was not trained on the distillation data, fine-tune
it on that data first, then prune/distill. For MiniCut the
teacher *is* the fine-tuned model on grammar-generated data,
so this step is already done — note it explicitly in the
experiment log rather than repeating it.

### 4.4 Bits per parameter (quantisation, separate axis)

Quantisation changes bytes, not parameter count, and it
stacks with pruning:

- F16 → Q8 → Q4_K_M in GGUF: the usual local ladder, already
  supported by llama.cpp. Measure task accuracy at each step;
  do not assume Q4 is free.
- BitNet b1.58 (Ma et al., 2024): every weight ternary
  {-1, 0, 1}, 1.58 bits/param, trained from scratch with
  quantisation in the loop; matches full-precision perplexity
  and end-task performance at the same size/tokens.
  https://arxiv.org/abs/2402.17764
  This is *not* a post-training step for MiniCPM — it is a
  different training recipe. List it here as the far end of
  the bits axis, not as a near-term experiment. If a future
  student is trained from scratch, 1.58-bit is the floor to
  compare against; for pruning MiniCPM, Q4 is the near-term floor.

### 4.5 What not to do

- **Do not train a small model from scratch as the default.**
  Minitron's result is that prune+distill beats from-scratch
  at the same size and costs far fewer tokens, because the
  student inherits the teacher's general English. From-scratch
  is a control arm at most, to quantify that gap for *this*
  task — not the plan.
- **Do not use a Mixture-of-Experts as the answer.** A MoE
  still shares embeddings, a router, and general capacity.
  The family-of-experts idea here is separate models, each
  trimmed to its own task floor, verified independently.
  A MoE router is a different (and weaker, for this purpose)
  claim.
- **Do not prune by parameter count alone.** A 1B model with a
  130k vocab and a 1B model with a trimmed vocab are different
  machines at decode time (lm_head reads, KV, latency). Report
  params, GGUF bytes, *and* measured tok/s on the target
  hardware for every student.

## 5. What "minimum" means here (acceptance, before any training)

No experiment under this spec starts until these are written
down. The reviewer should fill in or correct the numbers:

- **Primary metric:** intent accuracy on the frozen Phase 3
  eval (4 splits × 50, seed 20261120), scored exactly as the
  current baselines are scored — that is `intent_v2` /
  `task_pass` (§1.1), not the historical verification-only
  rate. Report per-split, per-tool, and the keep/remove confusion
  separately — a student that keeps 95% overall but inverts
  keep/remove is a failure, not a pass.
- **Acceptance threshold (proposal, to confirm):** ≥95% intent
  accuracy on frozen eval, with no per-tool regression vs the
  2B teacher larger than 2 points on any of the 8 tools, and
  no increase in the keep/remove inversion count. Clarify
  behaviour is scored separately: a clarify on a genuinely
  ambiguous input is correct; a clarify on an unambiguous
  input is an error (over-clarifying, the failure mode the
  near-miss sweep already saw).

### 5.1 Statistics: 200 cases is not enough to tell close students apart

This was underspecified in v1 and it matters, because the
proposed bar sits 1 case below the best tuned teacher
(191/200). On 200 cases, a 95% Wilson interval around
191/200 is roughly 91–98%; a 2-point per-tool bar is less
than one case on most tools (per-tool counts in the frozen
eval are uneven and small — see `eval/freeze.json`).
A student at 190 and a student at 185 are not distinguishable
on this eval alone, and a single seed can move a run by
~15 points (verified v2 seed 1 collapsed to 175; see
Appendix A). So:

- Report **paired** per-case differences against the teacher
  on the same grader and the same server, not just two
  headline percentages. A student that gains 5 and loses 5
  is not the same as one that gains 0 and loses 0.
- Report **seed-averaged** scores (seeds 1–3 at minimum, as
  in the pilot board) with the range, not a single seed.
  A student that passes on one seed and collapses on another
  has not passed.
- Report a **confidence interval** (Wilson, 95%) on the
  headline number, and treat a student whose interval
  overlaps the teacher's as "not distinguished", not "equal".
- Use the **release benchmark** (`eval/release/`, 38 pinned
  cases, controls 38/38, no model scored yet as of the base
  commit) as a **co-primary**, not a secondary: it is
  independently authored and frozen, so it guards against
  overfitting to the dev splits. It is also small (38 cases),
  so it is a guard, not a replacement — if dev and release
  disagree, that disagreement is itself a finding to report,
  not to average away.
- If the reviewer wants a sharper bar, the honest options are
  a larger frozen eval (a new frozen set, generated once and
  then frozen — a real cost, and it must obey the same freeze
  rules) or a lower acceptance bar with a stated margin
  (e.g. "within 3 points of the teacher, paired, seed-averaged").
  Do not sharpen the bar by re-reading the same 200 cases
  more confidently.
- **Secondary metrics:** release benchmark (`eval/release/`),
  seed sweep (seeds 1–3, as in the pilot board), and a
  held-out paraphrase split if one exists. General benchmarks
  are not acceptance metrics.
- **Deployment metrics (proposal):** GGUF bytes, RAM at
  runtime, tok/s on the target machine (state the machine),
  and cold-start time. A student that is smaller on disk but
  slower at decode because of a shape llama.cpp dislikes is
  not a win — measure it.
- **Stop rule:** stop trimming when the fitted curve predicts
  the next step down would breach the threshold, *and* one
  step past the knee has been measured to confirm the knee
  is real (one step of over-trimming, on purpose, with the
  result recorded). Do not stop at the first size that passes.

## 6. The experiment: a student ladder

### 6.1 Data (fixed for all students)

- Train: grammar-generated from the single source of truth
  (`config/tools.yaml` + the Rust core's label function +
  the phrasebook templates). Teacher labels come from the
  fine-tuned 2B model; hard labels come from the verified core.
  Record the generator commit, the seed, and the row counts.
- Eval: frozen Phase 3 eval only, plus the release benchmark.
  No new eval wording is introduced during the ladder —
  that would move the target while measuring it.
- Distillation set: the same prompts, with teacher logits
  saved (temperature noted). If logits are not saved, say so;
  hard-label-only runs are a different arm and must be
  labelled as such.

### 6.2 Ladder (illustrative sizes — arithmetic, not results)

Depth-only first, because it is the cleanest attribution.
All sizes below are arithmetic from §3 (47.2M non-embedding
per layer, 535M embeddings at full vocab), before any training.
Vocab is held at full size in this table so depth is the only
variable; vocab trimming is a separate axis (§4.1, §7).
**`N` in the fit is non-embedding parameters** (§2.2); the
table gives total as well so the two are never confused.
Depth points and width points must be fitted as **separate
curves** (or with shape as an explicit covariate) — a 24-layer
model and a 24-layer model with halved intermediate size are
not two points on one size curve, they are two shapes at
overlapping sizes, and fitting one power law through both
conflates shape with size.

| Student | Layers | Non-embedding | Total (full vocab) | Note |
|---|---|---|---|---|
| Teacher (2B) | 42 | 1.982B | 2.517B | named teacher, §1.1 — do not use 95.5% here |
| S1 | 32 | ~1.510B | ~2.045B | depth −10 |
| S2 | 24 | ~1.133B | ~1.668B | depth −18 |
| S3 | 16 | ~0.755B | ~1.290B | depth −26 |
| S4 | 12 | ~0.566B | ~1.101B | depth −30 |

Then, at the best depth, one width step as a separate arm:

| Student | Layers | Intermediate | Non-emb./layer | Total (full vocab) |
|---|---|---|---|---|
| S2w | 24 | 3072 | ~28.3M | ~1.214B |
| S2w2 | 24 | 2048 | ~22.0M | ~1.063B |

**The ladder as written never goes below ~1.06B.** If the
floor is 0.5B, this ladder cannot find it — it can only say
"still passing at 1.06B". Extend it, as a second phase, with
combined axes (depth + width, then + vocab), still arithmetic
until measured:

| Student | Layers | Intermediate | Non-embedding | Total (full vocab) | Note |
|---|---|---|---|---|---|
| S5 | 12 | 3072 | ~0.340B | ~0.875B | depth + width |
| S6 | 12 | 2048 | ~0.264B | ~0.799B | depth + width, narrower |
| S7 | 8 | 3072 | ~0.226B | ~0.761B | depth + width, shallow |

Sub-0.8B at full vocab is not reachable by depth+width
alone without going very shallow; going below that needs
vocab trimming as well (at, say, 32k vocab the embedding
bill falls from 535M to ~134M, and S6's total falls to
~0.40B). That is why vocab is a separate axis and why
§4.1's feasibility question has to be answered before the
second phase is scheduled — the sub-1B floor, if it exists,
lives in the combined region, not on any single axis.

Per-layer width arithmetic: MLP at intermediate 3072 is
3 × 2,048 × 3,072 = 18.9M; plus attention 9.4M = 28.3M/layer.
At intermediate 2048: MLP 12.6M + attention 9.4M = 22.0M/layer.

The reviewer should sanity-check these numbers against the
actual tensors before running anything — they are config
arithmetic, and a wrong intermediate size in the table is a
cheap error to catch now.

Candidate control student (to check, not assumed): MiniCPM5-1B,
the smaller sibling in the same series, as a from-scratch
baseline at ~1B. Record its config and whether it shares the
tokenizer before using it as a control. Do not assume it is
the same architecture scaled down — check.

### 6.3 Procedure per student

1. Rank and prune (Minitron order): importance-rank layers /
   heads / MLP channels on task data, trim to target, record
   exactly what was removed.
2. Distill-retrain: teacher = named teacher (§1.1), student = pruned
   model, KL distillation on the grammar-generated set plus
   hard labels from the verified core. Record temperature,
   epochs, LR, and tokens. Keep the LoRA-vs-full-FT choice
   explicit: the teacher was LoRA-tuned; a pruned student
   needs full fine-tuning of the surviving weights, not a
   LoRA on top of a pruned base, unless the experiment says
   otherwise and justifies it. **Feasibility note:** full FT
   of a ~1–2B model with AdamW in bf16 needs roughly
   2 bytes/param weights + 2 bytes/param grads + 8 bytes/param
   Adam states ≈ 12 bytes/param, plus activations — ~24 GB
   for a 2B model before activations, which is the whole of
   the RTX 3090's 24 GB. The LoRA pilot used 7.12/16.84 GiB
   peak allocated/reserved CUDA memory precisely because it
   was LoRA. Full FT on that card therefore needs a stated
   plan before it is scheduled: 8-bit Adam / Adafactor,
   gradient checkpointing, a smaller student first, or a
   different machine. Do not schedule a full-FT ladder
   assuming LoRA memory.
3. Convert to GGUF (same quant as the teacher's baseline,
   F16 first, then Q4 as a separate arm), run in llama.cpp
   with the same server, prompt, grammar, and decoding as
   the baseline. Parity of runtime is part of the result.
   **Pin the runtime in every run log:** llama.cpp commit
   (pilots used `01818e495`), GGUF file and quant, server
   flags (`--jinja`, context size), system prompt and
   few-shot (identical to `minicut/llm/prompts.py`),
   thinking off, greedy decoding, `temperature=1.0,
   top_p=0.95, min_p=0.0` per the model card (llama.cpp
   defaults `min_p=0.05` — set it to 0.0), and the GBNF
   grammar file. A student scored under different decoding
   is not comparable to the teacher.
4. Score frozen eval + release benchmark + seed sweep.
   Record per-tool, keep/remove, and clarify behaviour.
5. Fit `L_task(N) = E_task + A_task / N^α_task` to the
   points so far. After 3 points, publish the fitted
   `E_task`, `A_task`, `α_task` and the predicted `N_min`
   for the §5 threshold. Update the fit after each new point.
   If the fit is unstable (fewer than 3 points, or a point
   far off the curve), say so — do not extrapolate from 2 points.

### 6.4 What success looks like

- A fitted task scaling law with at least 4 student points
  on a single axis (depth first), a stated `E_task`, and a
  predicted `N_min` that the next measured point confirms
  within its error bars. Width and vocab points are fitted
  separately, not pooled into the depth curve (§6.2).
- A smallest passing student, named, with its GGUF bytes,
  tok/s, and per-tool scores, and one measured failing
  student one step smaller, so the knee is bracketed, not
  guessed.
- A short note on which axis paid most: vocab, depth, width,
  or bits. That note is the transferable result for lilguy
  and for the next expert in the family — it is not a
  MiniCut-only curiosity.

## 7. Role of the grammar / Verus core in the floor

This is the part that makes MiniCut's floor lower than a
general model's floor, and it should be measured, not asserted:

1. **Canonicalisation lowers input entropy.** Measure it:
   token entropy / distinct surface forms per canonical intent
   before vs after phrasebook canonicalisation, on the
   training corpus. The phrasebook (`docs/plans/2026-11-25-phrasebook.md`)
   and the template table (`docs/templates.md`) are the
   artefacts to measure against.
2. **The output grammar removes syntax learning.** The model
   never has to learn JSON shape, field names, or arg types —
   GBNF enforces them. The ablation that proves this is cheap
   and worth doing once: score the same student with the
   grammar on vs off (or with a looser grammar). If the gap
   is large, the grammar is doing work the model would
   otherwise have to do, and the floor is lower *because*
   of the grammar. Record the gap. Scoring rule for the
   grammar-off arm, so it is not just a parse-failure count:
   score intent from the model's raw text by the same
   `intent_v2` rules where a valid JSON can be extracted,
   and count unparseable outputs as a separate category
   (parse failure vs wrong intent vs wrong args), not as
   a single "wrong" bucket.
3. **The verified core removes label noise.** Training labels
   come from `try_label` and its siblings, not from a human
   or a bigger model guessing. That lowers `E_task` directly:
   a cleaner corpus has a lower irreducible loss. If a
   student plateaus, check label noise before blaming size.

### 7.3 Single-source generation is the dependency, not the premise

The grammar-from-dictionary idea — one dictionary source
generating both training rows and eval cases, so training/eval
template drift cannot happen — is the reason this spec can
treat distillation data as cheap. It is not built yet.
`generate_dataset.py` and `generate_eval.py` still have
separate hand-written templates and do not read
`config/tools.yaml` (or the Rust core). The drift it would
have prevented has already happened twice in this project:
`isolate` meant `extract_audio` in training (26/26 corpus
rows) but `trim` in eval (3 cases), and `720p` appeared
0 times in training while eval expects 1280×720 twice
(see Appendix A). Until the generator is single-source,
the spec's ladder depends on the freeze rules and on
manual drift checks (template-id overlap, arg-pool overlap,
prompt/label collision checks — the dataset tests already
do the last of these). Record the generator version with
every dataset, and treat a drift check failure as a
stop-the-ladder event, not a footnote.
4. **Clarify is a capacity valve.** Every genuinely ambiguous
   input that becomes a clarify instead of a forced guess
   removes a bit of entropy the model would otherwise have
   to resolve by memorising a convention. Report the clarify
   rate alongside accuracy for every student; a student that
   "passes" by over-clarifying has not passed.

## 8. Open questions for the reviewer

1. Acceptance threshold (§5, §5.1): is ≥95% on `intent_v2`
   with the per-tool and keep/remove conditions right, or
   should the bar be the named teacher's `intent_v2` score
   exactly (not the historical 95.5% verification-only rate,
   which is not comparable — §1.1)?
2. Vocab trimming in GGUF/llama.cpp: feasible cleanly, or
   deferred? What is the exact output-token set (§4.1)?
3. First ladder: depth-only as in §6.2, or start with a
   vocab-trim + depth combined arm? What GPU budget is
   available for the ladder (each student is a distill +
   eval run)?
4. Distillation loss: KL with what temperature, and is the
   teacher's logit distribution saved, or only hard labels?
   (Logits need a transformers teacher, not llama.cpp — §4.3.)
5. MiniCPM5-1B as a control student: check config/tokenizer
   first — is it a valid control?
6. LoRA vs full fine-tuning for the pruned student (§6.3 step 2):
   is full FT of the surviving weights the plan?
7. Deployment target: which machine / RAM / latency is the
   student actually for? That decides whether GGUF bytes or
   tok/s is the binding constraint.
8. Is there a smaller teacher worth distilling from first —
   e.g. would a 1B teacher's floor be a better starting point
   than pruning the 2B directly?
9. **Teacher (§1.1):** stock, merged verified-v1 LoRA, or a
   fresh teacher with a seed sweep? Which one is the ladder
   measured against?
10. **Clarify (§4.1):** templated clarify (core generates the
    question, model outputs tool + `missing` only) or English
    output head kept? And is the 12% general mix kept in
    distillation data, dropped, or kept as a separate arm?
11. **Statistics (§5.1):** paired + seed-averaged + Wilson
    interval + release benchmark as co-primary — accepted?
    Or is a larger frozen eval worth building first?
12. **Full FT on the 3090 (§6.3):** 8-bit Adam, gradient
    checkpointing, a smaller student first, or a different
    machine? What is the memory plan?
13. **Ladder range (§6.2):** is the second phase (S5–S7,
    combined axes, sub-1B) in scope for this round, or does
    the first phase stop at S4 and report "still passing
    at 1.06B"?

## Appendix A. Context the reviewer needs without hunting the repo

This appendix is background, not a decision. It records where
MiniCut stood at the base commit, so the spec's claims can be
checked without re-reading the whole history.

### A.1 Pilot board (intent_v2, 200 cases, stock 167/200 in every run)

| Data | Seed | Tuned | Gains / regressions | Keep/remove (of 64) | New inversions | Unnecessary clarifies | Gate |
|---|---|---:|---:|---:|---:|---:|---|
| pilot v1 | 42 | 191 | 27 / 3 | 61 | paraphrase-008 | 0 | Fail |
| pilot v1 | 1 | 191 | 27 / 3 | 61 | paraphrase-016 | 0 | Fail |
| pilot v1 | 2 | 188 | 25 / 4 | 60 | paraphrase-006, -008 | 0 | Fail |
| pilot v1 | 3 | 189 | 24 / 2 | 61 | paraphrase-008 | 0 | Fail |
| verified v1 | 42 | 191 | 26 / 2 | 61 | — | 1 | Pass |
| verified v2 | 1 | 175 | 18 / 10 | 49 | — | 18 | Fail |
| verified v2 | 2 | 190 | 25 / 2 | 62 | — | 1 | Pass |
| verified v2 | 3 | 190 | 25 / 2 | 62 | — | 2 | Pass |
| verified v3 (near-miss) | 1 | 182 | — | — | none | 7 | — |
| verified v3 (near-miss) | 2 | 191 | — | — | none | 1 | — |
| verified v3 (near-miss) | 3 | 180 | — | — | none | 13 | — |

Readings from `eval/PILOT_BOARD.md`: verified labels remove
the keep/remove inversion at every seed (0/7 runs vs 4/4 for
v1 data) but trade it for over-clarifying on weak seeds; the
near-miss pairs (v3) did not stabilise it (two weak seeds of
three). The failure concentrates on one wording: the dev case
"cut the first 10 seconds off clip.mp4" (expected cut_out,
4 dev cases) flips to clarify in every weak verified run —
it is one word from the ambiguous template `v3_amb_cut_first`
("cut the first N seconds **of** X"). The phrasebook plan
(§1) is the proposed fix: define that wording out of the
ambiguous table rather than teaching a one-letter distinction
at 100 updates. LoRA recipe throughout: rank 16 / alpha 32,
100 updates, microbatch 2 × accumulation 2, ~112–128 s on the
RTX 3090, PyTorch peak 7.12/16.84 GiB. Stock and tuned served
the same Q4_K_M base with llama.cpp `01818e495`.

### A.2 Training data v1 (3000 examples)

`training/train_v1.jsonl`, seed 20261121 (deliberately not the
eval seed 20261120). Per-tool: trim 360, cut_out 400, concat
270, scale 270, speed 300, convert 300, extract_audio 270,
mute 120, clarify 350, general 360. Weak spots from baselines
v2/v3 are over-represented (cut_out inversions, speed/convert/
extract_audio arg names, clarify exact missing lists).
Contrast pairs: 38 trim + 38 cut_out with the same times.
Unit variety: seconds, mm:ss, short "12s", percentages and
halves resolved against fixture duration at generation time
(81 percentage/half, 115 mm:ss); resolutions as WxH, named
540p/270p, relative half/quarter size; speeds as factors and
percentages. Filenames: mostly `clip.mp4` (559/3000, the eval
convention), 30 filenames total. Fixtures cap at 30s, so
minute-style times ("1m30s") cannot occur in v1. See
`training/README.md`.

### A.3 Eval splits and freeze rules

Four frozen splits × 50, seed 20261120, generated before any
training data existed: seen paraphrase, unseen paraphrase
(held-out paraphrases, never in training), unseen combo
(reserved arg combos), unseen fixture (`long_60s_1080p` only,
a fixture never in training). Freeze rules: held-out template
ids never in training, reserved fixture never in training,
combo arg pools never in training, eval cases never sampled
into training, frozen JSONL files never edited after the
baseline run. Release benchmark: `eval/release/`, 38 pinned
cases, controls 38/38, no model scored yet as of the base
commit. See `eval/README.md`, `eval/freeze.json`,
`eval/release/README.md`.

### A.4 Template drift, twice

- `isolate`: 26/26 training corpus rows meant
  `extract_audio`, but 3 eval cases meant `trim`. All 3
  pilot regressions in the first LoRA pilot were this
  template mismatch, not model capacity.
- `720p`: 0 occurrences in training, eval expects
  1280×720 twice; both scale regressions in the first
  pilot were this.
Fix is contrast pairs in the generator / a single-source
generator (§7.3), not more updates on the same corpus.

### A.5 Rust / Verus core, in one paragraph

PR #4 merged (22 commits): Verus-verified `minicut-core` +
grammar/validate binary + CI. Review reproduced: cargo fmt
clean, cargo test 10 passed, pytest 55 passed, parity
20,000 cases 0 mismatches, Verus 21 verified 0 errors,
mutation 12/12 killed. Known gaps (non-blocking, recorded
in the review): `label()` is proved correct for any intent
including out-of-range ones, so the Phase 3 sampler must
generate in-range intents; the wire layer path-checks the
model's output field but Python ignores it (unlisted
divergence); `--batch` panics on malformed rows; parity
runs only seed 0 in CI. `kept_ms(Concat)` is a documented
placeholder. The Dafny dictionary (PR #3) is closed
unmerged at Ivy's direction; Rust/Verus superseded Dafny
for MiniCut. Muse only opens PRs, never merges — Ivy
merges herself.

### A.6 What "stock stays default" means in practice

At the base commit there is no promoted tuned adapter.
Any pruning experiment that assumes a tuned teacher must
first promote (or re-train) one and record it (§1.1).
Any experiment that assumes the phrasebook must first
implement it (§1). Any experiment that assumes a
single-source generator must first build it (§7.3).
Those three are the spec's hidden prerequisites — this
appendix names them so the reviewer can sequence them
instead of discovering them mid-ladder.

## 9. References

Checked / accessed 2026-11-26. ArXiv links are to the abstract
page; the PDF is at the same ID. Where a paper's constants have
been audited or revised, that is noted in the text, not hidden.

- Kaplan, McCandlish, Henighan et al. (2020). Scaling Laws for
  Neural Language Models. https://arxiv.org/abs/2001.08361
  — Power laws in N, D, C; `L(N) = (Nc/N)^αN`, `αN ≈ 0.076`.
- Hoffmann, Borgeaud, Mensch et al. (2022). Training
  Compute-Optimal Large Language Models (Chinchilla).
  https://arxiv.org/abs/2203.15556
  — `L(N,D) = E + A/N^α + B/D^β`, `E = 1.69`, `D ≈ 20·N`.
- Muralidharan, Sreenivas, Joshi et al. (NVIDIA, 2024).
  Compact Language Models via Pruning and Knowledge
  Distillation (Minitron). https://arxiv.org/abs/2407.14679
  — Depth/width/attention/MLP pruning + distillation
  retraining; 2–4× compression, up to 40× fewer tokens
  than from-scratch, +16% MMLU vs from-scratch.
- Sreenivas, Muralidharan et al. (2024). LLM Pruning and
  Distillation in Practice: The Minitron Approach.
  https://arxiv.org/abs/2408.11796
  — Depth vs width pruning on Llama-3.1 8B → 4B and
  Mistral NeMo 12B → 8B; width wins at the same size;
  teacher correction.
- Hinton, Vinyals, Dean (2015). Distilling the Knowledge
  in a Neural Network. https://arxiv.org/abs/1503.02531
  — Knowledge distillation, soft targets.
- Ma, Wang, Liu et al. (2024). The Era of 1-bit LLMs:
  All Large Language Models are in 1.58 Bits (BitNet b1.58).
  https://arxiv.org/abs/2402.17764
  — Ternary {-1, 0, 1} weights, 1.58 bits/param, trained
  from scratch; the far end of the bits axis (§4.4).
- Zhao et al. (2025; latest version 27 Feb 2026). MobileLLM-R1:
  Exploring the Limits of Sub-Billion Language Model Reasoners
  with Open Training Recipes. ICLR 2026.
  https://arxiv.org/abs/2509.24945
  — 950M matches/surpasses Qwen3-0.6B on reasoning with
  11.7% of its tokens; data quality shifts the curve left.
- Ushio & Zhou. Efficient Multilingual Language Model
  Compression through Vocabulary Trimming.
  https://arxiv.org/abs/2305.15020
  — Vocabulary trimming retains performance while shrinking
  the embedding matrix, which dominates small models.
- Efficient Vocabulary Reduction for Small Language Models.
  COLING 2025 Industry Track.
  https://aclanthology.org/2025.coling-industry.64.pdf
  — Embedding share 3% (70B) / 13% (8B) / 34% (1.8B);
  vocab to 8k cuts ~1B params from Llama-3-8B.
- COMPACT: Commontoken Optimized Model Pruning Across
  Channels and Tokens. NeurIPS 2025 Workshop.
  https://openreview.net/pdf?id=0iXlSGtiH8
  — Joint vocab + FFN pruning, common-token-weighted.
- OpenBMB. MiniCPM5-2B model card.
  https://huggingface.co/openbmb/MiniCPM5-2B
  — Config, 53.9 avg, RL+OPD, sampling `temperature=1.0,
  top_p=0.95, min_p=0.0`. Accessed 2026-11-26.

MiniCut-internal references (in this repo, at the base commit):

- `spec.md` — goal, base model, tool schema, runtime.
- `config/tools.yaml` — v1 tool registry, source of truth.
- `docs/rust-port.md` — Rust + Verus core, single-enum contract.
- `docs/plans/2026-11-25-phrasebook.md` — user grammar / phrasebook.
- `docs/templates.md` — generated template table.
- `docs/pre-gpu-plan.md` — pre-GPU plan and release gate.
- `eval/freeze.json`, `eval/PILOT_BOARD.md` — frozen eval, pilot results.
- `PROGRESS.md` — baselines v2/v3, pilot history.
