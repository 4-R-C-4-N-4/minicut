# MiniCut pruning spec: finding the smallest model that can still do the job

Status: reviewed proposal — doc only. No training is authorized by this spec.
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

This first phase asks a narrower, measurable question: what is the
smallest **tested depth** that passes at full vocabulary, fixed width
and a fixed recovery recipe? The ladder stops at S4 (~1.10B total).
It does not establish a global minimum over vocabulary, width,
quantization or training budgets. Those are follow-up axes, not
scheduled arms.

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
  and the best observed tuned pilot is 192/200 (v5 seed 3), not a
  qualified reference score. Do not touch
  the frozen eval inputs in any experiment under this spec.
  See `eval/freeze.json`, `eval/PILOT_BOARD.md`, `PROGRESS.md`.
- Training data: generated from templates, with labels from the
  verified core — but *not yet* from a single source. As of the
  base commit, `training/generate_dataset.py` and
  `eval/generate_eval.py` still have separate hand-written
  templates and do not read `config/tools.yaml`; the
  dictionary-as-generator work that would make them one source
  is not yet built (see §7.1 and Appendix A). The claim in §6
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
  - The 14 pre-v5 pilots, scored on the same grader, span 175–191/200.
    Verified v1–v3 added no inversions in seven runs, but v4 added
    inversions at two of three seeds. The v5 short-clarify sweep scored
    **179 / 188 / 192**, with **13 / 1 / 0** unnecessary clarifies and gates
    **pass / fail / pass**. Seed 2 added two inversions. V5 also reshuffled
    the tool sample (1,261 of 2,009 tool rows shared with v4), so this is
    not an isolated test of question length. See Appendix A and
    `eval/PILOT_BOARD.md`. Stock remains the deployment default. V3 passed the pilot gate at
    all three tested seeds, making it a reference candidate, not an
    adopted reference. Its range was 180–191 and it has not been
    qualified on the release benchmark.

There is no accepted tuned teacher. Before scheduling the ladder, select a
2B recipe that passes the existing pilot gate at each of at least three
predeclared seeds, then pin a reference artifact using a selection rule
written before evaluation (not the best development score). Record the HF
revision, adapter/data hashes and seeds, merge status, GGUF hash and quant,
grammar, server and decoding. Re-score the exact artifact on the frozen
`intent_v2` grader and independent release benchmark. The observed stock
167/200 is for Q4_K_M, not a measured F16 score. Use stock as a diagnostic
control, not as a weaker substitute for a qualified teacher.

If the chosen reference uses LoRA, merge it into its HF base before slicing
tensors when initializing students from that checkpoint; verify the merged
reference separately. Do not apply the original 42-layer adapter to a pruned
base. Keep the full reference unpruned for comparisons and any later KL arm.

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

What *is* measurable is a tested bracket under §5: the smallest
observed passing depth and, if tested, the next smaller failing
configuration. A fit on held-out loss (§2.2) can describe the trend,
but neither establishes the bracket nor determines what ships.

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

`L_task` is mean teacher-forced negative log-likelihood in nats per
supervised assistant token (including the same end-of-response convention)
on a fixed, held-out task corpus. Use the same tokenizer, prompts, masking
and scoring implementation for every depth point. Report separate loss for
tool calls and clarifications; do not fit on the 200-case accuracy. Freeze a
sufficiently large and diverse generated holdout before training (§6.1).
Use a separate holdout from the loss set for early stopping if needed.

`N` counts non-embedding parameters for the fixed-vocabulary depth series;
report total parameters and GGUF bytes separately. The fitted `E_task` is
an empirical asymptote for this recipe, not an entropy measurement, a
proven lower bound, or the finite teacher's loss. Fit only if there are
more distinct sizes than free parameters; check stability across seeds and
against an unused size before interpreting the curve. Do not extrapolate a
minimum or map NLL to task accuracy without evidence. The decision is the
smallest *measured* candidate that passes §5, not the fitted knee.

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

Implications for the first ladder:

1. **Depth is the first controlled variable.** Each layer is ~47M
   non-embedding parameters; 42 → 24 saves ~850M at fixed width and vocab.
2. **Width is a later axis.** The MLP is ~80% of each layer, so reducing
   intermediate width from 6144 to 3072 halves the MLP, not the whole model.
3. **Vocabulary is a large fixed cost, not a free saving.** The two
   vocabulary matrices total ~535M parameters. Trimming them is deferred
   because tokenizer IDs and GGUF tensor shapes must agree (§4.1).
4. **Quantization is separate.** The recorded pilots use Q4_K_M in
   llama.cpp; 5.03 GB is the approximate 16-bit weight payload, not the
   deployed GGUF size. Compare any new quantization with its own
   same-quant reference (§4.4).

## 4. Trimming levers, in order

### 4.1 Vocabulary trimming (analysis only; deferred for deployment)

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

MiniCPM has untied input and output matrices (~267.4M parameters each).
The Llama GGUF loader sizes both from the tokenizer vocabulary and rejects
mismatched tensor shapes (`llama-model-loader.cpp`, shape check; inspected
at llama.cpp `01818e495`). Dropping output rows alone is not a stock GGUF
option. A shared reduced vocabulary needs a remapped tokenizer, consistent
BPE merges and special IDs, and matching input/output rows in a custom
conversion pipeline. Current LoRA targets only attention and MLP
projections; that removes one adapter-shape concern, not the tokenizer
or loader constraints.

The grammar's fixed literals are not the output-token set. `input`,
`output` and `clarify.question` are free strings in both the current grammar
and the Rust strict grammar. Filenames, decimal forms, escaping and English
questions require more than punctuation and enums. A token census of the
training split and a separate held-out corpus may characterize observed
coverage and fallback expansion; it cannot establish a safe production
allowlist. Never build that allowlist from frozen dev or release cases.
Keep the output contract and full tokenizer in the first ladder. Templated
clarification would be a separate contract change, not a prerequisite.

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
  layer, measure loss on a ranking-only calibration split; or use
  activation or gradient magnitude on task data), do not drop uniformly, and record
  *which* layers were dropped: dropping the last N layers,
  dropping evenly spaced layers, and dropping the
  importance-ranked lowest layers are three different
  experiments with different results. The ladder in §6.2
  assumes importance-ranked dropping; if the reviewer picks a
  different rule, the ladder's sizes stay the same but its
  results test a different pruning policy. If initializing from a LoRA-tuned
  reference, prune its merged HF weights, not the unmerged adapter (§1.1).
- **Width pruning:** remove attention heads, MLP intermediate
  channels, and/or hidden channels. Finer-grained, usually
  better accuracy at the same parameter count, but needs
  surgery on every matrix that touches the pruned dimension
  and a retraining pass to recover.
- **First ladder:** importance-rank → prune depth → recover with
  verified labels → measure. Change one axis at a time (§6).

### 4.3 Student recovery: verified labels first, KL as a separate arm

Train the first depth series with supervised hard labels from the verified
core and reviewed templates. This controls the supervision independently
of an imperfect teacher: it does not copy incorrect predictions or inherit
an unmeasured teacher's uncertainty. Proofs constrain the executable edit
label, not whether a template matches arbitrary natural language. Do not
use teacher-generated hard labels as a drop-in replacement.

Optional later arm: add KL to the same hard-label objective with a pinned
teacher and unchanged student/data/step budget. Start at `T=1` as an
experimental default; preregister the loss weight and any temperature
sweep. Compute teacher-forced logits offline on the same answer prefixes
using HF/PyTorch (the current llama.cpp evaluation path does not capture
full logits). Record masking, normalization, precision and artifact hashes.
For compressed top-k caches, specify how omitted probability mass is
handled and label the resulting loss approximate. Cache offline to avoid
requiring both models resident on the 3090. No KL or cache build is assumed
for the initial ladder. See Hinton et al., https://arxiv.org/abs/1503.02531.

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

Use one pinned teacher **artifact** (§1.1), not a seed average or a
different teacher per seed. Save its per-case verdicts, inversion case
IDs and unnecessary-clarification count with the artifact hash and
grader/runtime identifiers; every student seed compares to that same
record. Proposed acceptance policy, to approve **before** allocating GPU time:

- **Development:** on the 200 frozen `intent_v2` cases, evaluate at least
  three predeclared student seeds (start with 1–3). The mean may be at most
  **4 cases (2 percentage points)** below the reference. Also apply that
  margin to each seed as a stability guard; report the mean and range.
  For illustration, a pinned reference scoring 192 would set a floor
  of 188 for both the mean and each seed. The relative rule deliberately
  replaces the old 190-case absolute bar; 192 is not a selected reference.
- **Semantic guardrails:** no new keep/remove inversion on any seed
  (the student's inversion IDs must be a subset of the reference's);
  each seed's unnecessary-clarification count must not exceed the
  fixed reference count. Report per-tool and per-split
  counts and paired gains/regressions, not a 2-point per-tool cutoff on
  buckets too small to resolve it. Review clarification wording separately
  where it affects usability; `task_pass` grades the missing fields.
- **Release:** use the independently authored 38-case pinned benchmark
  as a separate gate on the finalist and reference, not for tuning or
  choosing checkpoints. Run `python -m eval.release controls --report
  "$TMPDIR/release-controls.json"` first and keep the report (use a writable
  scratch directory for `TMPDIR`).
  Require no new release intent failure or source-preservation failure
  relative to the reference, and no source-preservation failure at all.
  Report disagreements between the dev and release suites separately.
- **Deployment:** report GGUF bytes, peak runtime RAM/VRAM, cold start,
  time to first token and decode throughput at fixed context/offload and
  on the agreed target. A smaller file alone is not a deployment win.

### 5.1 What the statistics can and cannot establish

The margin is an operating rule, not a claim of statistical non-inferiority.
Use the same case IDs per arm and report discordant case counts. Repeated
prompts across splits are correlated, and three runs on the same 200 cases
are not 600 independent observations. If reporting uncertainty, resample
paired prompt groups (with all corresponding cases), report seed variation
separately and label the interval exploratory; do not use overlap of
unpaired Wilson intervals to infer equivalence. A borderline result merits
more independent cases or seeds, not a retroactive change to the gate.

Name the smallest measured passing configuration and the next smaller
measured failure, if any. If every measured size passes, report the lower
end of the tested range; if none pass, report no bracket. Do not claim a
universal minimum or require an extrapolated knee to stop.

## 6. The experiment: a student ladder

### 6.1 Data (fixed for all students)

- **Training and layer ranking:** use versioned Rust-generated tool rows
  with verified labels and reviewed templates; pin generator commit, template
  pool, seeds, fixture distribution, mix and exact row hashes. Keep the
  first ladder tool-only (including `clarify`). The historical 12% general
  mix belongs to `train_v1.jsonl`, not the pilot data; any general mix is
  a separate experiment.
- **Loss holdout:** create a separate, frozen generated split with fresh
  seeds and disjoint prompt groups from training, including held-out template
  families where feasible. Check canonical prompt/target collisions as well
  as template/fixture coverage; a different seed alone is not separation.
- **Freeze:** enforce `eval/freeze.json` via `training.build_verified` and
  audit against the pinned release corpus separately. Exclude both benchmark
  rosters and their held-out shapes from training, layer ranking, checkpoint
  selection, quantization calibration and any teacher-generated data or
  logits. The development set measures the ladder; the release set is for
  the finalist. A collision stops the experiment. Record checks and hashes
  before any training; no changes to either benchmark under this spec.

### 6.2 Ladder (illustrative sizes — arithmetic, not results)

Depth-only first, because it is the cleanest attribution.
All sizes below are arithmetic from §3 (47.2M non-embedding
per layer, 535M embeddings at full vocab), before any training.
Vocab is held at full size in this table so depth is the only
variable; vocab trimming is a separate axis (§4.1).
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

The initial depth-only ladder stops at S4 (~1.10B total). Width variants
above and the combined variants below are illustrative follow-ups, not
scheduled arms. Even if S4 passes, that does not establish a sub-1B result:

| Student | Layers | Intermediate | Non-embedding | Total (full vocab) | Note |
|---|---|---|---|---|---|
| S5 | 12 | 3072 | ~0.340B | ~0.875B | depth + width |
| S6 | 12 | 2048 | ~0.264B | ~0.799B | depth + width, narrower |
| S7 | 8 | 3072 | ~0.226B | ~0.761B | depth + width, shallow |

All retain ~535M vocabulary parameters; vocabulary trimming is not needed
to *specify* a sub-1B candidate, only a possible later way to save more.

Per-layer width arithmetic: MLP at intermediate 3072 is
3 × 2,048 × 3,072 = 18.9M; plus attention 9.4M = 28.3M/layer.
At intermediate 2048: MLP 12.6M + attention 9.4M = 22.0M/layer.

The reviewer should sanity-check these numbers against the
actual tensors before running anything — they are config
arithmetic, and a wrong intermediate size in the table is a
cheap error to catch now.

An existing smaller pretrained model is a useful control, not a
from-scratch baseline. MiniCPM5-1B availability, config, tokenizer and GGUF
support are unverified; do not assume it exists or shares the tokenizer.
If a compatible control is found, tune it on the same rows and score it
under the same runtime contract. Do not mix its per-token loss into the
MiniCPM depth fit if its tokenizer differs.

### 6.3 Procedure per student

1. Rank layers on the calibration/training side of §6.1, not the frozen
   eval. Pin the layer selection method and exact retained indices. Start
   with S2 (24 layers) as a smoke test, not a promoted result.
2. Recover with verified hard-label SFT under one pinned recipe and seed
   roster. Full FT is optional, not a prerequisite. Choose the memory plan
   **before** the timed smoke; if full FT does not fit, try activation
   checkpointing, a memory-efficient optimizer, or a separately labelled
   LoRA recipe on all compared depths. BF16 weights and gradients plus
   FP32 Adam moments alone can use ~12 bytes/parameter before activations,
   allocator overhead or master weights. The measured LoRA pilot's memory
   does not justify a full-FT run on a 24 GiB 3090.
3. Export to GGUF at the same quantization as the reference (Q4_K_M for the
   existing runtime); measure F16 or Q3 separately, with a same-quant
   reference. Pin llama.cpp revision, GGUF/grammar/prompt hashes, server
   flags, context/offload, thinking state and actual sampler parameters.
   Existing pilots used revision `01818e495` and `temperature=0.0`
   (`minicut/llm/client.py`); the model card's sampling recommendations
   are not the pilot configuration.
4. Measure held-out loss and evaluate on the pinned paired dev roster.
   Expand only after the smoke confirms export, memory headroom and cost.
   The existing 100-step LoRA pilots took about 8.5 min each including
   conversion and both eval arms; this is not a full-FT estimate. Proposed
   first allocation is one S2 smoke capped at 100 updates and 30 minutes
   wall time (including conversion and dev eval). This checks export,
   memory headroom and throughput; the dev score is diagnostic only.
   A low smoke score can reflect undertraining and is not evidence that
   24 layers are insufficient. Pin the actual recovery budget before
   drawing conclusions from the ladder. Then forecast the cost
   of the 4 depths × 3 seeds from measured throughput, peak VRAM and
   eval runtime; get approval for the remaining GPU-hours before queuing.
5. Apply §5. Report a measured bracket if found; fit §2.2 only when data
   supports it. Use the independent release benchmark only for the
   nominated finalist and the pinned reference.

### 6.4 What success looks like

- A smallest passing tested student (or an explicit no-pass result), a
  measured bracket if possible, and paired case counts, seed range and
  deployment measurements.
- A reproducible comparison that changes depth alone. Any curve is a
  secondary descriptive result; an unidentifiable fit is not a blocker.

## 7. Role of the grammar / Verus core in the floor

This is the part that makes MiniCut's floor lower than a
general model's floor, and it should be measured, not asserted:

1. **Canonicalisation lowers input entropy.** Measure it:
   token entropy / distinct surface forms per canonical intent
   before vs after phrasebook canonicalisation, on the
   training corpus. The phrasebook (`docs/plans/2026-11-25-phrasebook.md`)
   and the template table (`docs/templates.md`) are the
   artefacts to measure against.
2. **The output grammar removes syntax learning.** GBNF constrains JSON shape, field names and arg types at inference; it
   does not choose the intended tool or ensure correct arguments. The ablation that proves this is cheap
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
   or a bigger model guessing. This can reduce label noise; it does not establish a lower `E_task`. If a
   student plateaus, check label noise before blaming size.

4. **Clarify is a capacity valve.** Every genuinely ambiguous
   input that becomes a clarify instead of a forced guess
   removes a bit of entropy the model would otherwise have
   to resolve by memorising a convention. Report the clarify
   rate alongside accuracy for every student; a student that
   "passes" by over-clarifying has not passed.

### 7.1 Single-source generation is not a prerequisite

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

## 8. Review decisions and launch prerequisites

The proposal adopts teacher-relative acceptance (§5), verified hard-label
SFT first (§4.3), and a full-vocabulary, depth-only first phase (§6).
Vocab surgery, a clarify contract change, KL, mixed general-text training,
width and sub-1B arms are deferred. The smaller pretrained checkpoint is
an optional independent control, not a teacher prerequisite.

Still required before using the GPU: approve the numerical margin and
stability guard; qualify and pin the reference teacher; approve the first
smoke and the memory recipe, then approve remaining GPU-hours after the
smoke. Confirm the deployment target and whether RAM or latency is the
binding budget before promoting a winner. A doc merge is not authorization
to start training.

## Appendix A. Context the reviewer needs without hunting the repo

This appendix is background, not a decision. It distinguishes the base
commit from later experimental branches and locally verified runs.

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
| verified v3 (near-miss) | 1 | 182 | 23 / 8 | 54 | — | 7 | Pass |
| verified v3 (near-miss) | 2 | 191 | 26 / 2 | 62 | — | 1 | Pass |
| verified v3 (near-miss) | 3 | 180 | 24 / 11 | 57 | — | 13 | Pass |
| verified v4 (phrasebook) | 1 | 180 | 17 / 4 | 56 | paraphrase-012 | 8 | Fail |
| verified v4 (phrasebook) | 2 | 186 | 25 / 6 | 58 | — | 1 | Pass |
| verified v4 (phrasebook) | 3 | 186 | 23 / 4 | 61 | paraphrase-014 | 7 | Fail |
| verified v5 (short clarify) | 1 | 179 | 19 / 7 | 62 | — | 13 | Pass |
| verified v5 (short clarify) | 2 | 188 | 24 / 3 | 60 | paraphrase-012, -014 | 1 | Fail |
| verified v5 (short clarify) | 3 | 192 | 27 / 2 | 62 | — | 0 | Pass |

V4 and v5 results are post-base-commit evidence from the phrasebook and
short-clarify branches respectively. Neither recipe has consistent
inversion and clarification behavior across its seed sweep. In the frozen
`eval/eval_unseen_paraphrase.jsonl`, `eval_unseen_paraphrase-012` is
“clip.mp4 without the part from 5 to 10 seconds” and
`eval_unseen_paraphrase-014` is “take the 0 to 5 second bit out of
clip.mp4”; these match the v4 pilot-board descriptions.
The pilots used 100 updates with LoRA rank 16 / alpha 32 and microbatch
2 × accumulation 2; stock and tuned served the same Q4_K_M base with
llama.cpp `01818e495`.

#### Sweep provenance (v3–v5)

The nine runs below were rechecked on CPU with
`python -m training.factory board --run training/output/<run>`.
The board recomputes the gate from the stored graded rows; unnecessary
clarifications come from `unnecessary_clarifications.final` in the
comparison report. These are local artifacts, not files shipped in the
PR. The relative paths and SHA-256 hashes pin the exact reports behind
the Appendix A figures; rechecking the original runs requires access to
those artifacts.

| Run under `training/output/` | SHA-256 of `<run>/comparison.json` |
|---|---|
| `sweep-verified_v3-s1` | `9da70a6aae418ab47f6429b8d9cf096f711a28a43f2df687478d00cc7e639a64` |
| `sweep-verified_v3-s2` | `3fd29e884e946daa3d408215f01941ea19759e770235808b7b751a6d7191b05f` |
| `sweep-verified_v3-s3` | `90944e9e9a54ebbdb00b41cab36e5ba19433b74178e69fe545fcf0ff3e58c536` |
| `sweep-verified_v4-s1` | `9e712b2924d7b6ca7ade17c9408f045bc0c31ad9580d79bba05ec00545846ac4` |
| `sweep-verified_v4-s2` | `833e96ba3065642a4fd2fd97ec4a202dafaaf8c3aee4ca1a9fdc48704985609e` |
| `sweep-verified_v4-s3` | `dd408f875a52edc8a5dd9d666f5126d672e72b3a12641ec92f861f99fb113d83` |
| `sweep-verified_v5-s1` | `aba92aede2134615e86ed447cb17c84ca22d13f5ac990d408a209eda97e2915a` |
| `sweep-verified_v5-s2` | `f6854ffd94a64dd93804b0c416fd70bd141135555e7057f59f885707e78065dc` |
| `sweep-verified_v5-s3` | `6deaf71548e61815e89166fef6c7699c185fa44dce3c17f77dc7325780a835e2` |

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
  `extract_audio`, but 3 eval cases meant `trim`. One regression in the
  first LoRA pilot was an inversion on an `isolate` prompt; the other
  two were scale regressions. This does not isolate model capacity.
- `720p`: 0 occurrences in training, eval expects
  1280×720 twice; both scale regressions in the first
  pilot were this.
Fix is contrast pairs in the generator / a single-source
generator (§7.1), not more updates on the same corpus.

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
Qualify a reference without changing the deployment default (§1.1).
The phrasebook exists on its experiment branch; the v4/v5 results do not
qualify either recipe. The first ladder does not require a new single-source
generator; it requires the explicit freeze and drift checks in §6.1.

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
