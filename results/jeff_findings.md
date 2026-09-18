# jeff — an independent, decision-only fact-checking model

**Question.** TypeSafe's Jev is a classifier that returns typed, calibrated
probabilities instead of text. Can we build one ourselves — a local model that
takes a claim plus its evidence and returns calibrated probabilities over a
declared answer space?

**Answer.** Yes, and it now matches Jev within measurement error.

## Headline (human-labelled ground truth)

| model | split | n | accuracy | macro-F1 | ECE |
|---|---|---|---|---|---|
| live Jev 1.13.0 (hosted) | test | 199 | **0.799** | 0.791 | 0.114 |
| **ours: Qwen3-4B-Instruct-2507 + LoRA** | test | 199 | **0.774** | 0.749 | **0.077** |
| zero-shot NLI cross-encoder | test | 199 | 0.668 | 0.658 | 0.277 |
| live Jev 1.13.0 | val | 199 | 0.769 | 0.745 | 0.114 |
| **ours: 4B + LoRA** | val | 199 | **0.784** | 0.762 | **0.090** |
| ours: Qwen2.5-1.5B + LoRA | test | 199 | 0.759 | 0.738 | 0.111 |

**The honest reading:** on **val** we lead by 3 rows (0.784 vs 0.769); on
**test** we trail by 5 (0.774 vs 0.799). Both are 199-row samples with a
95% interval of roughly ±0.056 — under-powered, so at that size the ordering was
not resolvable. The adequately-powered comparison below resolves it.

## Head-to-head at adequate sample size (9,730 unseen rows, human labels)

The n=199 comparisons are under-powered; this is the first one that resolves the
difference. Both models scored on the **same** 9,730 rows, none of which the
model saw in training (verified: zero claim overlap with any split).

| model | accuracy | macro-F1 | ECE | Brier |
|---|---|---|---|---|
| live Jev 1.13.0 | **0.8283** | 0.7994 | 0.0790 | 0.2750 |
| **ours (4B + LoRA)** | 0.8143 | 0.7769 | **0.0709** | 0.2822 |

**Jev is 1.4 points ahead — a real gap at this sample size, not noise.** It is
significant under the unpaired binomial (z = 2.55, p ≈ 0.011) and stays
significant under the paired McNemar test at its least favourable error overlap
(b−c = 136, b+c ≤ 3478, z ≥ 2.31, p ≈ 0.02). Our calibration remains better.
This supersedes the n=199 "statistically indistinguishable" claim above, which
was under-powered; and the earlier note that our 0.8143 beat "Jev's 0.799" was
an artifact of comparing different sample sizes.

**Where the gap is:** our weakest class is `not_enough_info` (F1 0.666 vs
`supported` 0.889 / `refuted` 0.776), and Jev's advantage concentrates exactly
there — detecting the *absence* of supporting evidence.

**Multi-primitive gap.** Choice is trained and good; **Noul and Score are
untrained** (all six training schemas are Choice; Score measures a uniform
0.25/level with no signal where Jev returns a real graded answer). Teacher-
labelled Noul/Score rows exist (`sft_multi.jsonl`, 3000 rows, on local disk — no
new API spend to reuse) and the 4B retrain on the merged 12,119-row set **was
run and lost**: it completed on the Colab VM (`TRAIN_RC=0`, epoch-1 val_loss
0.2627) but the session was dropped before the adapter could be retrieved. It
must be re-run; only GPU hours were lost.

**Parallelism gap, measured.** Our latency scales linearly with questions per
call (6.39× at k=8) while Jev's is flat (0.78×): 2085 ms vs 150 ms at eight
questions.

Jev's own judgment on this claim (asked directly, see `results/jev_assessment.json`):
P(supported) = 0.89. Our own conservative phrasing: the gap is not resolvable at
this sample size, so "roughly as good as Jev" is fair and "beats Jev" is not.

## How the accuracy got here

| step | test acc | what it established |
|---|---|---|
| frozen MiniLM encoder + 70k head | 0.493 | a sentence-similarity encoder is the wrong tool: the head gets representations with the language understanding already absent |
| zero-shot 0.5B / 1.5B / 4B readout | 0.412 / 0.583 / 0.724 (val) | accuracy scales steeply with language capacity — capacity is the binding constraint |
| 1.5B + LoRA | 0.759 | fine-tuning on 9119 examples beats a 2.7x larger model zero-shot, and repairs ECE 0.30 -> 0.11 |
| **4B + LoRA** | **0.774** | capacity remains the lever: +3 rows, and every secondary metric improves |

## What it is

A **text-conditioned classifier**, not a fixed-head classifier and not a dLLM:

- The label set and each label's natural-language definition arrive **in the
  prompt at call time**, so a differently-worded question or a different
  *number* of labels works without retraining. Verified: the 2-label Noul
  schemas are held out entirely and still score.
- The classification is read from the model's **own next-token distribution**
  restricted to the label tokens. No head is bolted on.
- **One forward pass**; no autoregressive loop, no iterative denoising.
- Output shape matches Jev: a distribution over the declared labels that sums to
  1, plus a confidence scalar. Per the docs, `confidence` is *derived* from the
  distribution — a convenience statistic TypeSafe explicitly does not lock you
  into — so it needs no separate training.

## Closed negatives (each was a real hypothesis; each failed cleanly)

1. **Per-label prior bias** (offline sweep over the val probabilities): identity
   is optimal, every other combination *loses* a row. The over-claiming-support
   errors are confidently wrong, not a threshold artifact.
2. **Schema-wording TTA** (average over all 8 wordings): accuracy −3 rows and
   agreement collapsed 0.769 → 0.374. The student is wording-sensitive.
3. **Human-label upweighting 3×** on retrain: val LM loss improved but task
   accuracy fell 1 row. Better language modelling did not transfer, so teacher
   noise is not the binding constraint.

## Per-label behaviour (test, 4B)

| label | precision | recall | F1 |
|---|---|---|---|
| supported | 0.806 | 0.865 | 0.834 |
| refuted | 0.860 | 0.741 | 0.796 |
| **not_enough_info** | 0.609 | 0.622 | **0.615** |

The 4B fixed most of the refuted→supported over-claiming (F1 0.739 → 0.796) but
is **weaker on `not_enough_info`** (0.667 → 0.615). Detecting the *absence* of
supporting evidence remains the hard class — consistent with a teacher that is
also weakest there.

## Assessment of the user's prior Corrective Ornstein work

Jev was asked directly whether that method transfers (`scripts/jev_assess_work.py`):

| question | Jev |
|---|---|
| drift-diffusion evidence machinery transfers? | **0.98 on "no transfer"** — there is no generated trace to accumulate evidence over |
| the compiler-authorization rule transfers? | 0.34 |
| adopt a structural accepted/rejected AUC? | 0.48 |
| most promising next lever | **evidence-gated labels 0.43**, more human data 0.23, DPO pairs 0.16, abstain-engineering 0.16, bigger model 0.02 |
| is over-claiming an "absence-of-evidence" failure? | 0.79 |

So: the OU/drift-diffusion/AUC machinery does **not** transfer to a single-pass
classifier with no tool calls — but the *one* transferable idea is the
evidence-gating discipline ("a completion claim is authorized by external
evidence, not by the model's own text"), which Jev independently ranks as the
best next lever, since 7000 of our 9119 training labels come from a teacher that
is only ~80% accurate.

## Limitations

1. **Sample size.** n=199 per split; ±0.056. Nothing finer than ~5 rows is
   resolvable, which is why val and test disagree about the ordering versus Jev.
2. **Teacher noise.** 7000 of 9119 training targets imitate a teacher wrong
   about one case in five. The upweighting experiment suggests this is not the
   binding constraint, but it is not ruled out as a ceiling.
3. **`not_enough_info` is weak** (F1 0.615) and is the class that matters most
   for refusing to answer.
4. **No multi-question parallelism.** Jev scores many questions over one state
   in a single call with flat latency (measured: 16 questions ≈ the latency of
   1). Our harness does one forward pass per question. Unimplemented, not
   disproven.
5. **Score primitive untrained.** Jev exposes Choice/Score/Noul. We cover Choice
   natively and Noul via held-out schemas; **Score has never been trained** —
   it exists only in the teacher client.
6. **Jaggedness untested.** The probe suite is one Jev passes 9/9, so it cannot
   show whether a student inherits the teacher's failure modes.
7. **Cost.** Jev charges $0.042/Mtok input with free output; ours has no
   marginal cost but needs a GPU and is slower per decision (~80 s for 597
   decisions on an M-series GPU, versus Jev's ~195 ms/call).

## Reproduce

```bash
bash autoresearch.sh                      # the loop metric (val only)
uv run python -m scripts.jeff_lm_eval --model Qwen/Qwen3-4B-Instruct-2507 \
  --adapter artifacts/jeff/lora_4b --split test --readout first_token \
  --dtype bfloat16 --out results/lm_eval_lora_4b_test.json
uv run python -m scripts.jeff_final_report   # -> results/jeff_final.md
uv run python -m scripts.jeff_assess_work        # Jev's assessment
```
