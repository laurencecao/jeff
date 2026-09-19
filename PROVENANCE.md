# Jeff 1 — provenance record

A measurement record for **Jeff 1**, an open-source, locally runnable replacement
for the hosted TypeSafe **Jev 1.13.0** fact-checking API. It is not a submission
and not a training-loss report. The machine-readable form of everything here is
`PROVENANCE.json`; that file is canonical and this README is generated from its
content.

Every Jev number below is **our own re-measurement** of the hosted API on our own
rows. None of it is comparable to any figure TypeSafe has published, and none of
it should be quoted as though it were.

## 1. Frame

Given a claim plus evidence passages, the model returns a calibrated
distribution over `supported` / `refuted` / `not_enough_info` — plus `noul`
(presence) and `score` (graded) primitives on the multi arm. The classifier *is*
the LM's own next-token distribution over the label tokens; there is no
bolted-on head, and label sets and wording arrive in the prompt at call time.

The keep rule throughout: **human-label accuracy and ECE, decided on the
9,730-row scale split.** Agreement-with-Jev is reported separately and never
folded into a quality claim.

## 2. Task

Scored artifact = `argmax(student_probs)` matched against the human gold label.
Two independent splits:

| split | n | gold file | our adapter | our preds | Jev preds |
|---|---|---|---|---|---|
| val (sealed) | 199 | `data/factcheck/ground_truth.jsonl` (`split='val'`) | `artifacts/jev_clf/lora_4b` | `preds_autoresearch_val.jsonl` | `preds_jev_val.jsonl` |
| test (sealed) | 199 | `data/factcheck/ground_truth.jsonl` (`split='test'`) | `artifacts/jev_clf/lora_4b_multi` | `results/lm_eval_4b_multi_test.json` | — |
| scale | 9,730 | `data/factcheck/eval_large.jsonl` | `artifacts/jev_clf/lora_4b_multi` | `preds_ours_large.jsonl` | `preds_jev_large.jsonl` |

The **test** row is a third, separate sealed holdout and is *not* the rows in the
headline val table. The server's `GET /v1/models` reports it as
`accuracy.test_n199 = 0.794` and `calibration.ece_test_n199 = 0.063`; both are
backed by `results/lm_eval_4b_multi_test.json` (split `test`, n=199, accuracy
0.79397, ECE 0.06344). Do not conflate these three 199-row splits — an earlier
draft of this record wrongly called that server figure unmeasured.

`ground_truth.jsonl` holds 1,589 `train` / 199 `val` / 199 `test`. An earlier
draft of the release notes called the headline val split `test`, which is wrong
and is corrected here; the `test` rows are their own holdout.

## 3. Box / harness

- **Training: Google Colab A100 40GB.** Training on the local Apple Silicon box
  is not viable for this corpus — a 4B LoRA at batch 4×2048 exceeds the 36 GB
  unified pool and OOM-killed at step 2.
- **Evaluation device as recorded:** `mps` (Apple M3 Max).
- **Readout:** `jev_clf/readout.py` — first-token iff all labels have distinct
  first tokens, else whole-sequence.
- **Harness:** an omp autoresearch loop over the `jev_clf` repo.
- **Off limits:** `scripts/jev_clf_autoresearch.py`, `scripts/jev_clf_lm_eval.py`,
  `jev_clf/schema.py`.

Protocol deviations, stated as flags with reasons:

| gate | state | reason |
|---|---|---|
| val underpowered | closed as a decision instrument | bootstrap 95% CI ±5.6 accuracy points (±11 rows), ~7× wider than scale's ±0.75; McNemar p=0.59; val and scale disagree on fix direction |
| choice readout bit-identity | **passed** | 12 rows, 0 argmax mismatches, worst prob delta 6.98e-08 (`results/probe_readout_regression_PASS.log`) |

## 4. Scoring surface

| metric | formula | direction | Jeff 1 | Jev 1.13.0 | verdict |
|---|---|---|---|---|---|
| val accuracy | `mean(argmax == gold)`, n=199 | higher | 0.7839 | 0.7688 | not significant (p=0.59) |
| val macro-F1 | mean per-label F1 | higher | 0.7620 | — | — |
| val Brier | `mean_i sum_k (p_ik - y_ik)^2` | lower | 0.3186 | — | — |
| val agreement w/ Jev | `mean(argmax_ours == argmax_jev)` | higher | 0.8518 | — | separate from quality |
| **scale accuracy** | `mean(argmax == gold)`, n=9,730 | higher | **0.8183** | **0.8283** | Jev ahead, significant |
| **scale ECE** | 10 equal-width pop-weighted bins, confidence = max class prob, same definition both models | lower | **0.0807** | **0.0932** | **Jeff 1 wins** |
| scale macro-F1 | mean per-label F1 | higher | 0.7789 | 0.7994 | — |
| scale Brier | `mean_i sum_k (p_ik - y_ik)^2` | lower | 0.2839 | 0.2750 | — |

**Paired tests.** Scale: ours 7,962 correct vs Jev 8,059; 497 discordant in our
favour, 594 against; z=2.94, **p=0.0033**, paired-diff 95% CI [−0.0165,
−0.0035]. Val: 156 vs 153; 17 vs 14 discordant; z=0.54, **p=0.59 — not
significant**.

## 5. Results — one row per run

| # | role | arm | change | val acc | scale acc | scale ECE |
|---|---|---|---|---|---|---|
| 1 | keep | 4B zero-shot | capacity baseline | — | — | — |
| 2 | discard | 1.5B + LoRA (Choice) | 9,119 rows, 2 epochs | 0.7588 | — | — |
| 3 | keep | **4B + LoRA (Choice)** | 9,119 rows; val champion | **0.7839** | 0.8143 | 0.0709 |
| 4 | keep | **4B + LoRA (multi)** | +1,800 Score +1,200 Noul, n=12,119; scale + demo arm | 0.7789 | **0.8183** | **0.0807** |
| 5 | keep | sequence readout for Score | capability fix (dead-uniform → ordinal) | 0.7839 | — | — |
| 6 | keep | calibration definition | reporting fix | 0.7839 | — | — |
| 7 | pending | 4B + LoRA soft-distill | CE + KL to the teacher distribution | see pre-registration | | |

Runs 3 and 4 are **different adapters**: 0.7839 (val) and 0.8183 (scale) are not
the same model. Run 5 is metric-neutral by construction — it repairs the Score
primitive, whose labels `" 0".." 3"` tokenize to `[220,15..18]`, all sharing
first token 220, so the first-token readout read a single logit four times and
returned a constant 0.25.

## 6. Retracted results

**The calibration figure had it backwards.**

- *What it looked like:* Jev was better calibrated than Jeff 1 on the scale
  split — Jev ECE 0.0790 vs ours 0.0807. This appeared in
  `results/calibration_plot.png` (and its served copy), the demo page, and
  `GET /v1/models`.
- *The tell:* Jev's stored `confidence` field is **not** the maximum class
  probability. Measured over the 9,730 scale rows it differs by up to **0.33**,
  mean **0.040**. The figure drew its ECE bars from that stored field while
  drawing its reliability curve from `max(prob)` — so the legend contradicted
  its own curve.
- *The control:* recompute **both** models from raw per-row predictions under one
  definition. Ours **0.08075**, Jev **0.09321**. Like-for-like, **Jeff 1 is the
  better-calibrated model.**
- *Adopted rule:* one confidence definition per comparison. Jev's self-reported
  0.0790 may be mentioned only alongside the statement that it is not comparable
  to our 0.0807.

The same audit surfaced two further defects, both corrected: stale figures on the
demo page and `/v1/models` (accuracy 0.8174 → 0.8183, ECE 0.0805 → 0.0807), and
the wrong split label (`test` → `val`).

## 7. Closed levers

Measured and rejected; each carries a mechanism-level reason so the next solver
does not spend compute rediscovering it.

- **`global-supported-logit-bias`** — a multiplicative bias moves the decision
  boundary but cannot repair confidently-wrong rows. Swept b=−0.6…+0.6: identity
  is optimal at 0.8183, every non-zero bias worse. Tuned on val it gains +2 rows
  there and loses 9 on scale.
- **`nei-probability-threshold-sweep`** — NEI mass exists but is not
  discriminative *at the margin*. Sweep t=0.05…0.96: best gain **2 rows**.
  Ranking AUC 0.8865, but 740/2,106 gold-NEI rows already have NEI as runner-up,
  and the rows a threshold would flip are currently correct.
- **`schema-wording-test-time-augmentation`** — averaging 8 wordings mixes
  incompatible first-token posteriors. −3 rows; agreement collapsed 0.769 → 0.374.
- **`human-label-upweighting`** — teacher noise is not the binding constraint.
  3× ground_truth: val LM loss improved, task accuracy −1 row.
- **`adapter-averaging`** — a 0.5/0.5 weight-space merge of adapters trained on
  different mixtures has no guarantee either skill survives; `lora_merged` was
  never evaluated, so shipping it would be an un-gated claim. Closed on absence
  of a gate.
- **`batching-in-the-benchmarked-eval-path`** — bf16 batched GEMM reduces in a
  different order. 4.43× faster (2159 → 488 ms/question) but first-token deltas
  up to 4.2e-03, enough to flip borderline rows. Opt-in demo accelerator only.

## 8. The measured gap: where it is, and what it is not

Jeff 1 over-claims on **both** classes, more often than Jev on both:
refuted→supported **12.3%** vs Jev's **6.2%**; not_enough_info→supported **24.4%**
vs Jev's **14.3%**. Being stronger on supported-class recall is a different
statistic and does not offset it. What the stratification below shows is *where
the over-claiming costs net accuracy* — the deficit lands in NEI recall, while
supported-class recall runs in our favour. It does NOT show we over-claim less.

By gold class, Jeff 1 makes **fewer** errors than Jev on supported claims and
**more** on weak evidence:

| gold class | our errors | Jev errors | delta |
|---|---|---|---|
| supported | 346 | 558 | **−212 (we are better)** |
| refuted | 534 | 506 | +28 |
| not_enough_info | 888 | 607 | **+281** |

Restricted to the **single-passage** rows that make up 88.8% of the split:

| gold class (n=8,643) | ours | Jev | rows |
|---|---|---|---|
| supported | 0.9469 | 0.9339 | **+60 us** |
| refuted | 0.8007 | 0.8080 | −18 |
| **not_enough_info** | **0.5229** | **0.6554** | **−208** |

On the 859 multi-passage rows (all `climate_fever`) we are **more** accurate than
Jev (0.6694 vs 0.5914).

Three distinct oracles, not to be conflated:

| restore | rows | accuracy | vs Jev (8,059) |
|---|---|---|---|
| NEI recall, single-passage rows only | +208 | 8,170/9,730 = **0.8397** | +111 ahead |
| NEI recall, **all** strata | +281 | 8,243/9,730 = **0.8472** | +184 ahead |
| **every** cell where we trail Jev (refuted included) | +311 | 8,273/9,730 = **0.8503** | +214 ahead |

So closing the `not_enough_info` deficit across all strata would on its own be
enough to pass Jev, without touching any other cell.

**What this does and does not mean.** These are oracle relabelling figures: they
reassign the model's own existing predictions to the right answer where it was
wrong. They measure **headroom**, i.e. that the surviving errors are concentrated
in one identifiable class rather than spread thinly. They do NOT show the trained
model can realize that headroom — doing so requires the model to produce
different predictions, which is exactly what a retrain would have to demonstrate.

Two controlled probes agree, each run against live Jev on the same rows:

- `scripts/probe_conjunction.py` — **ours 5/9, Jev 9/9.** We pass the
  falsified-conjunct arm (3/3) and the both-halves-true control (2/2). This is
  **not** a general "conjunctions break Jeff 1" rule — it fails specifically when
  a conjunct is **absent** (0/2 — we say `supported` at 0.67–0.72, Jev says
  `not_enough_info` at 0.99) or the claim **overstates** a measured number (0/2 —
  we say `supported` at 0.83–0.91, Jev says `refuted` at 0.87–1.00). The
  demo-shaped **tied** case failed too, so the failure is not limited to absent
  evidence. These 9 probes do **not** show we over-claim less often than Jev in
  general: on the scale split we over-claim on *both* classes —
  refuted→supported 12.3% vs Jev's 6.2%, and not_enough_info→supported 24.4% vs
  Jev's 14.3%.
- `scripts/probe_granularity_jev.py` — **ours 0/5, Jev 5/5**. Near-identical
  variants changed **both the wording and the passage structure**, so this probe
  does **not** isolate passage structure as the cause; wording and chunking moved
  together. What it establishes is **brittleness**: near-identical inputs flip
  the verdict, with confidence swinging 0.506 → 0.946. Jev is right at 0.99–1.00
  exactly where we are wrong.

## 9. Protocol

- One coherent change per run; nothing is kept without an independent check.
- Fail-closed: `jev_clf/eval.py` raises on zero matched pairs rather than
  reporting 0.0, and training asserts no `split='test'` row is ever loaded.
- The val split's noise floor (±5.6 points, ±11 rows) makes it a smoke test.
  **Only the 9,730-row scale split decides accuracy against Jev.**
- Never tune on val once scale has disagreed with it.
- ECE is only compared under one confidence definition, stated inline.

## 10. Files / sources

Cite the sealed prediction files, not restated decimals:
`data/factcheck/preds_{autoresearch_val,jev_val,ours_large,jev_large}.jsonl`,
`data/factcheck/{ground_truth,eval_large,sft_train_multi,sft_val}.jsonl`,
`results/{jev_clf_autoresearch_last,jev_clf_baselines,lm_eval_4b_multi_large,
lm_eval_4b_large,jev_large,probe_conjunction}.json`,
`results/probe_readout_regression_PASS.log`,
`artifacts/jev_clf/{lora_4b,lora_4b_multi}/train_metrics.json`,
`PREREGISTRATION_soft_distill.md`.

No latency, throughput, parameter-count, training-compute, or license claims are
made here beyond what a cited repo file states.
