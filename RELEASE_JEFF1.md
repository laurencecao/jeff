# Jeff 1 — Release Notes

Jeff 1 is an open-source, locally runnable replacement for TypeSafe Jev 1.13.0, the
hosted fact-checking judgment API. Given a claim and a set of evidence passages, it
returns typed judgments over the same request shape Jev clients already speak: a
labeled **choice**, a presence judgment (**noul**), or a graded **score**. The model
is `Qwen/Qwen3-4B-Instruct-2507` (Apache 2.0 — license evidence in the model card)
plus a LoRA adapter, running entirely locally on Apple Silicon (MPS). No hosted
service is involved at inference time. This note covers the repository state at
commit `527f2a4`.

All figures below are reported to 4 decimal places. Every figure stated here was
either measured on the two documented splits or read from a file in this repository
(cited inline where applicable). No latency, parameter-count, or training-compute
claims are made.

## Results

Two splits, measured like-for-like on the same rows:

| Split                | n      | Model              | Accuracy | ECE    |
|----------------------|--------|--------------------|----------|--------|
| Validation (sealed)  | 199    | Jeff 1             | 0.7839   | 0.0902 |
| Validation (sealed)  | 199    | live Jev 1.13.0    | 0.7688   | 0.1137 |
| Scale (human labels) | 9,730  | Jeff 1             | 0.8183   | 0.0807 |
| Scale (human labels) | 9,730  | live Jev 1.13.0    | 0.8283   | 0.0932 |

Additional metrics for Jeff 1 on the sealed validation split:

- macro-F1: 0.7620
- Brier score: 0.3186
- Agreement with live Jev 1.13.0: 0.8518

### The validation split is sealed

The validation split is the sealed holdout: 199 questions carried in
`data/factcheck/ground_truth.jsonl` with `split='val'` (the file holds 1,987 rows:
1,589 `train`, 199 `val`, 199 `test`). The 199 `split='test'` rows are a
*different* sealed holdout and are not the rows scored above; the metrics in the
table come from the 199 `split='val'` rows, which is what the loop harness
measures and what the recorded per-row predictions
(`data/factcheck/preds_autoresearch_val.jsonl`) cover. Note the wording trap: the
prediction rows carry no `split` field, and the eval entrypoint accepts
`--split test`, so an earlier draft of this note mislabelled the val split as
`test`. The row IDs are what identify the split; they are all `val`.

The ground-truth file is assembled from four source benchmarks — vitaminc,
scifact, climate_fever, fever — at roughly 500 labels per source. No tuning
reported in this note used the sealed validation target.

### The scale split

The scale split is `data/factcheck/eval_large.jsonl`: 9,730 questions, all with
human labels. Source breakdown: vitaminc 3,979, fever 3,910, scifact 982,
climate_fever 859.

### How it was measured

- **Accuracy** is exact match of the predicted label against the human gold label.
- **Live Jev** means the hosted Jev 1.13.0 API called on the identical rows.
- **Calibration (ECE)**: 10 equal-width bins, population-weighted. On the scale
  split, *confidence* is defined as the maximum class probability — identical
  definition for both models — so 0.0807 vs 0.0932 is a like-for-like number and
  Jeff 1 is the better-calibrated model.
- **Caveat (must accompany any mention of Jev ECE):** Jev additionally reports its
  own internal confidence statistic; scored under the same binning that
  self-confidence scores 0.0790. That figure is **not comparable** to the table
  above. The two confidence definitions disagree by up to 0.33 (mean divergence
  0.040), so Jev's 0.0790 must never be read as Jev beating Jeff 1's 0.0807.
  (The demo page carries this caveat, `scripts/jeff_demo_page.py`.)
- In-repo corroboration: `results/jeff_final.md` carries the validation rows at 3
  decimal places (the roundings of the 4-digit values above), each row naming its
  source prediction file. The scale figures also appear, matching this table, in
  the demo page's in-repo results table (`scripts/jeff_demo_page.py`) and in the
  server's `GET /v1/models` payload (`scripts/jev_clf_server.py`).

### Head-to-head read

**The validation split is too small to establish a lead, and must not be quoted
as one.** Jeff 1 scores 0.7839 vs Jev's 0.7688 there, but that is 3 rows out of
199. A paired McNemar test on the same rows gives 17 discordant pairs in Jeff 1's
favour and 14 against, z=0.54, **p=0.59** — not significant. A bootstrap 95% CI
on the validation accuracy spans ±5.6 points (±11 rows), roughly 7× wider than
the scale split's ±0.75 points. Treat the validation split as a smoke test only.

On the 9,730-row scale split the picture is stable, and the accuracy difference
there *is* significant: Jev is more accurate (0.8283 vs 0.8183), a paired McNemar
z=2.94, **p=0.0033**. Jeff 1 is better calibrated (0.0807 vs 0.0932). So the
honest summary is: **Jeff 1 is a calibration win and a small but real accuracy
loss on the split large enough to measure it.**

That gap is concentrated, not diffuse: the bulk sits in `not_enough_info` recall
on single-passage rows (see Failure Modes), and on multi-passage rows Jeff 1 is
*more* accurate than Jev.

Restoring our recall to Jev's level in various cells would add:

| restore | rows | accuracy | vs Jev (8,059) |
|---|---|---|---|
| NEI, single-passage rows only | +208 | 8,170/9,730 = **0.8397** | +111 ahead |
| NEI, all strata | +281 | 8,243/9,730 = **0.8472** | +184 ahead |
| every trailing cell (refuted included) | +311 | 8,273/9,730 = **0.8503** | +214 ahead |

These are **oracle relabelling** figures — they reassign Jeff 1's existing wrong
predictions to the right label. They measure **headroom**: the surviving errors
sit in one identifiable class rather than spread thinly. They do **not** show the
trained model can reach those numbers; that would require it to emit different
predictions, which only a retrain can demonstrate.

**But do not read this as "not over-claiming."** Jeff 1 over-claims on **both**
classes, and more often than Jev on both: it answers `supported` for **12.3%** of
gold-`refuted` rows against Jev's **6.2%**, and for **24.4%** of
gold-`not_enough_info` rows against Jev's **14.3%**. Higher supported-class recall
is a different statistic and does not offset that.

One further caution for anyone tuning this model: the two splits disagree on the
direction of a fix. Applying a bias to the `supported` label gains 2 rows on the
sealed validation split while *losing* 9 rows on the scale split. Do not tune
against the 199-row split.

## Capability parity with Jev

Jev exposes three typed judgment primitives. Jeff 1 implements all three, with
label sets and label wording specified in free text **at call time** — the model
is plainly text-conditioned, not a fixed-output classifier:

- **choice** — pick one label from an arbitrary caller-supplied set; returns
  per-label probabilities and a confidence.
- **noul** — a presence judgment over the evidence (e.g. "Does the evidence
  contain a date or a number?").
- **score** — a graded judgment over an arbitrary level ladder, with per-level
  probabilities and per-level criteria text.

Parity was verified in a live call exercise covering: a 2-label binary choice, a
5-label multi-word choice, a noul question, a 3-level score, a 5-level score, and
the 3-label fact-check verdict (supported / refuted / not_enough_info).

## Known failure modes

Written candidly, because they are the honest way to use this model:

- **Over-claiming is the dominant error.** On the scale split, Jeff 1 predicts
  'supported' on 12.3% of questions whose gold label is 'refuted' and on 24.4% of
  questions labeled 'not_enough_info'; live Jev's corresponding rates are 6.2% and
  14.3%. Jeff 1 is systematically more willing to call a claim supported.
- **Worked example (over-claiming).** Claim: "The new training program made
  participants both faster and more accurate than standard training." Evidence:
  participants were faster (42s vs 55s) but accuracy was tied (91% in both
  conditions). The claim is conjunctive, so the gold label is 'refuted' — the
  "more accurate" half is false. Jeff 1 returned 'supported' at confidence 0.796,
  riding the single true conjunct.
- **Adversarial jaggedness probe.** Jeff 1 scores 8/9; live Jev 1.13.0's reference
  score is 9/9 (recorded in `scripts/probe_jaggedness.py`). The single Jeff 1
  miss is the counting probe: gold 'refuted', answered 'not_enough_info' at
  confidence 0.639.
- **The gap is `not_enough_info` recall, and it is concentrated.** On the 9,730-row
  scale split, errors by gold class show Jeff 1 is *better* than Jev on supported
  claims (346 errors vs 558) and worse where evidence is weak: 888 errors on gold
  `not_enough_info` vs Jev's 607 (+281), which is the whole gap. Restricted to
  single-passage rows (8,643 of 9,730) the picture is sharp — supported recall
  0.9469 vs 0.9339 (we win), refuted 0.8007 vs 0.8080 (even), `not_enough_info`
  **0.5229 vs 0.6554** (we lose 208 rows). On the 859 multi-passage rows Jeff 1 is
  *more* accurate than Jev (0.6694 vs 0.5914). The defect is **located**, not
  excused: we treat a topically-relevant passage as if it entailed the claim, and
  that *is* over-claiming — Jeff 1 over-claims on both classes and more than Jev
  on both (refuted→supported 12.3% vs 6.2%; NEI→supported 24.4% vs 14.3%). What
  the stratification adds is *where* it costs accuracy: the net deficit lands in
  NEI recall. Oracle relabelling adds, by scope: 208 rows on single-passage NEI
  (→ 0.8397, +111 vs Jev), 281 rows on NEI across all strata (→ 0.8472, +184), or
  311 rows across every trailing cell (→ 0.8503, +214). These are headroom
  figures, not achievable-accuracy claims — see the Results section.
- **Controlled probes locate the failure (both run against live Jev on the same
  rows).** `scripts/probe_conjunction.py`: Jeff 1 5/9, Jev 9/9. Note the shape of
  that result — it is **not** a general "conjunctions break Jeff 1" rule. Jeff 1
  passes the explicitly falsified-conjunct arm **3/3** and the both-halves-true
  control **2/2**; it fails specifically when a conjunct is **absent**
  (0/2; gold `not_enough_info`, Jeff 1 says 'supported' at 0.67–0.72) or the
  claim **overstates** a measured number (0/2; gold 'refuted', Jeff 1 says
  'supported' at 0.83–0.91). On the demo-shaped tied case — one conjunct true,
  the other merely equal, not better — Jeff 1 also returned 'supported', so the
  failure includes a tied component, not only an absent one.
  `scripts/probe_granularity_jev.py`: **near-identical variants
  changed both the wording and the passage structure** — Jeff 1 **0/5**, Jev
  **5/5**, with Jeff 1's confidence swinging 0.506 → 0.946. Because wording and
  chunking moved together, this probe does NOT isolate passage structure as the
  cause; what it shows is **brittleness**: near-identical inputs flip the verdict.
  Live Jev is correct at 0.99–1.00 confidence exactly where Jeff 1 is wrong.
- **Inference-time logit bias is not the fix.** A global 'supported' logit nudge
  is strictly worse than identity on the 9,730-row scale split; the honest
  correction is retraining with more counterexamples, not a bias term. A direct
  `not_enough_info` probability threshold sweep is equally exhausted: over
  t=0.05…0.96 the best gain is 2 rows. The `not_enough_info` signal exists
  (ranking AUC 0.8865) but is not discriminative at the margin — 740 of 2,106
  gold-`not_enough_info` rows already have it as the runner-up, and the rows a
  threshold would flip are currently right.
- **Validation-scope caution.** A 'supported' bias tuned against the sealed
  validation split improves validation by +2 rows while degrading the 9,730-row
  scale by −9 rows. The two splits disagree about the direction of the fix; an
  improvement observed on the validation split alone must not be presented as a
  gain.

## What ships (adapters)

All adapters live under `artifacts/jev_clf/`:

- **`lora_4b`** — the choice-only champion; validation accuracy 0.78392. It was
  trained on 9,119 Choice rows with zero Noul/Score rows (comment block in
  `scripts/jev_clf_server.py`), so its score output is dead-uniform (0.25 per
  level) and it does not handle noul.
- **`lora_4b_multi`** — the demo default; adds 1,800 Score and 1,200 Noul rows to
  training. All three primitives work (validation accuracy 0.778894). Both
  adapters share the LoRA shape: rank 16, applied to `q_proj`, `k_proj`,
  `v_proj`, `o_proj` (see
  `artifacts/jev_clf/lora_4b_multi/adapter_config.json` and
  `artifacts/jev_clf/lora_4b/adapter_config.json`).
- **`lora_merged`** — present in the tree and **unvalidated**. Do not use it.

## How to run

Two scripts in `scripts/`; no external service required.

**`scripts/jev_clf_server.py`** — starts the local Jev-compatible HTTP server
(uvicorn) on port **8079** (the code constant `PORT = 8079` is authoritative; a
docstring at the top of the file says 8078 and is stale). It reports model ID
`jeff-1` and wires up its adapter path in code (the demo default is the multi
adapter, `artifacts/jev_clf/lora_4b_multi`). Endpoints:

- `POST /v1/systemone` — the Jev-compatible judgment endpoint (claim + evidence
  + one typed question).
- `GET /v1/models` — self-describing payload. Note: a couple of accuracy/ECE
  values hardcoded elsewhere in that payload (e.g. `ece_test_n199: 0.063`) do not
  match this release note; the numbers in this note are authoritative. It also
  self-reports a 32,768-token context — that is the server's own claim,
  independently unverified.
- `GET /` — the demo page.
- `GET /health` — health check.
- `/static` — static files served out of `results/static`.

**`scripts/jeff_demo_page.py`** — the single-page demo served at `GET /`. Three
typed question cards against the claim + evidence input (a fact-check verdict over
supported/refuted/not_enough_info; a noul question — "Does the evidence contain a
date or a number?"; and a 4-criteria strength score), each click POSTs to
`/v1/systemone` and renders the response (verdict + confidence, noul, and the
per-level score histogram with criteria). It also shows a reliability diagram
served from `results/static/calibration_plot.png` and an in-repo results table
carrying the same scale-split numbers as this note. If the page displays a
latency figure, it is the server-reported field on the response, not a
client- or benchmark-measured number.

## Supersede note: `README.md` is stale

`README.md` in this repository documents an **earlier, 1.5B-generation model**
(Qwen2.5-1.5B-Instruct + LoRA) and must be treated as superseded by this release
note wherever they conflict.

1. Its results table ("Result (test split, n=199, human labels)": live Jev 1.13.0
   0.799 / 0.791 / 0.114; "this model" 0.759 / 0.738 / 0.111) documents the 1.5B
   generation and a *different*, earlier n=199 "test" split (the Jev row there
   corresponds to `preds_jev_test.jsonl`, not the current sealed validation
   split, where live Jev 1.13.0 scores 0.7688 / ECE 0.1137). Do not cite those
   rows for Jeff 1.
2. "this model: Qwen2.5-1.5B-Instruct + LoRA" names the wrong base model; Jeff 1
   is `Qwen/Qwen3-4B-Instruct-2507`, at validation accuracy 0.7839 / ECE 0.0902.
3. Its run instructions — `scripts.jeff_lm_eval` with
   `--model Qwen/Qwen2.5-1.5B-Instruct --adapter artifacts/jeff/lora_lm` and
   `bash autoresearch.sh` — are stale. Current adapters live under
   `artifacts/jev_clf/`.
4. The layout section names the old package (`jeff/schema.py`, `jeff/lm.py`,
   `jeff/jev.py`, `jeff/data.py`) and old scripts (`scripts/jeff_autoresearch.py`,
   `scripts/jeff_lora_train.py`, `jeff_lm_eval.py`); the current code is under
   `jev_clf/` and `scripts/jev_clf_*`.
5. `configs/jeff_infer.yaml`, listed in the README layout, **does not exist** in
   the current tree (verified).
6. The README's "zero-shot NLI" (0.668) and "previous approach (MiniLM)" (0.493)
   rows reference superseded baselines.
7. The README has no scale split at all; the 9,730-row split above postdates it.