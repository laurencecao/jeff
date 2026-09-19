# Jeff 1 — Model Card

Jeff 1 is a locally runnable replacement for TypeSafe Jev 1.13.0, the hosted
fact-checking judgment API. Given a claim plus a set of evidence passages (the
request shape Jev clients already speak), it returns one of three typed
judgments over whatever label set the caller supplies: a labeled **choice**, a
presence judgment (**noul**), or a graded multi-criteria **score**. The entire
inference stack runs on a laptop (Apple Silicon, MPS); there is no hosted
service at inference time.

**Task:** text-conditioned fact-checking verdicts ("supported / refuted /
not_enough_info" or any caller-chosen label set), plus generic typed primitives
(2–5 label choices, yes/no presence checks, ordinal score levels with per-level
probability distributions).

**Publication:** code [github.com/Gestalt-Lab/jeff](https://github.com/Gestalt-Lab/jeff);
weights [huggingface.co/GestaltLabs/Jeff-1](https://huggingface.co/GestaltLabs/Jeff-1);
license Apache 2.0 (`LICENSE`, `NOTICE`). Agent entrypoint: `AGENTS.md`.
Headline scale numbers are the insertion-order recompute in
`results/researchmax_gap_audit.md` (Jeff 7,962/9,730 vs Jev 8,059/9,730).
This card no longer pins commit `527f2a4` as the release SHA — use `git rev-parse HEAD`
after the release commit. Adapter SHA256
`13cc3805495f7e901ca3121c7a3647fc6abcfe1fdc098ddc9ab1acd74f436a6a`.

---

## Model architecture / lineage

| | |
|---|---|
| Base model | `Qwen/Qwen3-4B-Instruct-2507` (Apache 2.0, per `https://huggingface.co/api/models/Qwen/Qwen3-4B-Instruct-2507`) |
| Adapter | LoRA, `r=16`, target modules `q_proj, k_proj, v_proj, o_proj` (per `artifacts/jev_clf/lora_4b_multi/adapter_config.json`; `artifacts/jev_clf/lora_4b/adapter_config.json` has the same shape) |
| Decoding | local inference (CUDA or Apple Silicon MPS) via `scripts/jev_clf_server.py` |
| Adapter license | Apache 2.0 (this repository `LICENSE`; same terms on Hugging Face) |

The model is a *text-conditioned* judge, not a fixed-output classifier: labels
and their wording are supplied at call time, and the model is not tied to the
three fact-check verdicts. Capability parity with the hosted API was checked
with one live call exercising a 2-label binary choice, a 5-label multi-word
choice, a noul, a 3-level score, a 5-level score, and a 3-label
supported/refuted/not_enough_info verdict.

## Adapters shipped with this repository

| Directory | Role | Sealed val (n=199) accuracy |
|---|---|---|
| `artifacts/jev_clf/lora_4b` | Choice-only champion. Trained on 9,119 Choice rows, 0 Noul / 0 Score rows (per the comment in `scripts/jev_clf_server.py`); its score output is a dead-uniform 0.25 per level. | 0.7839 |
| `artifacts/jev_clf/lora_4b_multi` | **Demo default.** Adds 1,800 Score rows and 1,200 Noul rows to the same base (per the comment in `scripts/jev_clf_server.py`). | 0.7789 |

A merged checkpoint (`artifacts/jev_clf/lora_merged`) exists in the tree but is
**unvalidated — do not use it**; it has no verified evaluation numbers.

## Evaluation

Two splits, both evaluated the same way (details in *How measured*):

| Split | n | Model | Accuracy | ECE |
|---|---|---|---|---|
| Val (sealed) | 199 | **Jeff 1** | **0.7839** | **0.0902** |
| Val (sealed) | 199 | live Jev 1.13.0 | 0.7688 | 0.1137 |
| Scale (human labels) | 9,730 | **Jeff 1** | **0.8183** | **0.0807** |
| Scale (human labels) | 9,730 | live Jev 1.13.0 | 0.8283 | 0.0932 |

The **val split (n=199) is sealed**: it is the `split='val'` subset of
`data/factcheck/ground_truth.jsonl` (1,987 rows total: 1,589 `train`, 199 `val`,
199 `test`), ~500 rows each from vitaminc, scifact, climate_fever, and fever.
The 199 `split='test'` rows are a separate sealed holdout and are not the rows
scored in the table. The **scale split (n=9,730)** is
`data/factcheck/eval_large.jsonl`, all rows human-labeled, with source counts
vitaminc 3,979, fever 3,910, scifact 982, climate_fever 859.

**How measured.** Accuracy is exact-match against the human gold label. Live
Jev 1.13.0 numbers come from the hosted API scored on the *identical* rows.
ECE is 10 equal-width, population-weighted bins with confidence = max class
probability, computed identically for both models — which is why the two ECE
columns are directly comparable and why Jeff 1 is better calibrated on the
scale split (0.0807 vs 0.0932). In-repo, `results/jeff_final.md` reports the
same val figures at 3 decimal places (0.784 / 0.090 / 0.769 / 0.114), and the
scale figures appear in the server's `/v1/models` payload and the demo page.

> **Calibration caveat — read before quoting any ECE.** Jev also reports an
> *internal* confidence statistic. Scored on that statistic, Jev's ECE is
> 0.0790. That number is **not comparable** to the table above: it uses a
> different confidence definition than Jeff 1's max-class probability, and
> Jev's internal statistic diverges from the max-class probability by up to
> 0.33 (mean 0.040). It must never be presented as Jev beating Jeff 1's
> 0.0807; the like-for-like comparison is 0.0932 vs 0.0807, and Jeff 1 wins it.

Additional Jeff 1 numbers on the sealed val split: **macro-F1 0.7620,
Brier score 0.3186, agreement-with-Jev 0.8518**.

> **Do not quote the val split as a lead over Jev.** The 0.7839 vs 0.7688
> difference is 3 rows out of 199. A paired McNemar test on those same rows
> gives 17 discordant pairs for Jeff 1 and 14 against, z=0.54, **p=0.59** —
> not significant — and a bootstrap 95% CI on val accuracy spans ±5.6 points
> (±11 rows), about 7× wider than the scale split's ±0.75 points. On the scale
> split, where the split is large enough to decide, the accuracy difference
> runs the other way and *is* significant: Jev 0.8283 vs Jeff 1 0.8183,
> paired McNemar z=2.94, **p=0.0033**. Jeff 1's advantage is calibration
> (0.0807 vs 0.0932), not accuracy. Also note the two splits disagree on the
> direction of a fix — biasing the `supported` label gains 2 rows on val while
> losing 9 on scale — so val must not be used for tuning.

## Intended use

- Local, offline, self-hosted fact-check verdicts over claim + evidence, using
  the Jev request/response shape so existing Jev clients can point at the local
  server instead of the hosted API (`scripts/jev_clf_server.py`).
- The three typed primitives (choice / noul / score) with caller-chosen label
  sets — e.g. triage labels, presence checks, ordinal strength grades.
- Research and development where per-request cost and data egress to a hosted
  service are constraints.

## Out of scope / limitations

- **Training data composition is not restated here** beyond what is citable in
  the repo (Choice/Score/Noul row counts above); no training-compute figures
  are claimed.
- The model is text-conditioned and will produce *plausible but wrong*
  judgments; it does not verify claims against external sources — it only
  judges whether the supplied evidence supports the claim.
- Latency, throughput, and maximum context were not benchmarked for this
  release; the server's self-reported `context: 32768` in `/v1/models` is a
  config value, not a measurement.
- `artifacts/jev_clf/lora_merged` is unvalidated and must not be used.

## Known failure modes

These are measured, not hypothetical — the numbers are the only ones stated
for these failure classes.

1. **Over-claiming (dominant failure).** On the scale split, Jeff 1 answers
   "supported" for **12.3%** of gold-refuted claims and **24.4%** of
   gold-not_enough_info claims, versus **6.2%** and **14.3%** for live Jev.
   Jev is the more conservative judge; Jeff 1 over-claims roughly twice as
   often on refuted rows.
2. **Worked example of the over-claim.** Claim: *"The new training program
   made participants both faster and more accurate than standard training."*
   Evidence: the new program was faster (42s vs 55s) but accuracy was tied
   (91% both). The conjunction makes the claim false — gold is **refuted** —
   but Jeff 1 returned **supported @ 0.796**. The model is weak at
   conjunctions where one conjunct fails.
3. **Adversarial jaggedness.** On a 9-item adversarial jaggedness probe,
   Jeff 1 scored **8/9** versus live Jev's 9/9 (the reference score is recorded
   in `scripts/probe_jaggedness.py`). The one miss was a counting probe: gold
   **refuted**, Jeff 1 answered **not_enough_info @ 0.639** — it abstained
   instead of falsifying.
4. **Fixes, honestly.** A global "supported" logit bias at inference time is
   strictly worse than identity (no bias) on the scale split, so the honest
   fix is **retraining**, not a post-hoc nudge. A related caution: a
   "supported" bias tuned on the sealed val split improves that split by 2
   rows but degrades the 9,730-row scale split by 9 rows — the two splits
   disagree, and val-only tuning must not be presented as a gain.

## Usage

Full run instructions are in `RELEASE_JEFF1.md`; in one paragraph: run
`scripts/jev_clf_server.py` (serves on port 8079; endpoints `POST
/v1/systemone`, `GET /` demo page, `GET /health`, `GET /v1/models`), which
loads the multi adapter `artifacts/jev_clf/lora_4b_multi` by default; serve
requests with the Jev-systemone body; `scripts/jeff_demo_page.py` documents
the demo UI and the reliability diagram served from
`results/static/calibration_plot.png`.

## Citation / provenance

- Repo: this directory, commit `527f2a4`.
- Base model: `Qwen/Qwen3-4B-Instruct-2507` (Apache 2.0).
- Evaluation splits: `data/factcheck/ground_truth.jsonl` (sealed val,
  n=199) and `data/factcheck/eval_large.jsonl` (scale, n=9,730).
- Reference implementation of the comparison harness and server:
  `scripts/jev_clf_server.py`, `scripts/jeff_demo_page.py`,
  `scripts/probe_jaggedness.py`.