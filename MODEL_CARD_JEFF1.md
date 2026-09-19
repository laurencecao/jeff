# Jeff 1 — Model Card

Jeff 1 is a locally runnable, open-weight replacement for TypeSafe Jev 1.13.0,
the hosted fact-checking judgment API. Given a claim plus evidence passages —
or any structured state — it returns typed judgments over whatever label set
the caller supplies: a labeled **choice**, a presence judgment (**noul**), or a
graded multi-level **score**. Label sets and label wording are declared at call
time; the answer is read from the model's own next-token distribution
restricted to those labels. Inference runs entirely locally (CUDA or Apple
Silicon MPS); no hosted service is involved.

- Code: [github.com/Gestalt-Lab/jeff](https://github.com/Gestalt-Lab/jeff)
- Weights: [huggingface.co/GestaltLabs/Jeff-1](https://huggingface.co/GestaltLabs/Jeff-1)
- License: Apache 2.0 (`LICENSE`, `NOTICE`). Jeff is independent and not
  affiliated with TypeSafe AI.

## Model details

| | |
|---|---|
| Base model | `Qwen/Qwen3-4B-Instruct-2507` (Apache 2.0) |
| Released adapter | `artifacts/jev_clf/lora_4b_multi`, published as `GestaltLabs/Jeff-1` |
| Adapter SHA256 | `13cc3805495f7e901ca3121c7a3647fc6abcfe1fdc098ddc9ab1acd74f436a6a` (`adapter_model.safetensors`) |
| Adapter config | LoRA `r=16`, `alpha=32`, `dropout=0.05`, targets `q_proj, k_proj, v_proj, o_proj` |
| Trainable params | 11,796,480 (0.29% of 4.03B total) |
| Training data | 12,119 rows: 9,119 Choice + 1,800 Score + 1,200 Noul; 2 epochs |
| Serving | `scripts/jev_clf_server.py` (local HTTP, port 8079) or direct PEFT load |

Other adapter directories under `artifacts/jev_clf/` are development artifacts,
are not part of this release, and are unevaluated for release use.

## Evaluation

Headline result, from the canonical paired audit
(`results/researchmax_gap_audit.json` / `.md`): both models scored on the same
9,730 human-labeled rows (`data/factcheck/eval_large.jsonl`; vitaminc 3,979,
fever 3,910, scifact 982, climate_fever 859).

| Model | n | Accuracy | Macro-F1 | Brier | ECE (max-prob) |
|---|---:|---:|---:|---:|---:|
| **Jeff 1** | 9,730 | 0.8183 (7,962) | 0.7789 | 0.2839 | **0.0807** |
| live Jev 1.13.0 | 9,730 | **0.8283** (8,059) | 0.7994 | **0.2750** | 0.0932 |

Accuracy is exact match against the human gold label; argmax ties break by
label insertion order. The accuracy difference is small but real: McNemar
p = 0.0036, bootstrap 95% CI on the difference [+0.0033, +0.0165]. Jeff 1 is
better calibrated under the shared max-class-probability ECE definition (10
equal-width bins, population-weighted); Jev is better on Brier score.

Jev additionally reports an internal confidence statistic that is a different
quantity from max-class probability (the two disagree by up to 0.33 per row).
Scored on that statistic Jev's ECE is 0.0790; it is not comparable to the
max-probability column above.

A separate sealed 199-row test split (`split='test'` of
`data/factcheck/ground_truth.jsonl`) was scored with the released adapter:
accuracy 0.7940, ECE 0.0634, versus live Jev accuracy 0.7990 on the same rows.
At n=199, differences of a few rows are within noise; treat this split as a
sanity check, not a ranking signal.

## Intended use

- Local, offline, self-hosted fact-check verdicts over claim + evidence, using
  the Jev request/response shape so existing Jev clients can point at the local
  server instead of the hosted API.
- The three typed primitives (choice / noul / score) with caller-chosen label
  sets — triage labels, presence checks, ordinal strength grades.
- Research and development where per-request cost or data egress to a hosted
  service is a constraint.

## Limitations

- **Over-claiming is the dominant error.** Jeff 1 answers "supported" for
  12.3% of gold-refuted rows and 24.4% of gold-not_enough_info rows, versus
  6.2% and 14.3% for live Jev. It is systematically more willing than Jev to
  call a claim supported — for example, a conjunctive claim whose evidence
  confirms one half but only ties the other was judged "supported" at 0.796
  confidence.
- **Weak insufficient-evidence handling.** Most of the accuracy gap sits in
  `not_enough_info` recall: 0.578 vs Jev's 0.712 overall, and 0.523 vs 0.655
  on single-passage rows. Jeff 1 tends to treat a topically relevant passage
  as if it entailed the claim. On multi-passage rows (climate_fever, n=859)
  Jeff 1 is more accurate than Jev (0.669 vs 0.591).
- **Small probes are directional only.** On a 9-item adversarial probe Jeff 1
  scored 8/9 vs Jev's 9/9; on a 9-item conjunction probe, 5/9 vs 9/9. These
  tiny probes locate failure modes; they are not evidence of broad
  equivalence or non-equivalence.
- **Not a verifier.** The model judges whether the supplied evidence supports
  the claim; it does not check claims against external sources and can return
  plausible but wrong judgments.
- **Evaluation data informed development.** The 9,730-row split was used for
  error analysis during development, so it is not a pristine holdout for
  future iterations.
- **Unmeasured operational properties.** Latency, throughput, and maximum
  context were not benchmarked for this release; the server's self-reported
  32,768-token context is a config value, not a measurement. Score questions
  use whole-sequence scoring (one extra pass per level), and each question
  costs a forward pass.

## Usage

See [`README.md`](README.md) for install and run instructions: the Python
client (`jev_clf.client.SystemOneClient`), the local HTTP server
(`POST /v1/systemone`, `GET /` demo, `GET /v1/models`, `GET /health` on port
8079), and direct PEFT loading from `GestaltLabs/Jeff-1`.

## Provenance

- Evaluation audit: `results/researchmax_gap_audit.json` and `.md` (canonical
  release figures; input file hashes recorded inside).
- Research history, including earlier adapters and superseded experiments:
  `PROVENANCE.md` and the `results/` archive. These document how the release
  was reached; they are not release instructions.
