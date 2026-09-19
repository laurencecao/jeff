# Jeff 1 — Release Notes

Jeff 1 is an open-source, locally runnable replacement for TypeSafe Jev 1.13.0,
the hosted fact-checking judgment API. Given a claim and evidence passages, it
returns typed judgments over the same request shape Jev clients already speak:
a labeled **choice**, a presence judgment (**noul**), or a graded **score**.
The model is `Qwen/Qwen3-4B-Instruct-2507` (Apache 2.0) plus a rank-16 LoRA
adapter, running entirely locally (CUDA or Apple Silicon MPS). No hosted
service is involved at inference time.

## What ships

The released artifact is the adapter `artifacts/jev_clf/lora_4b_multi`,
published on Hugging Face as **`GestaltLabs/Jeff-1`**
(`adapter_model.safetensors` SHA256
`13cc3805495f7e901ca3121c7a3647fc6abcfe1fdc098ddc9ab1acd74f436a6a`; see
`SHA256SUMS.txt` in the adapter directory). It was trained on 12,119 rows —
9,119 Choice, 1,800 Score, 1,200 Noul — and serves all three primitives.

Other directories under `artifacts/jev_clf/` (`lora_4b`, `lora_4b_soft`,
`lora_merged`, and others) are development artifacts. They are not part of the
release and are unevaluated for release use.

## Results

Canonical figures are the paired audit in
`results/researchmax_gap_audit.json` / `.md`: Jeff 1 and live Jev 1.13.0 scored
on the same 9,730 human-labeled rows (`data/factcheck/eval_large.jsonl`;
sources: vitaminc 3,979, fever 3,910, scifact 982, climate_fever 859).

| Model | n | Accuracy | Macro-F1 | Brier | ECE (max-prob) |
|---|---:|---:|---:|---:|---:|
| **Jeff 1** | 9,730 | 0.8183 (7,962) | 0.7789 | 0.2839 | **0.0807** |
| live Jev 1.13.0 | 9,730 | **0.8283** (8,059) | 0.7994 | **0.2750** | 0.0932 |

- **Accuracy** is exact match against the human gold label; argmax ties break
  by label insertion order.
- **ECE** uses 10 equal-width, population-weighted bins with confidence defined
  as the maximum class probability — the same definition for both models, so
  the two ECE columns are directly comparable.
- The accuracy gap is statistically real but small: McNemar p = 0.0036,
  bootstrap 95% CI on the difference [+0.0033, +0.0165].
- Jev also reports an internal confidence statistic; scored under the same
  binning it gives ECE 0.0790. That statistic is a different quantity from
  max-class probability (per-row disagreement up to 0.33), so it must not be
  compared against the max-probability column above.

On a separate sealed 199-row test split (`split='test'` of
`data/factcheck/ground_truth.jsonl`), the released adapter scores accuracy
0.7940 / ECE 0.0634 against live Jev's 0.7990 on the same rows. At n=199 this
is a sanity check, not a ranking signal.

### Where the gap sits

The accuracy deficit is concentrated, not diffuse. Jeff 1 is more accurate
than Jev on gold-`supported` rows (346 errors vs 558) and on multi-passage
rows (climate_fever: 0.669 vs 0.591). The deficit is `not_enough_info` recall:
0.578 vs 0.712 overall, and on single-passage rows (8,643 of 9,730) 0.523 vs
0.655. Jeff 1 tends to treat a topically relevant passage as if it entailed
the claim.

## Capability parity with Jev

Jev exposes three typed judgment primitives; Jeff 1 implements all three, with
label sets and wording supplied in free text at call time. Parity was exercised
live across a 2-label binary choice, a 5-label multi-word choice, a noul
question, 3-level and 5-level scores, and the 3-label fact-check verdict.

## Known failure modes

- **Over-claiming is the dominant error.** Jeff 1 answers "supported" for
  12.3% of gold-refuted rows and 24.4% of gold-not_enough_info rows, versus
  6.2% and 14.3% for live Jev. It is systematically more willing to call a
  claim supported.
- **Worked example.** Claim: "The new training program made participants both
  faster and more accurate than standard training." Evidence: faster (42s vs
  55s) but accuracy tied (91% both). The claim is conjunctive, so gold is
  "refuted"; Jeff 1 returned "supported" at 0.796 confidence, riding the true
  conjunct.
- **Small probes are directional only.** Adversarial jaggedness probe: Jeff 1
  8/9, Jev 9/9 (`scripts/probe_jaggedness.py`). Conjunction probe: Jeff 1 5/9,
  Jev 9/9 (`scripts/probe_conjunction.py`) — Jeff 1 passes falsified-conjunct
  and both-true cases but fails when a conjunct is absent or the claim
  overstates a measured number. A 5-item granularity probe
  (`scripts/probe_granularity_jev.py`) shows near-identical inputs flipping the
  verdict (Jeff 1 0/5, Jev 5/5). These probes locate failure modes; they are
  not evidence of broad equivalence or non-equivalence.
- **Evaluation data informed development.** The 9,730-row split was used for
  error analysis during development; it is not a pristine holdout for future
  iterations.
- **Unmeasured operational properties.** Latency, throughput, and maximum
  context were not benchmarked; the server's self-reported 32,768-token context
  is a config value. Score questions use whole-sequence scoring (one extra pass
  per level), and each question costs a forward pass.

## How to run

See [`README.md`](README.md) for install and usage: the Python client
(`jev_clf.client.SystemOneClient`), the local server
(`uv run python -m scripts.jev_clf_server`, port 8079, endpoints
`POST /v1/systemone`, `GET /`, `GET /v1/models`, `GET /health`), and direct
PEFT loading from `GestaltLabs/Jeff-1`.

## Provenance

`PROVENANCE.md` and the `results/` archive record the research history —
earlier adapters, superseded experiments, and intermediate evaluations. They
document how this release was reached and are not release instructions.
