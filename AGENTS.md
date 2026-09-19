# AGENTS.md

This repository is **Jeff 1**: an open-source, locally runnable typed
decision model. Point coding agents here, not at a hosted API.

Code: https://github.com/Gestalt-Lab/jeff
Weights: https://huggingface.co/GestaltLabs/Jeff-1
License: Apache 2.0 (`LICENSE`, `NOTICE`)

Read `README.md` first, then `RELEASE_JEFF1.md` and
`MODEL_CARD_JEFF1.md` before changing any numeric claim.

## What this is

A text-conditioned classifier. The caller supplies a state plus Choice /
Noul / Score questions. Labels and their wording arrive in the prompt at
call time. Classification is read from the language model's own
next-token distribution over those labels. It is not a chat model and
not a JSON generator.

Released adapter: `artifacts/jev_clf/lora_4b_multi` =
`GestaltLabs/Jeff-1` (Qwen3-4B-Instruct-2507 + LoRA r=16).

## Do

- Keep train and eval disjoint. Run
  `uv run python -m scripts.check_train_eval_disjoint`.
- Use insertion-order argmax:
  `max(probs.items(), key=lambda kv: kv[1])`.
  Sorting labels before argmax changes ties.
- Compare calibration with **max class probability** for every model.
  Jev's stored `confidence` field is a different statistic.
- Score gold accuracy and teacher agreement separately.
- Load the **multi** adapter for Choice + Noul + Score. `lora_4b` is
  Choice-only.
- Treat `results/researchmax_gap_audit.md` as the canonical 9,730-row
  recompute.

## Do not

- Claim Jeff beats live Jev 1.13.0 on accuracy. Scale: 7,962/9,730 vs
  8,059/9,730.
- Quote the 199-row val split as a lead (3 rows, McNemar p=0.59).
- Use `artifacts/jev_clf/lora_merged` or any unvalidated soft adapter.
- Treat tiny capability probes as complete semantic parity with Jev.
- Train on `eval_large.jsonl`, `ground_truth` val/test, or
  `eval_schemas.jsonl`.
- Mix Jev's self-reported ECE 0.0790 with Jeff's max-prob ECE 0.0807.
- Start 4B training on this Mac. Use Colab. Do not resume a trajectory
  whose identity is unproven.
- Invent latency, parameter, or carbon numbers.

## Layout

| path | role |
|---|---|
| `jev_clf/client.py` | drop-in `SystemOneClient` |
| `jev_clf/schema.py` | Choice / Noul / Score contract |
| `jev_clf/readout.py` | first-token vs whole-sequence |
| `scripts/jev_clf_server.py` | HTTP `POST /v1/systemone` on port **8079** |
| `scripts/audit_decision_results.py` | fail-closed paired audit |
| `data/factcheck/` | gold and prediction JSONL |
| `artifacts/` | gitignored; download weights from Hugging Face |

## Run

```bash
uv sync
uv run python -m scripts.jev_clf_server   # http://127.0.0.1:8079
```

Python:

```python
from jev_clf.client import SystemOneClient, Choice, Noul, Score
client = SystemOneClient()  # local multi adapter, else GestaltLabs/Jeff-1
```

## Known issues agents must not paper over

- Over-claims `supported` more than Jev (refuted→supported 12.3% vs
  6.2%; NEI→supported 24.4% vs 14.3%).
- Weak `not_enough_info` recall on single-passage rows.
- One forward pass **per question**, not one pass per state.
- Score labels that share a first token use slower sequence readout.
- n=199 splits are smoke tests. Scale has already been used for
  diagnosis; it is not a pristine future holdout.
- A prior OptionScorer run is an **invalid experiment**, not proof the
  architecture failed.
