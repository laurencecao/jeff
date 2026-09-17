# jev_clf — an independent, decision-only fact-checking model

Our own **Jev replacement**. It is not a wrapper around TypeSafe's API and not a
fine-tuned chat model: it is a **text-conditioned classifier** that takes a
claim plus its retrieved evidence and returns calibrated probabilities over
`supported` / `refuted` / `not_enough_info`.

## What it is

- The label set and each label's natural-language definition arrive **in the
  prompt at call time**, so a differently-worded question or a different
  *number* of labels works without retraining.
- The classification is read out of a real language model's **own next-token
  distribution**, restricted to the label tokens. No head is bolted on.
- **One forward pass** — no autoregressive loop, no iterative denoising (so it
  is not a diffusion LM).

## Result (test split, n=199, human labels)

| model | accuracy | macro-F1 | ECE |
|---|---|---|---|
| live Jev 1.13.0 (hosted) | 0.799 | 0.791 | 0.114 |
| **this model: Qwen2.5-1.5B-Instruct + LoRA** | **0.759** | 0.738 | **0.111** |
| zero-shot NLI cross-encoder | 0.668 | 0.658 | 0.277 |
| previous approach (MiniLM encoder + 70k head) | 0.493 | 0.437 | 0.053 |

Accuracy and agreement-with-Jev are reported **separately** and are never
averaged: a model can imitate the teacher faithfully while being wrong.

## Layout

| path | what |
|---|---|
| `jev_clf/schema.py` | frozen contract: questions, `DecisionRow`, `PredictionRow` |
| `jev_clf/lm.py` | label readout from a language model |
| `jev_clf/jev.py` | live Jev teacher client (resumable cache) |
| `jev_clf/data.py` | FEVER / VitaminC / SciFact / Climate-FEVER loaders |
| `scripts/jev_clf_autoresearch.py` | the metric the loop optimises |
| `scripts/jev_clf_lora_train.py` | LoRA fine-tune |
| `scripts/jev_clf_lm_eval.py` | accuracy + agreement evaluation |
| `configs/jev_clf_infer.yaml` | **the loop's editable decision rule** |
| `results/jev_clf_findings.md` | full write-up, limitations included |
| `CONTRACTS_JEV_CLF.md` | cross-slice interface contract |

## Run

```bash
uv run python -m scripts.jev_clf_lm_eval --model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter artifacts/jev_clf/lora_lm --split test --readout first_token \
  --dtype bfloat16 --out results/lm_eval_lora_test.json

bash autoresearch.sh        # the loop metric (val only)
```

Data note: the large distilled/SFT JSONL and the raw HF cache are gitignored
(they are rebuilt from live Jev calls, ~$0.20 of teacher tokens).
