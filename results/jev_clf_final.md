# jev_clf — final comparison

Two numbers, never to be confused: **accuracy** against human labels (is it any good?) and **agreement with Jev** (is it a faithful clone?). A model can win one and lose the other.

## 1. Accuracy against human ground-truth labels

The non-circular number. `coverage` is scored rows / rows available on that split — a partial run is visible as e.g. `24/199`, not passed off as a full result.

| model / arm | split | n | coverage | accuracy | macro-F1 | ECE | Brier | source |
|---|---|---|---|---|---|---|---|---|
| gt_only | val | 199 | 199/199 | 0.492 | 0.437 | 0.053 | 0.598 | `preds_gt_only_val.jsonl` |
| jev-1.13.0 | test | 199 | 199/199 | 0.799 | 0.791 | 0.114 | 0.338 | `preds_jev_test.jsonl` |
| jev-1.13.0 | val | 199 | 199/199 | 0.769 | 0.745 | 0.114 | 0.367 | `preds_jev_val.jsonl` |
| lm:Qwen/Qwen2.5-0.5B:first_token | val | 199 | 199/199 | 0.412 | 0.292 | 0.245 | 0.775 | `preds_lm_qwen2_5_0_5b_first_token_val.jsonl` |
| lm:Qwen/Qwen2.5-0.5B:sequence | val | 199 | 199/199 | 0.412 | 0.292 | 0.245 | 0.775 | `preds_lm_qwen2_5_0_5b_sequence_val.jsonl` |
| lm:Qwen/Qwen3-4B-Instruct-2507:first_token | val | 199 | 199/199 | 0.724 | 0.686 | 0.267 | 0.545 | `preds_lm_qwen3_4b_instruct_2507_first_token_val.jsonl` |
| lm:Qwen/Qwen3-4B-Instruct-2507:sequence | val | 199 | 199/199 | 0.724 | 0.686 | 0.267 | 0.545 | `preds_lm_qwen3_4b_instruct_2507_sequence_val.jsonl` |
| nli:cross-encoder/nli-deberta-v3-small | test | 199 | 199/199 | 0.668 | 0.658 | 0.277 | 0.593 | `preds_nli_test.jsonl` |
| nli:cross-encoder/nli-deberta-v3-small | val | 199 | 199/199 | 0.608 | 0.594 | 0.334 | 0.697 | `preds_nli_val.jsonl` |
| Qwen/Qwen2.5-1.5B-Instruct+lora_lm (first_token) | val | 199 | 199/199 | 0.754 | 0.726 | 0.099 | 0.383 | `lm_eval_lora_colab.json` |
| Qwen/Qwen2.5-1.5B-Instruct+lora_lm (first_token) | test | 199 | 199/199 | 0.759 | 0.738 | 0.111 | 0.355 | `lm_eval_lora_test.json` |
| Qwen/Qwen2.5-1.5B-Instruct (first_token) | val | 199 | 199/199 | 0.583 | 0.489 | 0.301 | 0.666 | `lm_eval_qwen1.5b_instruct.json` |
| arm:gt_only | val | 199 | 199/199 | 0.492 | 0.437 | 0.053 | 0.598 | `artifacts/jev_clf/gt_only/metrics.json` |

## 2. Agreement with Jev on HELD-OUT question wordings (cloning)

Imitation of the teacher on schemas never seen in training. This is NOT a quality number and must not be read as one.

| model / arm | split | n | coverage | top-1 match | mean TV | source |
|---|---|---|---|---|---|---|
| lm:Qwen/Qwen2.5-0.5B:first_token | val | 398 | 398/398 | 0.445 | 0.521 | `preds_lm_qwen2_5_0_5b_first_token_evalschemas_val.jsonl` |
| lm:Qwen/Qwen2.5-0.5B:sequence | val | 398 | 398/398 | 0.445 | 0.521 | `preds_lm_qwen2_5_0_5b_sequence_evalschemas_val.jsonl` |
| Qwen/Qwen2.5-1.5B-Instruct+lora_lm (first_token) | val | 398 | 398/398 | 0.764 | 0.235 | `lm_eval_lora_colab.json` |
| Qwen/Qwen2.5-1.5B-Instruct (first_token) | val | 398 | 398/398 | 0.623 | 0.372 | `lm_eval_qwen1.5b_instruct.json` |

## 3. Caveats that must travel with the numbers above

- **The jaggedness suite does not discriminate.** Live Jev scored 9/9 on it, so it cannot demonstrate whether a student inherits the teacher's failure modes. Any claim about inherited jaggedness is untested.
- **The NLI baseline short-circuits empty-evidence rows** to `not_enough_info` (`meta.shortcut="no_evidence"`), mildly flattering its all-rows accuracy. The no-shortcut variant is in `results/jev_clf_baselines.json`.
- **Calibration is fit on val only**, never test.
- **The frozen-MiniLM arms are controls, not candidates.** They show what a sentence-similarity encoder plus a small unnormalized head achieves; the head has no input normalization and is therefore not scale-robust across encoders (Qwen's hidden-state absmax ~220 saturates its attention logits at init).
- **Live Jev is only ~0.80 accurate on this data**, so teacher-distilled soft targets inject roughly 20% label noise relative to ground truth.

## 4. Prediction files that matched NEITHER row set (not scored)

- preds_jev_jaggedness.jsonl (n=9, model=jev-1.13.0)
- preds_nli_jaggedness.jsonl (n=9, model=nli:cross-encoder/nli-deberta-v3-small)

