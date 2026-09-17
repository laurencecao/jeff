# jev_clf — an independent, decision-only fact-checking model

**Question.** TypeSafe's Jev is described as a classifier that returns typed,
calibrated probabilities instead of text. Can we build one ourselves — a local
model that takes a claim plus its evidence and returns calibrated probabilities
over a declared answer space?

**Answer.** Yes, and it gets close to Jev. Our model reaches **0.759 test
accuracy** against human labels versus **Jev's 0.799** and a zero-shot NLI
cross-encoder's **0.668**, with calibration essentially matching Jev
(ECE 0.111 vs 0.114). It is a 4.3M-parameter LoRA adapter on a 1.5B open-weight
model, trained in **15 minutes on one rented A100 for well under $1**.

## Headline numbers (test split, n=199, human ground truth)

| model | accuracy | macro-F1 | ECE | Brier |
|---|---|---|---|---|
| live Jev 1.13.0 (hosted) | **0.799** | 0.791 | 0.114 | 0.338 |
| **ours: Qwen2.5-1.5B-Instruct + LoRA** | **0.759** | 0.738 | **0.111** | 0.355 |
| zero-shot NLI cross-encoder | 0.668 | 0.658 | 0.277 | 0.593 |

Both numbers that matter are reported separately, and they measure different
things:

- **Accuracy** = vs human labels. Is it any good? **0.759**.
- **Agreement with Jev** = on question wordings held out from training.
  Is it a faithful clone? **0.764 top-1**, mean total-variation 0.235.

Agreement is *not* quality. The 0.5B smoke demonstrates the gap concretely:
0.917 agreement with Jev while being only 0.458 accurate — faithfully imitating a
teacher that is itself wrong about one case in five.

## What the model actually is

A **text-conditioned classifier**, not a fixed-head classifier and not a dLLM:

- The label set and each label's natural-language definition arrive **in the
  prompt at call time**, so a differently-worded question or a different
  *number* of labels is representable without retraining. Verified: the 2-label
  Noul schemas (`fc-n0`, `fc-n1`) are held out entirely and are still scored.
- The classification is read out of the model's **own next-token distribution**,
  restricted to the label tokens. No head is bolted on.
- **One forward pass**, no autoregressive generation loop, no iterative
  denoising — so it is not a diffusion LM.

## The mistake that mattered

The first build used `all-MiniLM-L6-v2` — a **sentence-similarity encoder** —
frozen, feeding a 70k-parameter head. It reached 0.4925, *worse than an
untrained off-the-shelf NLI model*. The encoder was the wrong tool: it was never
trained to represent claim/evidence relations, so the head received
representations with the language understanding already absent.

A supporting diagnosis: the head had **no input normalization**, and Qwen's
hidden states have absmax ~220 versus MiniLM's ~6, so the head's attention
logits saturated at init (~748) and predictions stayed pinned at exactly 1/3
until the learning rate was dropped. The head is not scale-robust across
encoders.

Swapping to a real language model reading its own label distribution fixed it
immediately. Zero-shot accuracy scales cleanly with model size, which is the
evidence that language capacity was the binding constraint:

| zero-shot model | val accuracy (n=199) | ECE |
|---|---|---|
| Qwen2.5-0.5B base | 0.412 | 0.245 |
| Qwen2.5-1.5B-Instruct | 0.583 | 0.301 |
| Qwen3-4B-Instruct | 0.724 | 0.267 |
| **1.5B + LoRA (ours)** | **0.754** | **0.099** |

Fine-tuning a 1.5B on 9119 examples beats a 2.7x larger model zero-shot, and
beats it on calibration by a wide margin. Zero-shot readout is poorly calibrated
(ECE 0.27-0.30); fine-tuning on label-only targets repaired it to 0.099.

## How it was trained

- **Data**: 9119 chat examples = 7000 labelled by live Jev (soft distributions),
  1589 by humans (FEVER / VitaminC / SciFact / Climate-FEVER), 530
  human-labelled rows re-rendered under alternative label wordings.
- **Supervision**: loss on the **assistant label tokens only**.
  Verified in-run: 101 of 27,296 positions supervised, and the supervised text
  was exactly `' <label><|im_end|>'` repeated — not the whole sequence.
- **Prompt contract**: the training prompt is **token-identical** to the eval
  prompt (verified by `scripts/test_prompt_alignment.py`, 233 vs 233 ids).
- LoRA r=16 alpha=32 on q/k/v/o, 4,358,144 trainable params (0.28%),
  2 epochs, 570 steps, effective batch 32, ~1.6 s/step.

## Limitations — read these with the numbers

1. **The teacher is imperfect.** Live Jev is only ~0.80 accurate on this data,
   so 7000 of 9119 training targets imitate a teacher that is wrong about one
   case in five. This is the main suspected reason `refuted` is the weakest
   label (test F1 0.739 vs `supported` 0.810).
2. **The jaggedness suite is untested, not passed.** Live Jev scored 9/9 on it,
   so it cannot show whether a student inherits the teacher's documented failure
   modes (counting, indirection, adversarial framing). Any claim about inherited
   jaggedness is unsupported.
3. **Truncation defect.** A row longer than `max_length` has its prompt
   right-truncated, which removes the `Verdict:` cue itself. Measured impact:
   **1 of 1589 training rows and 0 of every evaluation row**, so the reported
   numbers are unaffected. Fix: drop evidence passages, not the answer cue.
4. **Single seed, n=199 test.** The gap to Jev (4 points) is not accompanied by
   a confidence interval; treat the ordering as indicative.
5. **Calibration is raw.** No temperature scaling was applied; ECE 0.099-0.111
   is the model's own softmax over label tokens.
6. **Cost comparison.** Jev charges $0.042/Mtok input (output free). Our model
   has zero marginal cost per call but requires a GPU at inference and is slower
   per decision than a single Jev API call.

## Reproduce

```bash
# data (costs ~$0.20 in Jev API tokens)
uv run python -m scripts.jev_clf_gen --n 300 --only synthetic --out data/factcheck/synthetic_pilot.jsonl
uv run python -m scripts.jev_clf_distill_scale --target 7000
uv run python -m jev_clf.data
uv run python -m scripts.jev_clf_sft_data

# train (on a CUDA box; 15 min on an A100)
python3 scripts/jev_clf_lora_train.py --config configs/jev_clf_lora_colab.yaml \
  --out-dir artifacts/jev_clf/lora_lm

# evaluate — accuracy and agreement, reported separately
python3 -m scripts.jev_clf_lm_eval --model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter artifacts/jev_clf/lora_lm --split test --readout first_token --dtype bfloat16 \
  --out results/lm_eval_lora_test.json

# consolidate every model into one honest table
uv run python -m scripts.jev_clf_final_report   # -> results/jev_clf_final.md
```

Artifact: `artifacts/jev_clf/lora_lm/` (adapter_model.safetensors 17 MB,
adapter_config.json, tokenizer).
