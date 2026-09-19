---
license: apache-2.0
base_model: Qwen/Qwen3-4B-Instruct-2507
library_name: peft
pipeline_tag: text-classification
language:
- en
tags:
- lora
- fact-checking
- calibration
- system-one
- typed-decisions
- peft
---

# Jeff 1

**A local, open-weight typed decision model.** Give it a state and a
set of Choice / Noul / Score questions; it returns probability
distributions, not prose.

- Weights: this repo (`GestaltLabs/Jeff-1`)
- Code, server, eval, agent docs: [github.com/Gestalt-Lab/jeff](https://github.com/Gestalt-Lab/jeff)
- Base model: [`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) (Apache 2.0)
- Adapter: LoRA r=16, α=32, dropout 0.05, targets `q_proj k_proj v_proj o_proj`
- License: Apache 2.0

Jeff is independent. It is not affiliated with TypeSafe AI. “Jev” is
used only for API compatibility and a measured comparison.

## What

Jeff 1 is a **text-conditioned classifier**. The caller supplies:

1. a **state** — typically a claim plus evidence passages, but any
   JSON/text the prompt can render;
2. **questions** whose label sets and wording are declared **at call
   time**.

The model does not generate a verdict as text. It scores the allowed
labels from its own next-token distribution (first-token softmax when
those tokens are distinct; whole-sequence scoring when they collide,
as Score levels `" 0"`…`" 3"` do).

| primitive | output |
|---|---|
| `choice` | `{choice, probabilities, confidence}` |
| `noul` | `{noul}` — P(yes) in [0, 1] |
| `score` | `{score, probabilities, confidence}` — `score` is the probability-weighted level index |

`confidence` is `max(probabilities)`, a convenience statistic derived
from the distribution, not a separately trained head.

This card describes **only** the `lora_4b_multi` adapter. A Choice-only
sibling (`lora_4b`) exists in the code repo and is **not** these
weights. A merged checkpoint in that tree is unvalidated; do not use it.

## Why

Typed decisions — “supported / refuted / not enough info”, routing
labels, graded strength — are something you want as **numbers code can
compose**, with weights you can run offline.

Hosted Jev already does this well. It is closed: you cannot inspect the
weights, you cannot run without an API, and you cannot point an agent at
a repository instead of a vendor. Jeff exists so that those three things
are possible, even if the first public checkpoint is ~1 accuracy point
behind Jev on our 9,730-row human-label split.

We are releasing **this** adapter rather than waiting for a better one
because the measurements are stable enough to be useful, the failure
modes are named, and an unpublished local model cannot be audited by
anyone else. A later architecture (d-Jeff) is a separate project; it
does not replace this release.

## Why it matters

- **You can run it.** Clone the code, load this adapter, get
  distributions. Nothing leaves the machine at inference time.
- **You can measure it.** We scored Jeff and live Jev 1.13.0 on the
  **same 9,730 human-labelled fact-check rows**. The numbers below are
  from raw per-row files, not marketing rounding.
- **You can see where it fails.** Jeff over-claims `supported` and
  loses on `not_enough_info` when a single topically related passage
  does not actually establish the claim. That is the product, not a
  footnote.
- **Agents have a repo.** [AGENTS.md](https://github.com/Gestalt-Lab/jeff/blob/main/AGENTS.md)
  is the entrypoint for coding agents: what to run, what not to claim.

## Results

Canonical audit: [`results/researchmax_gap_audit.md`](https://github.com/Gestalt-Lab/jeff/blob/main/results/researchmax_gap_audit.md)
in the code repo. Argmax = insertion order (sorting labels changes
ties). ECE uses **max class probability** for both models.

**Scale split — 9,730 human labels** (`eval_large.jsonl`; sources:
VitaminC 3,979, FEVER 3,910, SciFact 982, Climate-FEVER 859):

| model | accuracy | macro-F1 | Brier ↓ | ECE ↓ (max-prob, 10-bin) |
|---|---:|---:|---:|---:|
| **Jeff 1 (this adapter)** | 0.8183 (7,962/9,730) | 0.7789 | 0.2839 | **0.0807** |
| live Jev 1.13.0 | **0.8283** (8,059/9,730) | 0.7994 | **0.2750** | 0.0932 |

- Discordant pairs: Jev-only correct 594, Jeff-only 497.
- Exact McNemar p = 0.0036.
- Paired bootstrap 95% CI on (Jev − Jeff) accuracy: [+0.0033, +0.0165]
  (6,000 resamples, seed 11). Group independence is **not** assumed.

**Honest summary:** Jev is about **1.0 accuracy point** ahead on this
split. Jeff is **better calibrated** under a shared max-probability
definition, and **worse on Brier**. Jev’s stored-confidence ECE of
0.0790 is a different statistic (mean divergence from max-prob ≈ 0.04,
max 0.33) and must not be compared to 0.0807.

**Recall by gold class (scale):**

| gold | n | Jeff recall | Jev recall |
|---|---:|---:|---:|
| supported | 5,031 | **0.931** | 0.889 |
| refuted | 2,593 | 0.794 | **0.805** |
| not_enough_info | 2,106 | 0.578 | **0.712** |

The accuracy gap is concentrated in NEI on **single-passage** rows
(8,643 of 9,730). On 859 five-passage Climate-FEVER rows Jeff is *more*
accurate (0.669 vs 0.591). Over-claim rates still favor Jev on both
error directions: refuted→supported 12.3% vs 6.2%; NEI→supported 24.4%
vs 14.3%.

**Do not quote n=199 as a lead.** A sealed val split exists (Jeff
`lora_4b` 0.7839 vs Jev 0.7688). That is 3 rows, McNemar p=0.59, and a
**different adapter**. This card’s model is `lora_4b_multi`.

Training identity (from `train_metrics.json`): 12,119 SFT rows (Choice +
1,800 Score + 1,200 Noul), 2 epochs, effective batch 32, lr 1e-4, seed
42, Colab A100 40GB, ~51 min. Base revision was not pinned in that run.

## How to use

Needs the 4B base model in memory (GPU or Apple Silicon). This repo is
the LoRA only (~47 MB).

### With the Jeff client (recommended)

```bash
git clone https://github.com/Gestalt-Lab/jeff
cd jeff
uv sync
```

```python
from jev_clf.client import SystemOneClient, Choice, Noul, Score

client = SystemOneClient(adapter="GestaltLabs/Jeff-1")

result = client.system_one(
    {
        "claim": "The new training program made participants both faster and more accurate.",
        "evidence": [
            "New program mean time 42s vs standard 55s.",
            "Both groups scored 91% correct.",
        ],
    },
    {
        "verdict": Choice(
            instructions="Judge the claim from the evidence only. No outside knowledge.",
            criteria={
                "supported": "The passages, if accurate, guarantee the claim.",
                "refuted": "The passages, if accurate, guarantee the claim is false.",
                "not_enough_info": "Silent, mixed, or insufficient.",
            },
        ),
        "has_number": Noul(instructions="Does the evidence contain a number?"),
        "strength": Score(
            instructions="How strongly does the evidence settle the claim?",
            criteria=["none", "weak", "moderate", "strong"],
        ),
    },
)

print(result.choices["verdict"].choice)
print(result.choices["verdict"].probabilities)
print(result.nouls["has_number"].noul)
print(result.scores["strength"].score)
```

The worked example above is a **known failure**: gold is `refuted`
(accuracy is tied, so “more accurate” is false). Jeff often answers
`supported`. See Known issues.

### HTTP

```bash
uv run python -m scripts.jev_clf_server
# POST http://127.0.0.1:8079/v1/systemone
```

Body shape matches TypeSafe `systemone`: `state` + `questions` with
`type: choice | noul | score`.

### Raw PEFT

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = "Qwen/Qwen3-4B-Instruct-2507"
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, dtype="bfloat16")
model = PeftModel.from_pretrained(model, "GestaltLabs/Jeff-1").eval()
```

Load the tokenizer from the **base model**, not from this adapter.
Readout lives in the code repo (`jev_clf/readout.py`); do not assume
greedy generation of a label string.

`adapter_model.safetensors` SHA256:
`13cc3805495f7e901ca3121c7a3647fc6abcfe1fdc098ddc9ab1acd74f436a6a`
(47,224,624 bytes).

## Known issues

Released on purpose, not hidden.

1. **Over-claiming is the dominant error.** Jeff calls `supported` too
   often when one conjunct is true and another is false or absent, and
   when a passage is on-topic but does not entail the claim.
2. **Weak NEI on one passage.** Restoring Jev’s NEI recall on
   single-passage rows would close the accuracy gap (oracle bookkeeping,
   not an achieved model).
3. **Serial compute.** One forward pass per question. Hosted Jev stays
   roughly flat as you add questions over the same state; Jeff does not.
4. **Score is slower.** Shared first-token collision → sequence readout.
5. **Tiny probes ≠ capability parity.** Jaggedness 8/9 vs Jev 9/9;
   conjunction 5/9 vs 9/9. The HTTP schema accepts the three primitives;
   that does not mean Jeff matches Jev on every reasoning pattern.
6. **This scale split is burnt for design.** It has been used for error
   analysis and threshold searches. Treat new work as needing a fresh
   holdout.
7. **No extra confidence head, no post-hoc temperature in the served
   client.** Reported confidence is max probability.
8. **Do not ship** `lora_merged` or unpublished soft-distillation
   adapters as Jeff 1.

## Files

| file | role |
|---|---|
| `adapter_model.safetensors` | LoRA weights |
| `adapter_config.json` | PEFT config (base `Qwen/Qwen3-4B-Instruct-2507`) |
| `train_metrics.json` | training identity snapshot |
| `LICENSE` / `NOTICE` | Apache 2.0 + Qwen attribution |

## Citation

```bibtex
@software{jeff1_2026,
  title  = {Jeff 1: a local open-weight typed decision model},
  author = {Gestalt-Lab},
  year   = {2026},
  url    = {https://github.com/Gestalt-Lab/jeff},
  note   = {Adapter: https://huggingface.co/GestaltLabs/Jeff-1}
}
```
