# Jeff 1

**An open-source model for typed decisions, with local inference and a guide to training your own.**

Jeff takes text or structured data and a set of questions, then returns labels
and probabilities instead of generated prose. Label names and descriptions are
supplied at call time.

- [Model weights](https://huggingface.co/GestaltLabs/Jeff-1)
- [Source code](https://github.com/Gestalt-Lab/jeff)
- [Train your own Jeff](docs/TRAIN_YOUR_OWN.md)
- [Coding-agent guide](AGENTS.md)
- **License:** Apache 2.0 for the code and adapter

## What it does

Jeff supports three question types:

| Type | Input | Output |
|---|---|---|
| `choice` | Named labels with descriptions | Selected label, probabilities, and confidence |
| `noul` | A yes/no question | Probability of yes |
| `score` | Ordered levels with descriptions | Level probabilities, expected level index, and confidence |

The model is a LoRA adapter for
[Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).
It scores candidate labels using the language model's token probabilities.
It uses first-token scoring when candidates have distinct first tokens and
whole-sequence scoring otherwise. Confidence is the largest label probability.

## Why Jeff

Typed outputs let applications use model judgments without parsing generated
text. Open weights let you inspect, adapt, and run the model locally; after the
initial download, inference does not require a hosted model API.

Jeff supports Jev-style Choice, Noul, and Score requests. It is an independent
project, not affiliated with TypeSafe AI. API compatibility does not imply
identical judgments or performance.

## Results

The released adapter was evaluated against live Jev 1.13.0 on the same
**9,730 human-labelled fact-checking examples** from FEVER, VitaminC, SciFact,
and Climate-FEVER.

| Model | Accuracy | Macro-F1 | Brier ↓ | ECE ↓ |
|---|---:|---:|---:|---:|
| Jeff 1 | 0.8183 (7,962/9,730) | 0.7789 | 0.2839 | 0.0807 |
| Jev 1.13.0 | 0.8283 (8,059/9,730) | 0.7994 | 0.2750 | 0.0932 |

![Accuracy and calibration](docs/figures/headline.png)

Jeff has lower accuracy and Brier performance than Jev on this evaluation,
but lower expected calibration error (ECE). Both ECE values use maximum class
probability and ten equal-width bins. Jev's separate API confidence statistic
is not used in this comparison. Lower ECE does not guarantee that an individual
prediction is correct.

![Recall by class](docs/figures/recall.png)

Jeff identifies supported claims more reliably than claims with insufficient
evidence. The reliability plot compares predicted confidence with observed
accuracy:

![Reliability](docs/figures/reliability.png)

These results describe `GestaltLabs/Jeff-1` (`lora_4b_multi`), not the older
Choice-only adapter. This evaluation set has also been used for error analysis;
it is not an untouched benchmark for future versions.
[Evaluation details](MODEL_CARD_JEFF1.md) and the
[recomputed metrics](results/researchmax_gap_audit.md) document the comparison.

## Quickstart

Clone the full repository and install its dependencies with
[uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/Gestalt-Lab/jeff
cd jeff
uv sync
```

Run Python examples with `uv run python`. The first model load downloads the
base model and adapter. Inference needs enough memory for the 4B base model;
CUDA and Apple Silicon MPS are supported.

```python
from jev_clf.client import SystemOneClient, Choice, Noul, Score

client = SystemOneClient(adapter="GestaltLabs/Jeff-1")
result = client.system_one(
    {
        "claim": "The museum opened in 1998.",
        "evidence": ["The museum first opened to visitors in 1998."],
    },
    {
        "verdict": Choice(
            instructions="Judge the claim using only the supplied evidence.",
            criteria={
                "supported": "The evidence establishes the claim.",
                "refuted": "The evidence contradicts the claim.",
                "not_enough_info": "The evidence is insufficient to decide.",
            },
        ),
        "has_date": Noul(instructions="Does the evidence include a year?"),
        "strength": Score(
            instructions="How strongly does the evidence settle the claim?",
            criteria=["none", "weak", "moderate", "strong"],
        ),
    },
)
print(result.choices["verdict"].choice)
print(result.choices["verdict"].probabilities)
print(result.nouls["has_date"].noul)
print(result.scores["strength"].score)
```

### Local HTTP server

```bash
uv run python -m scripts.jev_clf_server
```

Open `http://127.0.0.1:8079` for the demo. The server exposes
`POST /v1/systemone`, `GET /health`, and `GET /v1/models`.

### Load with PEFT

For lower-level access:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = "Qwen/Qwen3-4B-Instruct-2507"
tokenizer = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, dtype="bfloat16")
model = PeftModel.from_pretrained(model, "GestaltLabs/Jeff-1").eval()
```

Load the tokenizer from the base model. PEFT loads the weights; the Jeff client
adds prompt construction and label scoring in `jev_clf/readout.py`.

## Train your own Jeff

The [training walkthrough](docs/TRAIN_YOUR_OWN.md) explains how to:

1. Represent your task as states, questions, and labelled answers.
2. Prepare separate training, validation, and test data.
3. Fine-tune a LoRA adapter on a Colab GPU.
4. Evaluate the adapter and load it with the same client.

It explains the learning objective and label-scoring mechanism as well as the
commands. The small examples illustrate the format; a useful model needs a
representative dataset and independent evaluation.

## Limitations

- **Unsupported positive verdicts.** Jeff can mark a claim as supported when
  evidence supports only part of it or merely discusses the same topic. Claims
  combining several assertions are a known weakness.
- **Insufficient evidence.** Performance is weaker when the correct answer is
  `not_enough_info`, particularly with a single evidence passage.
- **Evidence only.** Jeff does not retrieve sources or independently establish
  whether supplied evidence is true. It can give confidently wrong answers.
- **Per-question computation.** Questions are scored separately. Labels that
  require whole-sequence scoring incur additional computation.
- **Limited generalization evidence.** The main evaluation is fact-checking.
  Small Choice, Noul, and Score checks verify supported interfaces, not broad
  reasoning equivalence to Jev or reliability on a new task.

## Repository layout

| Path | Purpose |
|---|---|
| `jev_clf/` | Question schemas, client, label scoring, and evaluation |
| `scripts/jev_clf_server.py` | Local HTTP server |
| `scripts/jev_clf_lora_train.py` | LoRA training |
| `scripts/audit_decision_results.py` | Paired prediction audit |
| `docs/TRAIN_YOUR_OWN.md` | Training tutorial |
| `AGENTS.md` | Development instructions for coding agents |

See [release notes](RELEASE_JEFF1.md) for the published artifact and
[research history](PROVENANCE.md) for earlier experiments. Model weights are
hosted on Hugging Face, not stored in Git.
