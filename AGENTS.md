# Development guide

Jeff 1 is a text-conditioned decision model with Choice, Noul, and Score
interfaces. Read [README.md](README.md) for usage and
[docs/TRAIN_YOUR_OWN.md](docs/TRAIN_YOUR_OWN.md) for the training workflow.

## Released model

- Base: `Qwen/Qwen3-4B-Instruct-2507`
- Adapter: `GestaltLabs/Jeff-1` (`lora_4b_multi`)
- License: Apache 2.0 (`LICENSE`, `NOTICE`)

The client uses the local release adapter when present and otherwise downloads
it from Hugging Face. Other experimental adapters are not part of Jeff 1.

## Setup and entrypoints

```bash
uv sync
uv run python -m scripts.jev_clf_server
```

The demo runs at `http://127.0.0.1:8079`; requests use `POST /v1/systemone`.

| Path | Purpose |
|---|---|
| `jev_clf/schema.py` | Question and prediction schemas |
| `jev_clf/client.py` | Python client |
| `jev_clf/readout.py` | First-token and whole-sequence label scoring |
| `scripts/jev_clf_lora_train.py` | LoRA training |
| `scripts/jev_clf_lm_eval.py` | Model evaluation |
| `scripts/audit_decision_results.py` | Saved-prediction audit |

## Development requirements

- Preserve label descriptions and caller-supplied label order. Use
  `max(probs.items(), key=lambda kv: kv[1])` for the recorded argmax convention;
  sorting labels changes ties.
- Use sequence scoring when candidate labels share their first token.
- Keep training, validation, and test data separate by source/group as well as
  row identifier. The published fact-check evaluation files are not training data.
- Real training runs use Colab GPUs. Local checks should not start model training.
- Test changes to schemas, prompt construction, and readout with representative
  Choice, Noul, and Score inputs. Include labels with shared first tokens.
- Keep unrelated research changes out of release commits.

## Evaluation and documentation

The release metrics are recorded in `results/researchmax_gap_audit.json` and
its Markdown companion. Recompute them from the saved predictions with:

```bash
uv run python -m scripts.audit_decision_results
```

The raw data and prediction files must be available for this command. They
are not all included in a fresh clone.

Use maximum class probability for like-for-like ECE comparisons. Report
accuracy against human labels separately from agreement with a teacher.
Identify the adapter and dataset alongside every result; older Choice-only
validation results do not describe the released adapter. Evaluation data used
for development is not an untouched holdout for later versions.

Current limitations are documented in the README and model card. Preserve
those disclosures when changing documentation. Small interface checks do not
establish comprehensive reasoning equivalence with another model.
