#!/usr/bin/env bash
# Autoresearch harness — goal: "make it as good as jev".
#
# Measures our Jev replacement's accuracy on the VAL split against human
# ground-truth labels. The test split is SEALED: optimising against it would
# make "as good as Jev" a self-fulfilling number.
#
# Bar to beat: live Jev 1.13.0 val accuracy 0.769 (test 0.799).
#
# The editable surface is configs/jev_clf_infer.yaml (readout, temperature,
# per-label bias) — a decision-rule change is measurable in one ~2 min run,
# whereas anything needing the model itself to change needs a retrain.
#
# Deterministic: fixed seed, offline (no hub access), identical val rows.
#
# Prints one line per metric:  METRIC <name>=<value>

set -euo pipefail
cd "$(dirname "$0")"

export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

exec uv run python -m scripts.jev_clf_autoresearch
