#!/usr/bin/env bash
# Score one adapter on the 9,730-row scale split, identically to how
# preds_ours_large.jsonl was produced, so arms are comparable.
#
# Usage: bash scripts/run_scale_ab.sh <adapter-dir> <tag>
#   e.g. bash scripts/run_scale_ab.sh artifacts/jev_clf/lora_4b_soft soft
#
# Writes results/scale_<tag>.json and data/factcheck/preds_<tag>_large.jsonl.
# Detached by the caller; this script itself is meant to run via nohup.

set -euo pipefail
cd "$(dirname "$0")/.."

ADAPTER="${1:?adapter dir required}"
TAG="${2:?tag required}"

uv run python -m scripts.jev_clf_lm_eval \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --adapter "$ADAPTER" \
  --split test \
  --readout first_token \
  --data-file data/factcheck/eval_large.jsonl \
  --save-preds "data/factcheck/preds_${TAG}_large.jsonl" \
  --out "results/scale_${TAG}.json"

echo "SCALE_AB_DONE tag=${TAG}"
