#!/usr/bin/env bash
# Score the soft-distill arm on the scale split and A/B it against the multi arm.
# Run AFTER the Colab adapter has been downloaded to artifacts/jev_clf/lora_4b_soft.
#
#   bash scripts/score_soft_arm.sh
#
# Steps: verify the adapter is real -> score on scale -> paired A/B vs the
# baseline. The A/B is the decision instrument; the 199-row val split is not.

set -euo pipefail
cd "$(dirname "$0")/.."

ARM=artifacts/jev_clf/lora_4b_soft
if [ ! -f "$ARM/adapter_model.safetensors" ]; then
  echo "ERROR: $ARM/adapter_model.safetensors missing -- download the Colab result first" >&2
  exit 1
fi
echo "=== adapter ==="
ls -la "$ARM"
python3 -c "
import json
c=json.load(open('$ARM/adapter_config.json'))
print('base:', c.get('base_model_name_or_path'), 'r:', c.get('r'), 'alpha:', c.get('lora_alpha'))
print('targets:', c.get('target_modules'))
"

echo
echo "=== scoring on the 9,730-row scale split (identical protocol to preds_ours_large) ==="
uv run python -m scripts.jev_clf_lm_eval \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --adapter "$ARM" \
  --split test \
  --readout first_token \
  --data-file data/factcheck/eval_large.jsonl \
  --save-preds data/factcheck/preds_soft_large.jsonl \
  --out results/scale_soft.json

echo
echo "=== paired A/B: baseline (multi) vs soft arm ==="
uv run python -m scripts.ab_scale_preds \
  data/factcheck/preds_ours_large.jsonl \
  data/factcheck/preds_soft_large.jsonl

echo
echo "=== compare against the pre-registered prediction ==="
echo "Pre-registration said: scale accuracy IMPROVES by +10..+40 rows,"
echo "gain concentrated in 1-passage not_enough_info recall, and probably"
echo "NOT individually significant (p > 0.05). See PREREGISTRATION_soft_distill.md"
