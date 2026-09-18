#!/usr/bin/env bash
# Self-contained Colab launcher for the soft-target-distillation arm.
#
# Why a launcher script rather than a sequence of exec calls: Colab sessions
# drop every ~30-40 min AND lose /content, so the whole run must be one
# detached process that also retrieves its own result. This script:
#   installs deps (guarding the torchao/peft trap)
#   unpacks the uploaded bundle
#   runs the one-variable A/B check against the multi config
#   trains, then packs the adapter, and prints a PACKED_SIZE marker
#
# Usage on the VM:  nohup bash /content/run_soft.sh > /content/run_soft.log 2>&1 &

set -x

# --- deps ------------------------------------------------------------------
# Install NOTHING unless an import actually fails. Colab's transformers/peft/
# torch are already a matched set; pinning peft down (e.g. 0.17.1) breaks it,
# because Colab's transformers imports HybridCache from peft's newer API:
#   ImportError: cannot import name 'HybridCache' from 'transformers'
# So probe first, and only then consider repairing.
set +e
if ! python -c "import peft, transformers, torch" 2>/dev/null; then
  echo "import probe FAILED -- diagnosing"
  python -c "import peft, transformers, torch" 2>&1 | tail -5
  # Only fix the two known traps, and never downgrade peft: Colab's
  # transformers 5.x needs peft's newer API (HybridCache), so pinning peft
  # DOWN to e.g. 0.17.1 breaks the import in the opposite direction.
  pip uninstall -y torchao >/dev/null 2>&1 || true
  python - <<'EOF'
import importlib, subprocess, sys
try:
    tf = importlib.import_module("transformers").__version__
except Exception:
    tf = "0"
print("transformers", tf)
# Colab's transformers 5.x needs a matching modern peft.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft==0.21.0"])
EOF
fi
set -e
python -c "import peft, transformers, torch; print('peft', peft.__version__, 'tf', transformers.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"

cd /content
mkdir -p jev
tar xzf /content/bundle.tar.gz -C /content/jev
cd /content/jev

echo "=== bundle contents ==="
ls
echo "=== data present ==="
wc -l data/factcheck/sft_train_multi.jsonl data/factcheck/sft_val.jsonl

echo "=== config diff (must be ONLY out_dir/run_name/soft_target_weight/batch) ==="
diff configs/jev_clf_lora_multi.yaml configs/jev_clf_lora_soft.yaml && echo "NO_DIFF"

# --- train -----------------------------------------------------------------
export PYTHONUNBUFFERED=1
python3 -u -m scripts.jev_clf_lora_train --config configs/jev_clf_lora_soft.yaml
rc=$?

echo "TRAIN_RC=$rc"

# --- self-retrieve ---------------------------------------------------------
if [ -d artifacts/jev_clf/lora_4b_soft ]; then
  tar czf /content/soft_out.tar.gz artifacts/jev_clf/lora_4b_soft
  stat -c PACKED_SIZE=%s /content/soft_out.tar.gz
  ls -la artifacts/jev_clf/lora_4b_soft
fi
echo "LAUNCHER_DONE rc=$rc"
exit $rc
