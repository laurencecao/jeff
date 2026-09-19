#!/usr/bin/env bash
# Self-contained Colab launcher for the soft-target-distillation arm.
#
# Colab recycles sessions and WIPES /content with them, so a run that only packs
# its artifact at the very end can lose hours. This launcher instead:
#   * trains with --resume, so a re-launch continues from the last epoch
#   * polls in the background and packs the adapter after EVERY epoch
#   * prints a PACKED_SIZE marker each time, so the artifact is downloadable
#     within one epoch of any recycle event
#
# Usage on the VM:  nohup bash /content/run_soft.sh > /content/run_soft.log 2>&1 &

set -x

# --- deps ------------------------------------------------------------------
# Install NOTHING unless an import actually fails. Colab's transformers/peft/
# torch are already a matched set; pinning peft down (e.g. 0.17.1) breaks it,
# because Colab's transformers 5.x imports peft's newer API:
#   ImportError: cannot import name 'HybridCache' from 'transformers'
set +e
if ! python -c "import peft, transformers, torch" 2>/dev/null; then
  echo "import probe FAILED -- repairing peft, never downgrading it"
  python -c "import peft, transformers, torch" 2>&1 | tail -5
  python -m pip install -q "peft==0.21.0"
fi
# torchao check. A plain `import torchao` is NOT the right test: the failure
# only fires when peft builds a LoRA layer, because its dispatcher calls
# is_torchao_available() which RAISES on an old version:
#   ImportError: Found an incompatible version of torchao. Found version 0.10.0,
#   but only versions above 0.16.0 are supported
# So test the version, not the import.
python - <<'EOF'
import importlib, subprocess, sys
try:
    m = importlib.import_module("torchao")
    v = getattr(m, "__version__", "0")
    major, minor = (int(x) for x in v.split(".")[:2])
    too_old = (major, minor) < (0, 16)
except ImportError:
    too_old = False
if too_old:
    print("torchao is too old for peft; uninstalling (safe: peft treats it as optional)")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"])
else:
    print("torchao OK or absent")
EOF
set -e
python -c "import peft, transformers, torch; print('peft', peft.__version__, 'tf', transformers.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"

cd /content
mkdir -p jev
tar xzf /content/bundle.tar.gz -C /content/jev
cd /content/jev

echo "=== data present ==="
wc -l data/factcheck/sft_train_multi.jsonl data/factcheck/sft_val.jsonl

echo "=== config diff (must be ONLY out_dir/run_name/soft_target_weight) ==="
diff configs/jev_clf_lora_multi.yaml configs/jev_clf_lora_soft.yaml || true

ARM=artifacts/jev_clf/lora_4b_soft

# --- pack helper: called after every epoch boundary ------------------------
pack() {
  if [ -d "$ARM" ]; then
    tar czf /content/soft_out.tar.gz "$ARM"
    stat -c PACKED_SIZE=%s /content/soft_out.tar.gz
  fi
}

# --- watcher: pack as soon as each epoch checkpoint lands ------------------
# Runs alongside training so the artifact is retrievable mid-run, not just at
# the end. Guarded on the checkpoint file's mtime changing.
(
  last=""
  while pgrep -f "[j]ev_clf_lora_train" >/dev/null 2>&1; do
    if [ -f "$ARM/train_state.json" ]; then
      cur=$(stat -c '%Y' "$ARM/train_state.json" 2>/dev/null || echo "")
      if [ -n "$cur" ] && [ "$cur" != "$last" ]; then
        last="$cur"
        echo "[watch] new checkpoint detected"
        pack
      fi
    fi
    sleep 30
  done
  echo "[watch] training process gone"
) &
WATCHER=$!

# --- train -----------------------------------------------------------------
export PYTHONUNBUFFERED=1
python3 -u -m scripts.jev_clf_lora_train \
  --config configs/jev_clf_lora_soft.yaml --resume
rc=$?

echo "TRAIN_RC=$rc"
kill $WATCHER 2>/dev/null || true

# --- final pack ------------------------------------------------------------
pack
ls -la "$ARM" 2>/dev/null || true
echo "LAUNCHER_DONE rc=$rc"
exit $rc
