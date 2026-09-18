"""CLI: train the full_minilm arm (ground truth + Jev distillation).

Thin wrapper around scripts/jeff_train_arm.main() that adds the
full-arm-specific data assertions the generic runner does not make:

  * no ``split == "test"`` row id reaches the trainer (the runner also
    drops and re-asserts this; we assert on the raw load too),
  * zero ``row_id`` overlap with ``data/factcheck/eval_schemas.jsonl``
    (held-out schemas fc-c1/fc-c2/fc-n0/fc-n1 — eval only, never trained on),
  * exactly 8589 train rows (7000 distill_full + 1589 ground_truth train)
    and 199 val rows (ground_truth val).

Usage (from the repo root):

    uv run python -m scripts.jeff_train_full

Extra args are forwarded to the arm runner (e.g. ``--out``, ``--preds-out``).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from jeff.schema import read_rows  # noqa: E402
from scripts.jeff_train_arm import main as arm_main  # noqa: E402

ARM = "full_minilm"
CONFIG = "configs/jeff_full_minilm.yaml"
TRAIN_FILES = [
    "data/factcheck/distill_full.jsonl",
    "data/factcheck/ground_truth.jsonl",
]
EVAL_SCHEMAS = "data/factcheck/eval_schemas.jsonl"
EXPECTED_N_TRAIN = 8589  # 7000 distill_full + 1589 ground_truth train
EXPECTED_N_VAL = 199     # ground_truth val


def _preflight() -> None:
    rows = [r for f in TRAIN_FILES for r in read_rows(REPO_ROOT / f)]
    eval_ids = {r.row_id for r in read_rows(REPO_ROOT / EVAL_SCHEMAS)}

    used_ids = {r.row_id for r in rows}
    leaked_eval = used_ids & eval_ids
    assert not leaked_eval, (
        f"eval_schemas.jsonl row ids in training data: {sorted(leaked_eval)[:5]}"
    )

    test_ids = {r.row_id for r in rows if r.split == "test"}
    train_rows = [r for r in rows if r.split == "train"]
    val_rows = [r for r in rows if r.split == "val"]
    assert not (test_ids & {r.row_id for r in train_rows + val_rows}), (
        "test row id overlaps train/val ids"
    )
    assert len(train_rows) == EXPECTED_N_TRAIN, (
        f"expected {EXPECTED_N_TRAIN} train rows, got {len(train_rows)}"
    )
    assert len(val_rows) == EXPECTED_N_VAL, (
        f"expected {EXPECTED_N_VAL} val rows, got {len(val_rows)}"
    )
    print(
        f"[preflight] {len(rows)} rows loaded ({len(train_rows)} train / "
        f"{len(val_rows)} val / {len(test_ids)} test-to-drop); "
        f"0 eval_schemas row ids present, 0 test ids in train/val — asserted"
    )


def main() -> None:
    _preflight()
    sys.argv = [
        "jeff_train_arm",
        "--arm", ARM,
        "--config", CONFIG,
        *sys.argv[1:],
    ]
    arm_main()


if __name__ == "__main__":
    main()
