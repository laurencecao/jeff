"""CLI: train one jeff arm and evaluate it on val (never test).

Examples (from the repo root):

    uv run python -m scripts.jeff_train_arm --arm gt_only \
        --config configs/jeff_gt_only.yaml

    uv run python -m scripts.jeff_train_arm --arm distill_full \
        --config configs/jeff_distill_full.yaml \
        --data data/factcheck/ground_truth.jsonl \
        --data data/factcheck/distill_full.jsonl \
        --label-source ""          # no label_source filter

What it does, in order:
  1. Loads the arm config and the data files (config ``data.train_files``
     unless ``--data`` is given), optionally filtering to one
     ``label_source`` (config ``data.label_source`` / ``--label-source``).
  2. Asserts no test-split row reaches ``train()`` — test is reserved for
     the final cross-arm comparison.
  3. Calls ``jeff.train.train`` (soft-target CE + val calibration +
     its own guarded MLflow run).
  4. Flattens ``out_dir/checkpoint/`` into ``out_dir/`` so the arm dir has
     the frozen layout ``{head.pt, config.json, calibration.json,
     metrics.json}``.
  5. Predicts on val with the saved checkpoint (``OptionScorer.load``,
     eval mode), writes ``data/factcheck/preds_<arm>_val.jsonl``, and
     evaluates with ``ground_truth_metrics`` + ``reliability_bins``.
  6. Rewrites ``metrics.json`` with the full record (epoch losses, val
     accuracy / macro-F1 / per-label P/R, ECE before/temperature/isotonic,
     n_train/n_val, encoder, seed, device, wall time, checkpoint path,
     MLflow run id) and logs a second MLflow run tagged with the arm name.
  7. Prints a 3-row eval-mode forward pass as a sanity check.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from jeff.eval import ground_truth_metrics, reliability_bins  # noqa: E402
from jeff.model import OptionScorer  # noqa: E402
from jeff.schema import PredictionRow, read_rows, write_predictions  # noqa: E402
from jeff.train import ARTIFACTS, MLFLOW_EXPERIMENT, MLFLOW_URI, train  # noqa: E402


def _peak_rss_gb() -> float:
    """Process peak resident set size, GiB (ru_maxrss is bytes on macOS)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30


def _mps_allocated_gb() -> float | None:
    """MPS driver-allocated memory, GiB; None when MPS is not in use."""
    try:
        import torch

        if torch.backends.mps.is_available():
            return torch.mps.driver_allocated_memory() / 2**30
    except Exception:
        pass
    return None


def _predict(model: OptionScorer, rows, arm: str) -> list[PredictionRow]:
    """Eval-mode forward over rows -> PredictionRows (row-major, then
    question order — the same order ``forward`` emits)."""
    model.eval()
    out = model.forward([r.state for r in rows], [r.questions for r in rows])
    preds: list[PredictionRow] = []
    i = 0
    for row in rows:
        for qid in row.questions:
            probs = out[i]
            i += 1
            preds.append(
                PredictionRow(
                    row_id=row.row_id,
                    question_id=qid,
                    probs=probs,
                    confidence=max(probs.values()),
                    model=arm,
                )
            )
    assert i == len(out), f"consumed {i} of {len(out)} predictions"
    return preds


def _flatten(prefix: str, obj, into: dict) -> None:
    """Flatten nested dicts/lists of scalars for MLflow params/metrics."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(f"{prefix}.{k}" if prefix else str(k), v, into)
    elif isinstance(obj, (int, float, str, bool)):
        into[prefix] = obj


def _log_mlflow(arm: str, cfg: dict, metrics: dict, out_dir: Path) -> str | None:
    """Best-effort MLflow run tagged with the arm name. Returns run id."""
    try:
        import mlflow

        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        with mlflow.start_run(run_name=f"jeff-{arm}") as run:
            mlflow.set_tag("arm", arm)
            params: dict = {}
            _flatten("", cfg, params)
            mlflow.log_params({k: str(v) for k, v in params.items()})
            flat: dict = {}
            _flatten("", metrics, flat)
            mlflow.log_metrics(
                {k: float(v) for k, v in flat.items() if isinstance(v, (int, float))}
            )
            for name in ("metrics.json", "head.pt", "config.json", "calibration.json"):
                p = out_dir / name
                if p.exists():
                    mlflow.log_artifact(str(p))
            return run.info.run_id
    except Exception as exc:  # a dead tracker never fails a run
        print(f"[mlflow] skipped: {exc.__class__.__name__}: {exc}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", required=True, help="arm name, e.g. gt_only")
    parser.add_argument("--config", required=True, help="arm config yaml")
    parser.add_argument(
        "--data",
        action="append",
        default=None,
        help="training data file (repeatable); overrides config data.train_files",
    )
    parser.add_argument(
        "--label-source",
        default=None,
        help="keep only rows with this label_source; '' disables the filter "
        "(default: config data.label_source, else no filter)",
    )
    parser.add_argument("--out", default=None, help="default artifacts/jeff/<arm>")
    parser.add_argument(
        "--preds-out",
        default=None,
        help="default data/factcheck/preds_<arm>_val.jsonl",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(open(REPO_ROOT / args.config))
    out_dir = Path(args.out) if args.out else ARTIFACTS / args.arm
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- data ---------------------------------------------------------------
    data_files = args.data or cfg.get("data", {}).get("train_files", [])
    label_source = args.label_source
    if label_source is None:
        label_source = cfg.get("data", {}).get("label_source")
    if label_source == "":
        label_source = None

    rows = []
    for path in data_files:
        p = REPO_ROOT / path
        if not p.exists():
            raise FileNotFoundError(f"data file missing: {p}")
        rows.extend(read_rows(p))
    n_loaded = len(rows)
    if label_source:
        rows = [r for r in rows if r.label_source == label_source]

    # Test is reserved for the final cross-arm comparison: drop every
    # test-split row before train() sees the list, then assert on row ids
    # that none slipped through.
    n_pre_test = len(rows)
    rows = [r for r in rows if r.split != "test"]
    n_test_dropped = n_pre_test - len(rows)
    test_ids = {
        r.row_id
        for path in data_files
        for r in read_rows(REPO_ROOT / path)
        if r.split == "test"
    }
    used_ids = {r.row_id for r in rows}
    leaked = used_ids & test_ids
    assert not leaked, f"test rows reached the trainer: {sorted(leaked)[:5]}"
    assert all(r.split in ("train", "val") for r in rows), (
        "non-train/val split in training rows: "
        f"{sorted({r.split for r in rows})}"
    )
    print(
        f"[data] {n_loaded} rows loaded, {len(rows)} after label_source="
        f"{label_source!r} filter and test drop ({n_test_dropped} test rows "
        f"excluded; 0 test row_ids in the training set, asserted)"
    )

    # -- train --------------------------------------------------------------
    t0 = time.time()
    metrics = train(rows, cfg, out_dir)
    wall_s = time.time() - t0

    # Frozen arm layout: checkpoint files live directly in out_dir.
    ckpt_dir = out_dir / "checkpoint"
    if ckpt_dir.is_dir():
        for f in ckpt_dir.iterdir():
            f.replace(out_dir / f.name)
        ckpt_dir.rmdir()

    # -- val evaluation (never test) -----------------------------------------
    val_rows = [r for r in rows if r.split == "val"]
    model = OptionScorer.load(out_dir)  # load() returns eval mode
    assert not model.training, "loaded model must be in eval mode"
    preds = _predict(model, val_rows, args.arm)
    preds_path = Path(args.preds_out) if args.preds_out else (
        REPO_ROOT / "data" / "factcheck" / f"preds_{args.arm}_val.jsonl"
    )
    write_predictions(preds_path, preds)
    val_metrics = ground_truth_metrics(val_rows, preds)
    bins = reliability_bins(val_rows, preds, label_source="ground_truth")

    # -- metrics.json ---------------------------------------------------------
    metrics.update(
        {
            "arm": args.arm,
            "encoder": cfg.get("encoder_name"),
            "seed": int(cfg.get("seed", 42)),
            "label_source_filter": label_source,
            "checkpoint": str(out_dir),
            "val": val_metrics,
            "val_reliability_bins": bins,
            "wall_seconds": wall_s,
            "peak_rss_gb": _peak_rss_gb(),
            "mps_driver_allocated_gb": _mps_allocated_gb(),
            "preds_val": str(preds_path),
            "config": args.config,
        }
    )
    metrics_path = out_dir / "metrics.json"

    # -- MLflow (arm-tagged) ---------------------------------------------------
    # Write once without the run id so the artifact exists, log, then rewrite
    # with the run id for the on-disk record.
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str) + "\n")
    run_id = _log_mlflow(args.arm, cfg, metrics, out_dir)
    metrics["mlflow_run_id"] = run_id
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str) + "\n")

    # -- sanity: eval-mode forward on 3 val rows -------------------------------
    demo = model.forward(
        [r.state for r in val_rows[:3]], [r.questions for r in val_rows[:3]]
    )
    print("[sanity] eval-mode forward on 3 val rows:")
    for row, probs in zip(val_rows[:3], demo):
        total = sum(probs.values())
        assert abs(total - 1.0) < 1e-5, f"probs do not sum to 1: {total}"
        print(f"  {row.row_id} ({row.split}) -> {probs}  sum={total:.6f}")

    print(f"[done] arm={args.arm} checkpoint={out_dir} mlflow_run_id={run_id}")
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
