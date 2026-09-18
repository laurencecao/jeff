"""CLI: zero-shot LM label readout over the fact-check splits.

No training: an instruct LM reads the claim/evidence state plus the question
(labels + definitions in the prompt) and we read the answer distribution off
its next-token logits — see ``jeff.lm`` for the two readout rules.

Evaluates BOTH readouts (``sequence`` default, ``first_token``) on:
- ``ground_truth.jsonl`` val rows -> ``ground_truth_metrics`` + reliability bins
- ``eval_schemas.jsonl`` val rows -> ``agreement`` vs Jev's soft labels
  (cloning fidelity on held-out question wordings — NOT accuracy)

Writes ``preds_lm_<tag>_<readout>_<split>.jsonl`` under data/factcheck/ and a
JSON report under results/. Logs to MLflow experiment ``jeff`` best-effort.

Run:  uv run python scripts/jeff_lm_zeroshot.py --model Qwen/Qwen3-4B-Instruct-2507
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from jeff.eval import (  # noqa: E402
    agreement,
    ground_truth_metrics,
    reliability_bins,
)
from jeff.lm import READOUTS, LMLabeler  # noqa: E402
from jeff.schema import (  # noqa: E402
    DecisionRow,
    PredictionRow,
    read_rows,
    write_predictions,
)


def _peak_rss_gb() -> float:
    # macOS ru_maxrss is bytes.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def _tag(model_id: str) -> str:
    return model_id.split("/")[-1].lower().replace(".", "_").replace("-", "_")


def _score_gt(rows: list[DecisionRow], preds: list[PredictionRow]) -> dict[str, Any]:
    return {
        "metrics": ground_truth_metrics(rows, preds),
        "reliability_bins": reliability_bins(rows, preds, label_source="ground_truth"),
    }


def _per_row_latency(
    labeler: LMLabeler, rows: list[DecisionRow], n: int
) -> dict[str, float]:
    """p50/p95 of single-row predict latency on the first ``n`` rows."""
    lat = []
    for row in rows[:n]:
        t0 = time.perf_counter()
        labeler.predict([row])
        lat.append((time.perf_counter() - t0) * 1000.0)
    lat.sort()

    def pct(p: float) -> float:
        return lat[min(int(len(lat) * p), len(lat) - 1)]

    return {"n": len(lat), "p50_ms": pct(0.5), "p95_ms": pct(0.95)}


def _log_mlflow(report: dict[str, Any]) -> None:
    try:
        import mlflow

        mlflow.set_tracking_uri("http://127.0.0.1:5001")
        mlflow.set_experiment("jeff")
        with mlflow.start_run(run_name=f"lm_zeroshot_{report['model_tag']}"):
            mlflow.log_params(
                {
                    "model_id": report["model_id"],
                    "task": "zero-shot label readout",
                    "batch_size": report["batch_size"],
                }
            )
            flat: dict[str, float] = {}
            for readout, scored in report["ground_truth_val"].items():
                for k in ("n", "accuracy", "macro_f1", "ece", "brier"):
                    flat[f"gt_val.{readout}.{k}"] = float(scored["metrics"][k])
            for readout, ag in report["eval_schemas_val_agreement"].items():
                for k in ("n", "top1_match", "mean_tv"):
                    flat[f"evalschemas_val.{readout}.{k}"] = float(ag[k])
            for k, v in report["throughput"].items():
                if isinstance(v, (int, float)):
                    flat[f"throughput.{k}"] = float(v)
            flat["peak_rss_gb"] = report["peak_rss_gb"]
            mlflow.log_metrics(flat)
    except Exception as exc:  # noqa: BLE001 - logging must never break a run
        print(f"[mlflow] skipped: {exc.__class__.__name__}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="cap rows (smoke)")
    parser.add_argument("--latency-rows", type=int, default=30)
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="default results/jeff_lm_zeroshot_<tag>.json",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "data" / "factcheck"
    )
    parser.add_argument("--skip-agreement", action="store_true")
    args = parser.parse_args()

    tag = _tag(args.model)
    report_path = args.report or (
        REPO_ROOT / "results" / f"jeff_lm_zeroshot_{tag}.json"
    )

    gt_val = [
        r for r in read_rows(REPO_ROOT / "data" / "factcheck" / "ground_truth.jsonl")
        if r.split == "val"
    ]
    es_val = [
        r for r in read_rows(REPO_ROOT / "data" / "factcheck" / "eval_schemas.jsonl")
        if r.split == "val"
    ]
    if args.limit:
        gt_val, es_val = gt_val[: args.limit], es_val[: args.limit]
    print(f"rows: ground_truth val={len(gt_val)}  eval_schemas val={len(es_val)}")

    labeler = LMLabeler(args.model, batch_size=args.batch_size)
    print(f"loaded {args.model} on {labeler.device} dtype={labeler.model.dtype}")

    report: dict[str, Any] = {
        "model_id": args.model,
        "model_tag": tag,
        "batch_size": args.batch_size,
        "limit": args.limit,
        "readouts": list(READOUTS),
        "default_readout": labeler.readout,
    }

    # --- ground-truth val: accuracy family ---------------------------------
    t0 = time.perf_counter()
    gt_preds = labeler.predict_variants(gt_val)
    gt_s = time.perf_counter() - t0
    report["ground_truth_val"] = {
        r: _score_gt(gt_val, gt_preds[r]) for r in READOUTS
    }
    report["ground_truth_val_seconds"] = gt_s
    for r in READOUTS:
        write_predictions(
            args.out_dir / f"preds_lm_{tag}_{r}_val.jsonl", gt_preds[r]
        )

    # --- held-out schemas: agreement with Jev (NOT accuracy) ----------------
    if not args.skip_agreement:
        t0 = time.perf_counter()
        es_preds = labeler.predict_variants(es_val)
        es_s = time.perf_counter() - t0
        report["eval_schemas_val_agreement"] = {
            r: agreement(es_val, es_preds[r]) for r in READOUTS
        }
        report["eval_schemas_val_seconds"] = es_s
        for r in READOUTS:
            write_predictions(
                args.out_dir / f"preds_lm_{tag}_{r}_evalschemas_val.jsonl",
                es_preds[r],
            )

    # --- throughput ---------------------------------------------------------
    report["throughput"] = {
        "bulk_decisions_per_sec": len(gt_val) / gt_s,
        "bulk_seconds": gt_s,
        "bulk_n": len(gt_val),
        **_per_row_latency(labeler, gt_val, args.latency_rows),
    }
    report["peak_rss_gb"] = _peak_rss_gb()

    # --- print --------------------------------------------------------------
    print("\n=== zero-shot LM readout — ground_truth val ===")
    print(f"{'readout':<14} {'n':>4} {'acc':>6} {'mF1':>6} {'ece':>6} {'brier':>6}")
    for r in READOUTS:
        m = report["ground_truth_val"][r]["metrics"]
        print(
            f"{r:<14} {m['n']:>4} {m['accuracy']:>6.3f} {m['macro_f1']:>6.3f} "
            f"{m['ece']:>6.3f} {m['brier']:>6.3f}  confusion={m['confusion']}"
        )
    if "eval_schemas_val_agreement" in report:
        print("\n=== agreement with Jev on held-out schemas (NOT accuracy) ===")
        for r in READOUTS:
            a = report["eval_schemas_val_agreement"][r]
            print(
                f"{r:<14} n={a['n']} top1_match={a['top1_match']:.3f} "
                f"mean_tv={a['mean_tv']:.3f} "
                f"conf_corr={a['confidence_correctness_corr']}"
            )
    print(f"\nthroughput: {report['throughput']}")
    print(f"peak RSS: {report['peak_rss_gb']:.2f} GB")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"report -> {report_path}")
    _log_mlflow(report)


if __name__ == "__main__":
    main()
