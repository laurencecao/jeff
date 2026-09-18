"""CLI: run the external baselines (ZeroShotNLI, live JevClassifier) over the
evaluation splits, save their predictions, and score them against ground truth.

These are the comparison points for the final report. Jev is the
ceiling/reference; the NLI cross-encoder is the zero-shot floor.

Examples (from the repo root):

    uv run python -m scripts.jev_clf_baselines_run
    uv run python -m scripts.jev_clf_baselines_run --models nli --limit 20

Outputs (under --out-dir, default data/factcheck):
    preds_nli_{val,test,jaggedness}.jsonl
    preds_jev_{val,test,jaggedness}.jsonl
and a metrics report at --report (default results/jev_clf_baselines.json).

Honesty notes baked in:
  * ZeroShotNLI short-circuits empty-evidence rows to not_enough_info
    (meta["shortcut"]="no_evidence"). Metrics are reported BOTH over all rows
    and over rows excluding shortcut predictions — the all-rows number flatters
    the baseline wherever the shortcut happens to be right.
  * Jev cost/latency are measured on LIVE calls only; cache hits are reported
    separately and never counted as spend.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from jev_clf.baselines import JevClassifier, ZeroShotNLI  # noqa: E402
from jev_clf.eval import (  # noqa: E402
    ground_truth_metrics,
    jaggedness_suite,
    reliability_bins,
)
from jev_clf.schema import (  # noqa: E402
    DecisionRow,
    PredictionRow,
    label_space,
    read_rows,
    write_predictions,
)

# Jev pricing: $0.042 per million input tokens, output free.
JEV_INPUT_USD_PER_TOKEN = 0.042 / 1_000_000
EST_INPUT_TOKENS_PER_ROW = 450  # fallback when usage.input_tokens is missing

SPLITS = ("val", "test", "jaggedness")


def _pct(sorted_vals: list[float], p: float) -> float | None:
    if not sorted_vals:
        return None
    return sorted_vals[min(int(math.ceil(p * len(sorted_vals))) - 1, len(sorted_vals) - 1)]


def _check_malformed(
    preds: list[PredictionRow], rows_by_id: dict[str, DecisionRow]
) -> list[dict[str, Any]]:
    """Flag predictions whose probs do not match the question's label space."""
    bad: list[dict[str, Any]] = []
    for pred in preds:
        row = rows_by_id.get(pred.row_id)
        question = row.questions.get(pred.question_id) if row else None
        expected = set(label_space(question)) if question else None
        keys = set(pred.probs)
        problems = []
        if expected is not None and keys != expected:
            problems.append(f"label space {sorted(keys)} != {sorted(expected)}")
        if any(not math.isfinite(v) for v in pred.probs.values()):
            problems.append("non-finite probability")
        if any(v < 0 for v in pred.probs.values()):
            problems.append("negative probability")
        if sum(pred.probs.values()) <= 0:
            problems.append("zero total mass")
        if problems:
            bad.append(
                {
                    "row_id": pred.row_id,
                    "question_id": pred.question_id,
                    "problems": problems,
                }
            )
    return bad


def _predict_nli(
    nli: ZeroShotNLI, rows: list[DecisionRow]
) -> tuple[list[PredictionRow], list[dict[str, str]]]:
    """Batch predict; on failure fall back per-row so one bad row is isolated."""
    try:
        return nli.predict(rows), []
    except Exception as exc:  # noqa: BLE001 - isolate the failing row
        preds: list[PredictionRow] = []
        errors: list[dict[str, str]] = []
        for row in rows:
            try:
                preds.extend(nli.predict([row]))
            except Exception as row_exc:  # noqa: BLE001
                errors.append(
                    {
                        "row_id": row.row_id,
                        "error": f"{row_exc.__class__.__name__}: {row_exc}",
                    }
                )
        errors.insert(
            0,
            {
                "row_id": "<batch>",
                "error": f"batch predict failed, fell back per-row: "
                f"{exc.__class__.__name__}: {exc}",
            },
        )
        return preds, errors


def _predict_jev(
    clf: JevClassifier, rows: list[DecisionRow]
) -> tuple[list[PredictionRow], list[dict[str, str]], dict[str, Any]]:
    """Per-row predict so live-call spend, tokens, and latency are attributable."""
    preds: list[PredictionRow] = []
    errors: list[dict[str, str]] = []
    live_lat_ms: list[float] = []
    live_input_tokens = 0
    missing_usage = 0
    for row in rows:
        before = clf.spent_requests()
        t0 = time.perf_counter()
        try:
            row_preds = clf.predict([row])
        except Exception as exc:  # noqa: BLE001 - record, keep going
            errors.append(
                {"row_id": row.row_id, "error": f"{exc.__class__.__name__}: {exc}"}
            )
            continue
        wall_ms = (time.perf_counter() - t0) * 1000.0
        if clf.spent_requests() > before:  # a real network call happened
            live_lat_ms.append(wall_ms)
            usage = (row_preds[0].meta or {}).get("usage") if row_preds else None
            tokens = (usage or {}).get("input_tokens")
            if tokens is None:
                missing_usage += 1
                tokens = EST_INPUT_TOKENS_PER_ROW
            live_input_tokens += int(tokens)
        preds.extend(row_preds)
    accounting = {
        "live_latencies_ms": live_lat_ms,
        "live_input_tokens": live_input_tokens,
        "live_calls_missing_usage": missing_usage,
    }
    return preds, errors, accounting


def _score(
    rows: list[DecisionRow], preds: list[PredictionRow]
) -> dict[str, Any]:
    """ground_truth_metrics + reliability_bins; never hand-rolled."""
    return {
        "metrics": ground_truth_metrics(rows, preds),
        "reliability_bins": reliability_bins(rows, preds, label_source="ground_truth"),
    }


def _print_table(report: dict[str, Any]) -> None:
    print("\n=== comparison table (ground-truth metrics) ===")
    header = f"{'model':<44} {'split':<22} {'n':>4} {'acc':>6} {'mF1':>6} {'ece':>6} {'brier':>6}"
    print(header)
    print("-" * len(header))
    for model_name, m in report["models"].items():
        for split_key, scored in m["scored"].items():
            met = scored["metrics"]
            print(
                f"{model_name:<44} {split_key:<22} {met['n']:>4} "
                f"{met['accuracy']:>6.3f} {met['macro_f1']:>6.3f} "
                f"{met['ece']:>6.3f} {met['brier']:>6.3f}"
            )


def _log_mlflow(report: dict[str, Any]) -> None:
    """Best-effort MLflow logging; the tracker may be down — never fail."""
    try:
        import mlflow

        mlflow.set_tracking_uri("http://127.0.0.1:5001")
        mlflow.set_experiment("jeff")
        with mlflow.start_run(run_name="jev_clf_baselines"):
            mlflow.log_params(
                {"data": report["data"], "models": ",".join(report["models"])}
            )
            flat: dict[str, float] = {}
            for tag, m in report["models"].items():
                for split_key, scored in m["scored"].items():
                    met = scored["metrics"]
                    for k in ("n", "accuracy", "macro_f1", "ece", "brier"):
                        flat[f"{tag}.{split_key}.{k}"] = float(met[k])
                if "jev" in m:
                    flat[f"{tag}.requests_spent"] = float(m["jev"]["requests_spent"])
                    flat[f"{tag}.cost_usd"] = float(m["jev"]["cost_usd"])
            mlflow.log_metrics(flat)
    except Exception as exc:  # noqa: BLE001 - logging must never break a run
        print(f"[mlflow] skipped: {exc.__class__.__name__}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data",
        type=Path,
        default=REPO_ROOT / "data" / "factcheck" / "ground_truth.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "data" / "factcheck",
        help="where preds_<model>_<split>.jsonl are written",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "results" / "jev_clf_baselines.json",
    )
    parser.add_argument(
        "--models",
        default="nli,jev",
        help="comma subset of {nli,jev}",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap rows per split (smoke test)",
    )
    args = parser.parse_args()

    wanted = {m.strip() for m in args.models.split(",") if m.strip()}
    unknown = wanted - {"nli", "jev"}
    if unknown:
        parser.error(f"unknown models: {sorted(unknown)}")

    all_rows = read_rows(args.data)
    split_rows: dict[str, list[DecisionRow]] = {
        "val": [r for r in all_rows if r.split == "val"],
        "test": [r for r in all_rows if r.split == "test"],
        "jaggedness": jaggedness_suite(),
    }
    if args.limit is not None:
        split_rows = {k: v[: args.limit] for k, v in split_rows.items()}
    for name, rows in split_rows.items():
        if not rows:
            parser.error(f"split {name!r} is empty")

    report: dict[str, Any] = {
        "data": str(args.data),
        "limit": args.limit,
        "split_rows": {k: len(v) for k, v in split_rows.items()},
        "models": {},
    }

    runners: dict[str, Any] = {}
    if "nli" in wanted:
        print("[nli] loading cross-encoder/nli-deberta-v3-small ...")
        runners["nli"] = ZeroShotNLI()
    if "jev" in wanted:
        runners["jev"] = JevClassifier()

    for tag, model in runners.items():
        model_name = model.name
        entry: dict[str, Any] = {
            "files": {},
            "scored": {},
            "errors": {},
            "malformed": {},
        }
        if tag == "nli":
            entry["shortcut_rows"] = {}
        if tag == "jev":
            entry["jev"] = {
                "requests_spent": 0,
                "cached_rows": 0,
                "live_input_tokens": 0,
                "live_calls_missing_usage": 0,
                "cost_usd": 0.0,
                "live_latencies_ms": [],
            }

        for split_name, rows in split_rows.items():
            print(f"[{tag}] predicting {len(rows)} {split_name} rows ...")
            if tag == "nli":
                preds, errors = _predict_nli(model, rows)
            else:
                preds, errors, acct = _predict_jev(model, rows)
                j = entry["jev"]
                j["requests_spent"] += len(acct["live_latencies_ms"])
                j["cached_rows"] += len(rows) - len(acct["live_latencies_ms"]) - len(errors)
                j["live_input_tokens"] += acct["live_input_tokens"]
                j["live_calls_missing_usage"] += acct["live_calls_missing_usage"]
                j["live_latencies_ms"].extend(acct["live_latencies_ms"])

            out_path = args.out_dir / f"preds_{tag}_{split_name}.jsonl"
            written = write_predictions(out_path, preds)
            entry["files"][split_name] = {"path": str(out_path), "n": written}
            print(f"  wrote {written} preds -> {out_path}")
            if errors:
                entry["errors"][split_name] = errors
                print(f"  !! {len(errors)} row(s) raised: {errors}")

            rows_by_id = {r.row_id: r for r in rows}
            bad = _check_malformed(preds, rows_by_id)
            if bad:
                entry["malformed"][split_name] = bad
                print(f"  !! {len(bad)} malformed pred(s): {bad}")

            if tag == "nli":
                n_shortcut = sum(
                    1 for p in preds if (p.meta or {}).get("shortcut") == "no_evidence"
                )
                entry["shortcut_rows"][split_name] = n_shortcut

            entry["scored"][split_name] = _score(rows, preds)
            if tag == "nli":
                kept = [
                    p
                    for p in preds
                    if (p.meta or {}).get("shortcut") != "no_evidence"
                ]
                if kept:
                    entry["scored"][f"{split_name}_no_shortcut"] = _score(rows, kept)
                else:
                    entry["scored"][f"{split_name}_no_shortcut"] = {
                        "metrics": None,
                        "note": "every prediction used the no_evidence shortcut",
                    }

        if tag == "jev":
            j = entry["jev"]
            j["cost_usd"] = j["live_input_tokens"] * JEV_INPUT_USD_PER_TOKEN
            lats = sorted(j.pop("live_latencies_ms"))
            j["latency_ms"] = {
                "n_live": len(lats),
                "p50": _pct(lats, 0.50),
                "p95": _pct(lats, 0.95),
            }

        report["models"][model_name] = entry

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote report -> {args.report}")

    _print_table(report)

    for model_name, m in report["models"].items():
        if "shortcut_rows" in m:
            print(f"\n{model_name} no_evidence shortcut rows: {m['shortcut_rows']}")
        if "jev" in m:
            j = m["jev"]
            print(
                f"\n{model_name}: {j['requests_spent']} live requests "
                f"({j['cached_rows']} cached rows), "
                f"{j['live_input_tokens']} input tokens, "
                f"cost ${j['cost_usd']:.4f}"
                + (
                    f" [{j['live_calls_missing_usage']} call(s) missing usage,"
                    " estimated]"
                    if j["live_calls_missing_usage"]
                    else ""
                )
            )
            lat = j["latency_ms"]
            if lat["n_live"]:
                print(
                    f"  live latency: p50={lat['p50']:.0f}ms p95={lat['p95']:.0f}ms "
                    f"(n={lat['n_live']})"
                )
            else:
                print("  live latency: no live calls (all cache hits)")

    _log_mlflow(report)


if __name__ == "__main__":
    main()
