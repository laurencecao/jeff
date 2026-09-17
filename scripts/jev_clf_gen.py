"""CLI: generate jev_clf fact-check rows (synthetic oracle or Jev-distilled).

Examples (from the repo root):

    uv run python -m scripts.jev_clf_gen --n 300 --only synthetic \
        --out data/factcheck/synthetic_pilot.jsonl
    uv run python -m scripts.jev_clf_gen --n 200 --only jev \
        --out data/factcheck/distill_pilot.jsonl
    uv run python -m scripts.jev_clf_gen --n 200 --only jev --dry-run

Jev-labelled runs are resumable: rows already in the teacher's cache cost no
request, so re-running the same command only pays for what is missing.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from jev_clf.gen import generate_rows  # noqa: E402
from jev_clf.jev import JevTeacher  # noqa: E402
from jev_clf.schema import (  # noqa: E402
    FACTCHECK_QUESTION_ID,
    argmax_label,
    read_rows,
    write_rows,
)

# Jev pricing: $0.042 per million input tokens, output free. ~450 input
# tokens per fact-check call (measured on the fixture).
JEV_INPUT_USD_PER_TOKEN = 0.042 / 1_000_000
EST_INPUT_TOKENS_PER_ROW = 450
def _label_table(rows) -> dict[str, dict[str, int]]:
    """Argmax counts per answer space (choice rows and noul rows differ)."""
    tables: dict[str, Counter[str]] = {}
    for row in rows:
        labels = row.labels[FACTCHECK_QUESTION_ID]
        space = "noul" if set(labels) == {"yes", "no"} else "choice"
        tables.setdefault(space, Counter())[argmax_label(labels)] += 1
    return {space: dict(counts) for space, counts in tables.items()}


def _schema_report(rows) -> dict[str, list[str]]:
    """Schema ids observed per split — the wording-holdout audit."""
    by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for row in rows:
        by_split[row.split].add(row.meta["schema_id"])
    return {split: sorted(ids) for split, ids in by_split.items()}


def _check(rows, path: Path | None) -> None:
    """Invariants every emitted file must satisfy."""
    assert rows, "no rows generated"
    group_split: dict[str, str] = {}
    for row in rows:
        labels = row.labels[FACTCHECK_QUESTION_ID]
        total = sum(labels.values())
        assert abs(total - 1.0) < 1e-6, f"{row.row_id}: labels sum to {total}"
        prev = group_split.setdefault(row.group_id, row.split)
        assert prev == row.split, f"{row.row_id}: group {row.group_id} straddles splits"
    # Schema holdout: no schema id may appear in more than one split.
    schemas = _schema_report(rows)
    train_ids = set(schemas["train"])
    assert not (train_ids & set(schemas["val"])), "schema leak: train ∩ val"
    assert not (train_ids & set(schemas["test"])), "schema leak: train ∩ test"
    if path is not None and path.exists():
        reloaded = read_rows(path)
        assert len(reloaded) == len(rows), (
            f"reload mismatch: {len(reloaded)} rows in {path}, expected {len(rows)}"
        )


def _log_mlflow(n: int, args: argparse.Namespace, tables: dict[str, dict[str, int]]) -> None:
    """Best-effort MLflow logging; the tracker may be down — never fail."""
    try:
        import mlflow

        mlflow.set_tracking_uri("http://127.0.0.1:5001")
        mlflow.set_experiment("jev-clf")
        with mlflow.start_run(run_name="jev_clf_gen"):
            mlflow.log_params(
                {"n": n, "seed": args.seed, "only": args.only, "out": str(args.out)}
            )
            mlflow.log_metrics(
                {f"{space}_{label}": count for space, counts in tables.items()
                 for label, count in counts.items()}
            )
    except Exception as exc:  # noqa: BLE001 - logging must never break a run
        print(f"[mlflow] skipped: {exc.__class__.__name__}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, required=True, help="rows to generate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--only",
        choices=["synthetic", "jev", "mixed"],
        default="mixed",
        help="synthetic: oracle labels, no network; jev: teacher labels; "
        "mixed: alternate the two",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "data" / "factcheck" / "generated.jsonl",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan and cost estimate; call nothing, write nothing",
    )
    args = parser.parse_args()

    only = None if args.only == "mixed" else args.only
    jev_fraction = {"synthetic": 0.0, "jev": 1.0, "mixed": 0.5}[args.only]
    est_requests = int(args.n * jev_fraction)
    est_cost = est_requests * EST_INPUT_TOKENS_PER_ROW * JEV_INPUT_USD_PER_TOKEN

    if args.dry_run:
        rows = list(generate_rows(args.n, seed=args.seed, only="synthetic"))
        tables = _label_table(rows)
        print(f"dry run: {args.n} rows planned, only={args.only}")
        print(f"  oracle argmax counts: {tables}")
        print(f"  schemas per split: {_schema_report(rows)}")
        print(f"  estimated Jev requests: {est_requests}")
        print(f"  estimated cost: ${est_cost:.4f}")
        return

    teacher = JevTeacher() if args.only in ("jev", "mixed") else None
    rows = list(generate_rows(args.n, seed=args.seed, teacher=teacher, only=only))

    written = write_rows(args.out, rows)
    _check(rows, args.out)
    tables = _label_table(rows)
    print(f"wrote {written} rows -> {args.out}")
    print(f"  argmax label counts: {tables}")
    print(f"  schemas per split: {_schema_report(rows)}")
    choice_counts = tables.get("choice", {})
    n_choice = sum(choice_counts.values())
    assert n_choice > 0, "no choice-schema rows generated"
    for label, count in sorted(choice_counts.items()):
        frac = count / n_choice
    if teacher is not None:
        spent = teacher.spent_requests()
        cost = spent * EST_INPUT_TOKENS_PER_ROW * JEV_INPUT_USD_PER_TOKEN
        print(f"  teacher requests spent: {spent} (cached: {written - spent})")
        print(f"  estimated cost this run: ${cost:.4f}")
    _log_mlflow(written, args, tables)


if __name__ == "__main__":
    main()
