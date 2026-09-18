"""CLI: scale Jev-distilled training data and build the held-out-schema eval set.

The wave-1 generator draws states from a 40-fact synthetic pool — too
repetitive to train a generalizing student. This script keeps the generator's
question schemas but swaps the state pool for real ones:

  * ``distill_full.jsonl``  — TRAIN ONLY. States = train-split ground-truth
    rows (claim + evidence) plus the generator's synthetic train states,
    each asked under the six TRAIN schemas (fc-c0,c3,c4,c5,c6,c7). Labels are
    Jev's full soft distribution (``label_source="jev-1.13.0"``).
  * ``eval_schemas.jsonl``  — EVAL ONLY. Val/test ground-truth states asked
    under the HELD-OUT schemas (val: fc-c2,fc-n1; test: fc-c1,fc-n0). These
    measure schema generalization on real facts; they are never trained on.

Usage (from the repo root):

    uv run python -m scripts.jev_clf_distill_scale --dry-run
    uv run python -m scripts.jev_clf_distill_scale --target 7000

Runs are resumable through the teacher's append-only cache: a re-run only
pays for rows whose (state, schema) pair is not already cached. The run
aborts before spending if the pre-flight estimate exceeds --budget, and
stops mid-run if live spend crosses it.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from jev_clf import gen  # noqa: E402
from jev_clf.jev import JevTeacher  # noqa: E402
from jev_clf.schema import (  # noqa: E402
    FACTCHECK_LABELS,
    FACTCHECK_QUESTION_ID,
    DecisionRow,
    argmax_label,
    read_rows,
    write_rows,
)

GT_PATH = REPO_ROOT / "data" / "factcheck" / "ground_truth.jsonl"
TRAIN_OUT = REPO_ROOT / "data" / "factcheck" / "distill_full.jsonl"
EVAL_OUT = REPO_ROOT / "data" / "factcheck" / "eval_schemas.jsonl"

# Jev pricing: $0.042 per million input tokens, output free.
JEV_INPUT_USD_PER_TOKEN = 0.042 / 1_000_000
EST_INPUT_TOKENS_PER_ROW = 450  # measured on the fixture/pilot cache

# One state may be asked under at most one row per train schema — a state
# paired with all six train schemas is the hard ceiling, and we stay under it.
MAX_ROWS_PER_GT_STATE = 6
MAX_ROWS_PER_SYNTH_STATE = 3
# Fraction of each label quota filled from the synthetic pool (adversarial
# arms: empty evidence, injections, self-contradiction — things GT lacks).
SYNTH_SHARE = 0.10

TRAIN_SCHEMAS = ("fc-c0", "fc-c3", "fc-c4", "fc-c5", "fc-c6", "fc-c7")
HELD_OUT_SCHEMAS = {"val": ("fc-c2", "fc-n1"), "test": ("fc-c1", "fc-n0")}

# Per-row retries on top of the teacher's own retry policy: a short Jev
# outage (503s) must not kill a multi-thousand-row run.
MAX_ROW_ATTEMPTS = 10


def _state_key(state: Any) -> str:
    """Canonical identity of a state, for dedup and leak checks."""
    return json.dumps(state, sort_keys=True)


def _gt_label(row: DecisionRow) -> str:
    return argmax_label(row.labels[FACTCHECK_QUESTION_ID])


# ---------------------------------------------------------------------------
# Plan construction (no network)
# ---------------------------------------------------------------------------


def _schema_cycle(schema_ids: list[str], offset: int, k: int) -> list[str]:
    """`k` distinct schema ids starting at `offset`, round-robin."""
    n = len(schema_ids)
    return [schema_ids[(offset + j) % n] for j in range(k)]


def _build_train_plan(
    gt_train: list[DecisionRow],
    synth_rows: list[DecisionRow],
    target: int,
    seed: int,
) -> tuple[list[DecisionRow], dict[str, Any]]:
    """Pick (state, schema) pairs so Jev argmax labels land near thirds.

    GT states carry their ground-truth label as the proxy for the label Jev
    will assign; synthetic states carry the construction oracle. Each stratum
    fills ~target/3 rows: ~SYNTH_SHARE from synthetic states (cap 3 each),
    the rest from GT states of that label (cap MAX_ROWS_PER_GT_STATE each,
    never repeating a (state, schema) pair). Shortfall in one stratum rolls
    into the next.
    """
    import random

    rng = random.Random(seed)
    schemas = list(TRAIN_SCHEMAS)

    gt_by_label: dict[str, list[DecisionRow]] = defaultdict(list)
    for row in gt_train:
        gt_by_label[_gt_label(row)].append(row)

    # Distinct synthetic train states per oracle label.
    synth_by_label: dict[str, dict[str, DecisionRow]] = defaultdict(dict)
    for row in synth_rows:
        if row.split != "train":
            continue
        synth_by_label[row.meta["oracle_label"]].setdefault(
            _state_key(row.state), row
        )

    plan: list[DecisionRow] = []
    used_pairs: set[tuple[str, str]] = set()
    stats: dict[str, Any] = {"gt_state_cap": MAX_ROWS_PER_GT_STATE,
                           "synth_state_cap": MAX_ROWS_PER_SYNTH_STATE,
                           "strata": {}}
    remaining = target
    labels = list(FACTCHECK_LABELS)
    for li, label in enumerate(labels):
        quota = remaining // (len(labels) - li)
        got = 0
        synth_quota = min(
            int(round(quota * SYNTH_SHARE)),
            len(synth_by_label[label]) * MAX_ROWS_PER_SYNTH_STATE,
        )

        # --- synthetic slice ---
        synth_states = list(synth_by_label[label].values())
        rng.shuffle(synth_states)
        per_state = max(1, -(-synth_quota // max(1, len(synth_states))))
        per_state = min(per_state, MAX_ROWS_PER_SYNTH_STATE)
        n_synth = 0
        for si, srow in enumerate(synth_states):
            if n_synth >= synth_quota:
                break
            for sid in _schema_cycle(schemas, si, per_state):
                if n_synth >= synth_quota:
                    break
                key = (_state_key(srow.state), sid)
                if key in used_pairs:
                    continue
                used_pairs.add(key)
                schema = gen._SCHEMA_BY_ID[sid]
                plan.append(DecisionRow(
                    row_id=f"distill-{len(plan):05d}",
                    source="jev_distill",
                    split="train",
                    group_id=f"syn-{srow.group_id}",
                    state=srow.state,
                    questions=gen._build_question(schema),
                    labels={},  # filled by the teacher
                    label_source="jev-1.13.0",
                    meta={
                        "schema_id": sid,
                        "state_kind": "synthetic",
                        "arm": srow.meta["arm"],
                        "domain": srow.meta["domain"],
                        "oracle_label": srow.meta["oracle_label"],
                    },
                ))
                n_synth += 1
                got += 1

        # --- ground-truth slice ---
        gt_quota = quota - got
        gt_states = gt_by_label[label]
        rng.shuffle(gt_states)
        per_state = min(
            MAX_ROWS_PER_GT_STATE, -(-gt_quota // max(1, len(gt_states)))
        )
        n_gt = 0
        for si, grow in enumerate(gt_states):
            if n_gt >= gt_quota:
                break
            for sid in _schema_cycle(schemas, si, per_state):
                if n_gt >= gt_quota:
                    break
                key = (_state_key(grow.state), sid)
                if key in used_pairs:
                    continue
                used_pairs.add(key)
                schema = gen._SCHEMA_BY_ID[sid]
                plan.append(DecisionRow(
                    row_id=f"distill-{len(plan):05d}",
                    source=grow.source,
                    split="train",
                    group_id=grow.group_id,
                    state=grow.state,
                    questions=gen._build_question(schema),
                    labels={},
                    label_source="jev-1.13.0",
                    meta={
                        "schema_id": sid,
                        "state_kind": "ground_truth",
                        "gt_row_id": grow.row_id,
                        "gt_label": label,
                        "dataset": grow.meta.get("dataset"),
                    },
                ))
                n_gt += 1
                got += 1

        stats["strata"][label] = {
            "quota": quota, "planned": got,
            "gt_rows": n_gt, "synth_rows": n_synth,
        }
        remaining -= got

    rng.shuffle(plan)  # interleave strata so any prefix is representative
    for i, row in enumerate(plan):
        row.row_id = f"distill-{i:05d}"
    return plan, stats


def _build_eval_plan(gt_eval: list[DecisionRow]) -> list[DecisionRow]:
    """Every val/test GT state x its split's held-out schemas."""
    plan: list[DecisionRow] = []
    for grow in gt_eval:
        for sid in HELD_OUT_SCHEMAS[grow.split]:
            schema = gen._SCHEMA_BY_ID[sid]
            plan.append(DecisionRow(
                row_id=f"{grow.row_id}--{sid}",
                source=grow.source,
                split=grow.split,
                group_id=grow.group_id,
                state=grow.state,
                questions=gen._build_question(schema),
                labels={},
                label_source="jev-1.13.0",
                meta={
                    "schema_id": sid,
                    "state_kind": "ground_truth",
                    "gt_row_id": grow.row_id,
                    "gt_label": _gt_label(grow),
                    "dataset": grow.meta.get("dataset"),
                },
            ))
    return plan


# ---------------------------------------------------------------------------
# Budgeted parallel labelling
# ---------------------------------------------------------------------------


class BudgetExceeded(RuntimeError):
    pass


class _Budget:
    """Shared live-spend tracker across worker threads."""

    def __init__(self, budget_usd: float):
        self.budget_usd = budget_usd
        self.lock = threading.Lock()
        self.live_requests = 0
        self.cached_requests = 0
        self.input_tokens = 0

    def check(self) -> None:
        with self.lock:
            est = (self.input_tokens + EST_INPUT_TOKENS_PER_ROW) * JEV_INPUT_USD_PER_TOKEN
            if est > self.budget_usd:
                raise BudgetExceeded(
                    f"live spend ${self.input_tokens * JEV_INPUT_USD_PER_TOKEN:.4f} "
                    f"reached budget ${self.budget_usd:.2f}"
                )

    def add(self, usage: dict[str, Any], cached: bool) -> None:
        with self.lock:
            if cached:
                self.cached_requests += 1
            else:
                self.live_requests += 1
                self.input_tokens += int(usage.get("input_tokens") or 0)

    def cost_usd(self) -> float:
        return self.input_tokens * JEV_INPUT_USD_PER_TOKEN


def _label_chunk(
    rows: list[DecisionRow], budget: _Budget, out: list[DecisionRow],
    failed: list[tuple[str, str]],
) -> None:
    """Worker: label `rows` with a private teacher (shared disk cache).

    Each row gets its own retry loop on top of the teacher's: a short Jev
    outage (503s) must not kill a 7000-row run. Rows that still fail after
    MAX_ROW_ATTEMPTS are recorded in `failed` and skipped.
    """
    import time

    teacher = JevTeacher()
    for row in rows:
        try:
            budget.check()
        except BudgetExceeded:
            return
        attempt = 0
        while True:
            try:
                labelled = next(teacher.ask_rows([row]))
                break
            except Exception as exc:  # noqa: BLE001 - retried below
                attempt += 1
                if attempt >= MAX_ROW_ATTEMPTS:
                    failed.append((row.row_id, f"{exc.__class__.__name__}: {exc}"))
                    labelled = None
                    break
                time.sleep(min(2.0 * attempt, 30.0))
        if labelled is None:
            continue
        t = labelled.meta["teacher"]
        budget.add(t.get("usage") or {}, bool(t.get("cached")))
        out.append(labelled)


def _label_plan(
    plan: list[DecisionRow], budget: _Budget, workers: int
) -> tuple[list[DecisionRow], list[tuple[str, str]], bool]:
    """Label every planned row; returns (rows, failures, hit_budget)."""
    out: list[DecisionRow] = []
    failed: list[tuple[str, str]] = []
    chunks = [plan[i::workers] for i in range(workers)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda c: _label_chunk(c, budget, out, failed), chunks))
    hit_budget = len(out) < len(plan)
    # Restore plan order (threads append out of order).
    order = {r.row_id: i for i, r in enumerate(plan)}
    out.sort(key=lambda r: order[r.row_id])
    return out, failed, hit_budget


# ---------------------------------------------------------------------------
# Checks and tables
# ---------------------------------------------------------------------------


def _label_table(rows: list[DecisionRow]) -> dict[str, dict[str, int]]:
    tables: dict[str, Counter[str]] = {}
    for row in rows:
        labels = row.labels[FACTCHECK_QUESTION_ID]
        space = "noul" if set(labels) == {"yes", "no"} else "choice"
        tables.setdefault(space, Counter())[argmax_label(labels)] += 1
    return {s: dict(c) for s, c in tables.items()}


def _schema_table(rows: list[DecisionRow]) -> dict[str, int]:
    return dict(Counter(r.meta["schema_id"] for r in rows))


def _check_train(rows: list[DecisionRow], eval_rows: list[DecisionRow],
                 gt_eval_states: set[str]) -> None:
    assert rows, "no training rows"
    eval_pairs = {(_state_key(r.state), r.meta["schema_id"]) for r in eval_rows}
    shares: Counter[str] = Counter()
    for row in rows:
        assert row.split == "train", f"{row.row_id}: split {row.split}"
        sid = row.meta["schema_id"]
        assert sid in TRAIN_SCHEMAS, f"{row.row_id}: schema {sid} not in train set"
        assert (_state_key(row.state), sid) not in eval_pairs, (
            f"{row.row_id}: (state, schema) also in eval set"
        )
        assert _state_key(row.state) not in gt_eval_states, (
            f"{row.row_id}: val/test GT state leaked into training data"
        )
        total = sum(row.labels[FACTCHECK_QUESTION_ID].values())
        assert abs(total - 1.0) < 1e-6, f"{row.row_id}: labels sum to {total}"
        shares[argmax_label(row.labels[FACTCHECK_QUESTION_ID])] += 1
    n = len(rows)
    for label in FACTCHECK_LABELS:
        share = shares[label] / n
        assert 0.10 <= share <= 0.60, (
            f"degenerate label balance: {label} share {share:.3f}"
        )


def _check_eval(rows: list[DecisionRow]) -> None:
    assert rows, "no eval rows"
    for row in rows:
        assert row.split in ("val", "test"), f"{row.row_id}: split {row.split}"
        sid = row.meta["schema_id"]
        assert sid in HELD_OUT_SCHEMAS[row.split], (
            f"{row.row_id}: schema {sid} not held out for {row.split}"
        )
        total = sum(row.labels[FACTCHECK_QUESTION_ID].values())
        assert abs(total - 1.0) < 1e-6, f"{row.row_id}: labels sum to {total}"
    # Non-degenerate argmax on the choice-schema eval rows.
    choice = [r for r in rows if set(r.labels[FACTCHECK_QUESTION_ID]) != {"yes", "no"}]
    shares: Counter[str] = Counter(
        argmax_label(r.labels[FACTCHECK_QUESTION_ID]) for r in choice
    )
    for label in FACTCHECK_LABELS:
        share = shares[label] / len(choice)
        assert 0.10 <= share <= 0.60, (
            f"degenerate eval label balance: {label} share {share:.3f}"
        )


def _report(name: str, rows: list[DecisionRow]) -> None:
    print(f"\n== {name}: {len(rows)} rows ==")
    print("  per-label argmax:", _label_table(rows))
    print("  per-schema:", _schema_table(rows))
    print("  per-source:", dict(Counter(r.source for r in rows)))
    print("  per-split:", dict(Counter(r.split for r in rows)))


def _log_mlflow(args: argparse.Namespace, n_train: int, n_eval: int,
                budget: _Budget) -> None:
    try:
        import mlflow

        mlflow.set_tracking_uri("http://127.0.0.1:5001")
        mlflow.set_experiment("jeff")
        with mlflow.start_run(run_name="jev_clf_distill_scale"):
            mlflow.log_params({
                "target": args.target, "seed": args.seed,
                "workers": args.workers, "budget": args.budget,
            })
            mlflow.log_metrics({
                "train_rows": n_train, "eval_rows": n_eval,
                "live_requests": budget.live_requests,
                "cached_requests": budget.cached_requests,
                "input_tokens": budget.input_tokens,
                "cost_usd": budget.cost_usd(),
            })
    except Exception as exc:  # noqa: BLE001 - logging must never break a run
        print(f"[mlflow] skipped: {exc.__class__.__name__}: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", type=int, default=7000,
                        help="approximate rows for distill_full.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--budget", type=float, default=0.50,
                        help="max live Jev spend in USD; aborts beyond this")
    parser.add_argument("--train-out", type=Path, default=TRAIN_OUT)
    parser.add_argument("--eval-out", type=Path, default=EVAL_OUT)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and estimate; call nothing")
    args = parser.parse_args()

    gt = read_rows(GT_PATH)
    gt_train = [r for r in gt if r.split == "train"]
    gt_eval = [r for r in gt if r.split != "train"]
    gt_eval_states = {_state_key(r.state) for r in gt_eval}
    print(f"ground truth: {len(gt_train)} train states, "
          f"{len(gt_eval)} val/test states")

    # Synthetic train states: one full fact x arm cycle, oracle-labelled,
    # no network. ~360 distinct states, ~288 in the train split.
    synth_rows = list(gen.generate_rows(360, seed=args.seed, only="synthetic"))
    n_synth_train = sum(1 for r in synth_rows if r.split == "train")
    print(f"synthetic pool: {n_synth_train} train states "
          f"(of {len(synth_rows)} generated)")

    train_plan, stats = _build_train_plan(gt_train, synth_rows, args.target, args.seed)
    eval_plan = _build_eval_plan(gt_eval)
    print(f"plan: {len(train_plan)} train rows, {len(eval_plan)} eval rows")
    print(f"  state caps: gt<={stats['gt_state_cap']} "
          f"synth<={stats['synth_state_cap']} (one row per (state, schema))")
    for label, s in stats["strata"].items():
        print(f"  stratum {label}: quota {s['quota']} -> "
              f"{s['gt_rows']} gt + {s['synth_rows']} synth = {s['planned']}")

    n_calls = len(train_plan) + len(eval_plan)
    est_cost = n_calls * EST_INPUT_TOKENS_PER_ROW * JEV_INPUT_USD_PER_TOKEN
    print(f"estimated: {n_calls} teacher calls, ~${est_cost:.4f} "
          f"(cache hits reduce this)")
    if args.dry_run:
        return
    if est_cost > args.budget:
        print(f"ABORT: estimate ${est_cost:.4f} exceeds budget "
              f"${args.budget:.2f}; nothing spent, nothing written")
        return

    budget = _Budget(args.budget)
    print(f"\nlabelling {len(train_plan)} train rows "
          f"({args.workers} workers, budget ${args.budget:.2f}) ...")
    train_rows, train_failed, train_cut = _label_plan(
        train_plan, budget, args.workers)
    print(f"  train: {len(train_rows)}/{len(train_plan)} labelled"
          + ("  [BUDGET CUT]" if train_cut else ""))
    for rid, err in train_failed:
        print(f"    FAILED {rid}: {err}")

    eval_rows: list[DecisionRow] = []
    eval_cut = False
    if not train_cut:
        print(f"labelling {len(eval_plan)} eval rows ...")
        eval_rows, eval_failed, eval_cut = _label_plan(
            eval_plan, budget, args.workers)
        print(f"  eval: {len(eval_rows)}/{len(eval_plan)} labelled"
              + ("  [BUDGET CUT]" if eval_cut else ""))
        for rid, err in eval_failed:
            print(f"    FAILED {rid}: {err}")

    total_req = budget.live_requests + budget.cached_requests
    hit_rate = budget.cached_requests / total_req if total_req else 0.0
    print(f"\nteacher calls: {budget.live_requests} live, "
          f"{budget.cached_requests} cached (hit rate {hit_rate:.1%})")
    print(f"actual cost: ${budget.cost_usd():.4f} "
          f"({budget.input_tokens} input tokens)")

    if train_cut or eval_cut:
        print("\nBUDGET EXCEEDED before completion — partial files NOT written.")
        _report("train (partial, unwritten)", train_rows)
        _report("eval (partial, unwritten)", eval_rows)
        _log_mlflow(args, len(train_rows), len(eval_rows), budget)
        return

    _check_train(train_rows, eval_rows, gt_eval_states)
    _check_eval(eval_rows)
    print("\nall schema-holdout / leak / balance assertions passed")

    write_rows(args.train_out, train_rows)
    write_rows(args.eval_out, eval_rows)
    # Reload check: files must round-trip through the frozen schema.
    assert len(read_rows(args.train_out)) == len(train_rows)
    assert len(read_rows(args.eval_out)) == len(eval_rows)
    print(f"wrote {len(train_rows)} -> {args.train_out}")
    print(f"wrote {len(eval_rows)} -> {args.eval_out}")

    _report("distill_full", train_rows)
    _report("eval_schemas", eval_rows)
    _log_mlflow(args, len(train_rows), len(eval_rows), budget)


if __name__ == "__main__":
    main()
