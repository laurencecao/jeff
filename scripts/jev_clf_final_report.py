"""Consolidate every model's results into one honest comparison table.

Two numbers that must never be confused, and are therefore printed in separate
sections:

  * ACCURACY vs human ground-truth labels  -> is the model any good?
  * AGREEMENT with Jev on held-out question wordings -> is it a faithful clone?

A model can win one and lose the other, and averaging them would hide exactly
the thing this experiment set out to measure.

Every prediction file is routed by WHICH row set its row_ids actually match,
never by its filename, so a predictions file can never be scored against the
wrong rows. Partial coverage is stated explicitly rather than passed off as a
full result.

    uv run python -m scripts.jev_clf_final_report
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jev_clf import schema as S  # noqa: E402
from jev_clf.eval import agreement as agreement_metric  # noqa: E402
from jev_clf.eval import ground_truth_metrics  # noqa: E402

DATA = ROOT / "data" / "factcheck"
RESULTS = ROOT / "results"
ARTIFACTS = ROOT / "artifacts" / "jeff"


def by_split(path: Path) -> dict[str, list[S.DecisionRow]]:
    out: dict[str, list[S.DecisionRow]] = {}
    for r in S.read_rows(path):
        out.setdefault(r.split, []).append(r)
    return out


def match_rows(preds: list[S.PredictionRow], candidates: dict[str, list[S.DecisionRow]]):
    """Return (split, matched_rows) for the row set this prediction file actually covers."""
    by_id: dict[str, list[S.DecisionRow]] = {}
    for split, rows in candidates.items():
        for r in rows:
            by_id.setdefault(r.row_id, []).append(r)
    matched: list[S.DecisionRow] = []
    seen: set[str] = set()
    for p in preds:
        for r in by_id.get(p.row_id, []):
            if r.row_id not in seen:
                matched.append(r)
                seen.add(r.row_id)
    splits = {r.split for r in matched}
    return (next(iter(splits)) if len(splits) == 1 else "mixed"), matched


def main() -> None:
    gt = by_split(DATA / "ground_truth.jsonl")
    ev = by_split(DATA / "eval_schemas.jsonl")

    acc_lines = ["| model / arm | split | n | coverage | accuracy | macro-F1 | ECE | Brier | source |",
                 "|---|---|---|---|---|---|---|---|---|"]
    agr_lines = ["| model / arm | split | n | coverage | top-1 match | mean TV | source |",
                 "|---|---|---|---|---|---|---|"]
    unmatched: list[str] = []

    for preds_file in sorted(DATA.glob("preds_*.jsonl")):
        preds = S.read_predictions(preds_file)
        if not preds:
            continue
        model = preds[0].model

        split, gmatch = match_rows(preds, gt)
        if gmatch:
            met = ground_truth_metrics(gmatch, preds)
            cov = f"{met['n']}/{len(gt[split])}"
            acc_lines.append(
                f"| {model} | {split} | {met['n']} | {cov} | {met['accuracy']:.3f} | "
                f"{met['macro_f1']:.3f} | {met['ece']:.3f} | {met['brier']:.3f} | `{preds_file.name}` |"
            )
            continue

        split, ematch = match_rows(preds, ev)
        if ematch:
            agg = agreement_metric(ematch, preds)
            cov = f"{agg['n']}/{len(ev[split])}"
            agr_lines.append(
                f"| {model} | {split} | {agg['n']} | {cov} | {agg['top1_match']:.3f} | "
                f"{agg['mean_tv']:.3f} | `{preds_file.name}` |"
            )
            continue

        unmatched.append(f"{preds_file.name} (n={len(preds)}, model={model})")

    for res_file in sorted(RESULTS.glob("lm_eval_*.json")):
        d = json.loads(res_file.read_text())
        g = d.get("ground_truth")
        if g:
            acc_lines.append(
                f"| {d['model']} ({d.get('readout', '?')}) | {d.get('split', 'val')} | {g['n']} | "
                f"{g['n']}/199 | {g['accuracy']:.3f} | {g['macro_f1']:.3f} | {g['ece']:.3f} | "
                f"{g['brier']:.3f} | `{res_file.name}` |"
            )
        a = d.get("agreement_with_jev")
        if a:
            agr_lines.append(
                f"| {d['model']} ({d.get('readout', '?')}) | {d.get('split', 'val')} | {a['n']} | "
                f"{a['n']}/398 | {a['top1_match']:.3f} | {a['mean_tv']:.3f} | `{res_file.name}` |"
            )

    for met_file in sorted(ARTIFACTS.glob("*/metrics.json")):
        d = json.loads(met_file.read_text())
        v = d.get("val")
        if not v:
            continue
        acc_lines.append(
            f"| arm:{met_file.parent.name} | val | {v['n']} | {v['n']}/199 | {v['accuracy']:.3f} | "
            f"{v['macro_f1']:.3f} | {v['ece']:.3f} | {v['brier']:.3f} | "
            f"`artifacts/jeff/{met_file.parent.name}/metrics.json` |"
        )

    lines = [
        "# jeff — final comparison",
        "",
        "Two numbers, never to be confused: **accuracy** against human labels (is it any good?) "
        "and **agreement with Jev** (is it a faithful clone?). A model can win one and lose the other.",
        "",
        "## 1. Accuracy against human ground-truth labels",
        "",
        "The non-circular number. `coverage` is scored rows / rows available on that split — a "
        "partial run is visible as e.g. `24/199`, not passed off as a full result.",
        "",
        *acc_lines,
        "",
        "## 2. Agreement with Jev on HELD-OUT question wordings (cloning)",
        "",
        "Imitation of the teacher on schemas never seen in training. This is NOT a quality number "
        "and must not be read as one.",
        "",
        *agr_lines,
        "",
        "## 3. Caveats that must travel with the numbers above",
        "",
        "- **The jaggedness suite does not discriminate.** Live Jev scored 9/9 on it, so it cannot "
        "demonstrate whether a student inherits the teacher's failure modes. Any claim about "
        "inherited jaggedness is untested.",
        "- **The NLI baseline short-circuits empty-evidence rows** to `not_enough_info` "
        '(`meta.shortcut="no_evidence"`), mildly flattering its all-rows accuracy. The no-shortcut '
        "variant is in `results/jev_clf_baselines.json`.",
        "- **Calibration is fit on val only**, never test.",
        "- **The frozen-MiniLM arms are controls, not candidates.** They show what a "
        "sentence-similarity encoder plus a small unnormalized head achieves; the head has no input "
        "normalization and is therefore not scale-robust across encoders (Qwen's hidden-state "
        "absmax ~220 saturates its attention logits at init).",
        "- **Live Jev is only ~0.80 accurate on this data**, so teacher-distilled soft targets "
        "inject roughly 20% label noise relative to ground truth.",
        "",
    ]
    if unmatched:
        lines += ["## 4. Prediction files that matched NEITHER row set (not scored)", ""]
        lines += [f"- {u}" for u in unmatched]
        lines += [""]

    out = RESULTS / "jev_clf_final.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nwrote", out)


if __name__ == "__main__":
    main()
