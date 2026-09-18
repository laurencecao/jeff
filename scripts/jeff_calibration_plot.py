"""Reliability diagrams for our model vs live Jev, on human labels.

This is the plot TypeSafe never published: measured calibration on a dataset
with human ground truth, far from their four launch workflows.

    uv run python -m scripts.jeff_calibration_plot
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jeff import schema as S  # noqa: E402
from jeff.eval import reliability_bins  # noqa: E402

FILES = [
    ("ours (4B + LoRA), n=9730", ROOT / "results/lm_eval_4b_multi_large.json"),
    ("live Jev 1.13.0, n=9730", ROOT / "results/jev_large.json"),
]


def bins_from_preds(ours_path: Path, jev_path: Path, rows_path: Path):
    """Compute (conf, acc) reliability points from per-row predictions.

    Both models' summaries lack reliability bins, so recompute them from the
    per-row predictions against the human labels.
    """
    out = {}
    gt = {r.row_id: r for r in S.read_rows(rows_path)}
    for name, ppath in (("ours", ours_path), ("jev", jev_path)):
        p = Path(ppath)
        if not p.exists():
            print(f"({name}: {p.name} not found, skipping)")
            continue
        preds = S.read_predictions(p)
        conf, correct = [], []
        for pr in preds:
            row = gt.get(pr.row_id)
            if row is None:
                continue
            qid = pr.question_id
            row_q = row.questions.get(qid)
            if row_q is None:
                continue
            labels = S.label_space(row_q)
            target = S.argmax_label(row.targets(qid))
            top = max(pr.probs.items(), key=lambda kv: kv[1])
            conf.append(float(top[1]))
            correct.append(1.0 if top[0] == target else 0.0)
        if not conf:
            continue
        conf = np.array(conf)
        correct = np.array(correct)
        edges = np.linspace(0.0, 1.0, 11)
        pts_c, pts_a = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
            if m.sum() > 0:
                pts_c.append(float(conf[m].mean()))
                pts_a.append(float(correct[m].mean()))
        out[name] = (pts_c, pts_a)
    return out


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    titles = ["Calibration: ours (4B + LoRA) vs live Jev", "Reliability curve"]

    plotted = 0
    # Left panel is a bar chart of ECE (units: error, 0-1); the y=x diagonal is
    # meaningless there and visually implies the bars are "below the line".
    # It belongs only on the right panel, where both axes are probabilities.
    axes[0].set_ylabel("ECE")
    axes[0].set_xlabel("")          # bars are models, not a confidence axis
    axes[0].set_title(titles[0], fontsize=11)
    axes[0].grid(alpha=0.3, axis="y")
    axes[1].set_xlabel("stated confidence")
    axes[1].set_title(titles[1], fontsize=11)
    axes[1].grid(alpha=0.3)
    axes[1].plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    axes[1].set_ylabel("empirical accuracy")

    series = bins_from_preds(
        ROOT / "data/factcheck/preds_ours_large.jsonl",
        ROOT / "data/factcheck/preds_jev_large.jsonl",
        ROOT / "data/factcheck/eval_large.jsonl",
    )

    summary_ece = {}
    for label, path in FILES:
        if path.exists():
            summary_ece[label] = json.loads(path.read_text()).get("ground_truth", {}).get("ece")

    for i, (label, path) in enumerate(FILES):
        d = json.loads(path.read_text())
        g = d.get("ground_truth", {})
        ece = g.get("ece")
        if ece is None:
            # Jev's summary stores metrics at the top level, not under
            # ground_truth; accept either so the bar is never silently dropped.
            ece = d.get("ece")
        color = ["#1f77b4", "#d62728"][i % 2]

        shown = ece if ece is not None else summary_ece.get(label)
        if shown is not None:
            axes[0].bar([i], [shown], color=color, width=0.45)
            axes[0].text(i, shown + 0.004, f"{shown:.4f}", ha="center", fontsize=10)

        key = "ours" if i == 0 else "jev"
        if key in series:
            conf, accs = series[key]
            axes[1].plot(conf, accs, "o-", color=color, label=f"{label}  (ECE {ece if ece is None else round(ece,4)})")
            plotted += 1

    if plotted == 0:
        print("could not compute reliability curves from per-row predictions")
        return

    axes[0].set_title("Expected Calibration Error (lower is better)", fontsize=11)
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["ours", "live Jev"])
    axes[1].legend(fontsize=9)

    fig.suptitle(
        "Measured on 9,730 unseen claims with human ground truth\n"
        "jeff (local, 4B + LoRA) vs TypeSafe Jev 1.13.0",
        fontsize=12,
    )
    fig.tight_layout()
    dest = ROOT / "results" / "calibration_plot.png"
    fig.savefig(dest, dpi=150)
    print("wrote", dest)


if __name__ == "__main__":
    main()
