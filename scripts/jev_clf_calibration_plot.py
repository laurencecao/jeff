"""Reliability diagrams for our model vs live Jev, on human labels.

This is the plot TypeSafe never published: measured calibration on a dataset
with human ground truth, far from their four launch workflows.

    uv run python -m scripts.jev_clf_calibration_plot
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

from jev_clf import schema as S  # noqa: E402
from jev_clf.eval import reliability_bins  # noqa: E402

FILES = [
    ("ours (4B + LoRA), n=9730", ROOT / "results/lm_eval_4b_multi_large.json"),
    ("live Jev 1.13.0, n=9730", ROOT / "results/jev_large.json"),
]


def series_from_preds(preds_path: Path, rows_path: Path, conf_field: str = "max_prob"):
    """(confidence, correct) arrays for one model.

    `conf_field` selects HOW confidence is measured, and it MUST be the same
    for both models or the comparison is meaningless:

      max_prob   confidence = max probability over labels. This is what our
                 client reports, so it is the like-for-like measure.
      stored     the prediction's own ``confidence`` field. Ours equals
                 max(prob) exactly (verified, max diff 0.0000); live Jev
                 reports its own statistic, which is NOT max(prob) -- measured
                 it differs by up to 0.33, mean 0.04. Using this for Jev while
                 the curve uses max(prob) is what made an earlier version of
                 this figure contradict its own legend.
    """
    gt = {}
    for r in S.read_rows(rows_path):
        gt[r.row_id] = r
    preds = S.read_predictions(preds_path)
    conf, correct = [], []
    for pr in preds:
        row = gt.get(pr.row_id)
        if row is None:
            continue
        row_q = row.questions.get(pr.question_id)
        if row_q is None:
            continue
        target = S.argmax_label(row.targets(pr.question_id))
        top = max(pr.probs.items(), key=lambda kv: kv[1])
        if conf_field == "stored" and pr.confidence is not None:
            c = float(pr.confidence)
        else:
            c = float(top[1])
        conf.append(c)
        correct.append(1.0 if top[0] == target else 0.0)
    return np.array(conf), np.array(correct)


def ece_of(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """Population-weighted ECE over equal-width bins. One definition, reused
    for the bar and the curve so the two can never disagree."""
    n = len(conf)
    e = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        m = (conf >= lo) & ((conf < hi) if b < n_bins - 1 else (conf <= hi))
        if m.sum() == 0:
            continue
        e += m.sum() / n * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def curve_of(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10):
    """Reliability points: (mean confidence, empirical accuracy) per bin."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    pts_c, pts_a, pts_n = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if m.sum() > 0:
            pts_c.append(float(conf[m].mean()))
            pts_a.append(float(correct[m].mean()))
            pts_n.append(int(m.sum()))
    return pts_c, pts_a, pts_n


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))

    axes[0].set_ylabel("ECE")
    axes[0].set_xlabel("")          # bars are models, not a confidence axis
    axes[0].grid(alpha=0.3, axis="y")
    axes[1].set_xlabel("stated confidence")
    axes[1].grid(alpha=0.3)
    axes[1].plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    axes[1].set_ylabel("empirical accuracy")

    rows_path = ROOT / "data/factcheck/eval_large.jsonl"
    models = [
        ("ours", "ours (4B + LoRA)", ROOT / "data/factcheck/preds_ours_large.jsonl", "#1f77b4"),
        ("jev", "live Jev 1.13.0", ROOT / "data/factcheck/preds_jev_large.jsonl", "#d62728"),
    ]

    # ONE definition for both: confidence = max class probability.
    computed = {}
    for key, label, path, color in models:
        if not path.exists():
            print(f"({key}: {path.name} not found, skipping)")
            continue
        conf, correct = series_from_preds(path, rows_path, conf_field="max_prob")
        computed[key] = (label, conf, correct, color)

    for i, (key, (label, conf, correct, color)) in enumerate(computed.items()):
        e = ece_of(conf, correct)
        axes[0].bar([i], [e], color=color, width=0.45)
        axes[0].text(i, e + 0.004, f"{e:.4f}", ha="center", fontsize=10)
        pc, pa, pn = curve_of(conf, correct)
        axes[1].plot(pc, pa, "o-", color=color,
                     label=f"{label}, n={len(conf)}  (ECE {e:.4f})")

    # Jev's OWN reported statistic, for transparency -- not used in the bars,
    # so the figure cannot mix definitions again.
    jev_preds = ROOT / "data/factcheck/preds_jev_large.jsonl"
    if jev_preds.exists():
        jc, jok = series_from_preds(jev_preds, rows_path, conf_field="stored")
        j_ece_own = ece_of(jc, jok)
        print(f"[plot] Jev self-reported confidence ECE = {j_ece_own:.4f} "
              f"(NOT used for the bar; different definition)")

    axes[0].set_title("Expected Calibration Error (lower is better)", fontsize=11)
    axes[0].set_xticks(list(range(len(computed))))
    axes[0].set_xticklabels([v[0] for v in computed.values()], fontsize=9)
    axes[1].legend(fontsize=8.5, loc="upper left")

    fig.suptitle(
        "Measured on 9,730 unseen claims with human ground truth\n"
        "confidence = max class probability, the same definition for both models",
        fontsize=12,
    )
    fig.tight_layout()
    dest = ROOT / "results" / "calibration_plot.png"
    fig.savefig(dest, dpi=150)
    print("wrote", dest)

    # keep the page's copy in sync
    static_dir = ROOT / "results" / "static"
    if static_dir.is_dir():
        import shutil
        shutil.copyfile(dest, static_dir / "calibration_plot.png")
        print("synced", static_dir / "calibration_plot.png")


if __name__ == "__main__":
    main()
