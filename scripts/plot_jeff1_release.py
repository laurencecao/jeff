"""Simple Jeff 1 vs Jev figures for the public README and model card."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
STATIC = ROOT / "results" / "static"
JEFF = "#2563eb"
JEV = "#94a3b8"
INK = "#0f172a"
MUTED = "#64748b"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "axes.edgecolor": "#e2e8f0",
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
    }
)


def load_pairs():
    gold, jeff, jev = {}, {}, {}
    for line in (ROOT / "data/factcheck/eval_large.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        qid = next(iter(r["questions"]))
        labs = r["labels"][qid]
        gold[(r["row_id"], qid)] = max(labs, key=labs.get)
    for path, dest in (
        (ROOT / "data/factcheck/preds_ours_large.jsonl", jeff),
        (ROOT / "data/factcheck/preds_jev_large.jsonl", jev),
    ):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            dest[(r["row_id"], r["question_id"])] = r
    keys = sorted(gold)
    assert keys == sorted(jeff) == sorted(jev)
    return gold, jeff, jev, keys


def top(probs: dict) -> tuple[str, float]:
    lab, p = max(probs.items(), key=lambda kv: kv[1])
    return lab, float(p)


def ece(conf, ok, n_bins=10) -> float:
    conf, ok = np.asarray(conf), np.asarray(ok)
    n = len(conf)
    e = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        m = (conf >= lo) & ((conf < hi) if b < n_bins - 1 else (conf <= hi))
        if m.any():
            e += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return float(e)


def curve(conf, ok, n_bins=10):
    conf, ok = np.asarray(conf), np.asarray(ok)
    xs, ys = [], []
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        m = (conf >= lo) & ((conf < hi) if b < n_bins - 1 else (conf <= hi))
        if m.sum() >= 20:
            xs.append(float(conf[m].mean()))
            ys.append(float(ok[m].mean()))
    return xs, ys


def save(fig, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    STATIC.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.25)
    fig.savefig(STATIC / name, dpi=160, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", path)
    return path


def plot_headline(jeff_acc, jev_acc, jeff_ece, jev_ece):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    for ax, title, j, v, ylim, fmt, better in (
        (axes[0], "Accuracy", jeff_acc, jev_acc, (0, 1.0), "{:.1%}", "higher"),
        (axes[1], "Calibration error", jeff_ece, jev_ece, (0, 0.12), "{:.3f}", "lower"),
    ):
        ax.bar([0, 1], [j, v], color=[JEFF, JEV], width=0.55, linewidth=0)
        ax.set_xticks([0, 1], ["Jeff 1", "Jev 1.13.0"])
        ax.set_ylim(*ylim)
        ax.set_title(f"{title}  ·  {better} is better", loc="left", fontsize=11, pad=10)
        for x, y in enumerate([j, v]):
            ax.text(x, y, "  " + fmt.format(y), ha="center", va="bottom", fontsize=10, color=INK)
    fig.suptitle("9,730 human-labelled claims  ·  same rows", fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save(fig, "headline.png")


def plot_recall(jeff_rec, jev_rec):
    labels = ["supported", "refuted", "not enough info"]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.barh(y + 0.18, [jeff_rec[k] for k in labels], height=0.32, color=JEFF, label="Jeff 1")
    ax.barh(y - 0.18, [jev_rec[k] for k in labels], height=0.32, color=JEV, label="Jev 1.13.0")
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1.05)
    ax.invert_yaxis()
    ax.set_xlabel("recall")
    ax.set_title("Where the gap is", loc="left", fontsize=12, pad=10)
    ax.legend(frameon=False, loc="lower right")
    for i, k in enumerate(labels):
        ax.text(jeff_rec[k] + 0.015, i + 0.18, f"{jeff_rec[k]:.0%}", va="center", fontsize=9, color=JEFF)
        ax.text(jev_rec[k] + 0.015, i - 0.18, f"{jev_rec[k]:.0%}", va="center", fontsize=9, color=MUTED)
    fig.tight_layout()
    save(fig, "recall.png")


def plot_reliability(jeff_c, jeff_ok, jev_c, jev_ok, jeff_ece, jev_ece):
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#cbd5e1", label="perfect")
    jx, jy = curve(jeff_c, jeff_ok)
    vx, vy = curve(jev_c, jev_ok)
    ax.plot(jx, jy, "o-", color=JEFF, lw=2, ms=6, label=f"Jeff 1  (ECE {jeff_ece:.3f})")
    ax.plot(vx, vy, "o-", color=JEV, lw=2, ms=6, label=f"Jev 1.13.0  (ECE {jev_ece:.3f})")
    ax.set_xlim(0.3, 1.02)
    ax.set_ylim(0.15, 1.02)
    ax.set_xlabel("confidence  (max class probability)")
    ax.set_ylabel("accuracy in that bin")
    ax.set_title("Reliability", loc="left", fontsize=12, pad=10)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "reliability.png")


def main() -> None:
    gold, jeff, jev, keys = load_pairs()
    labels = ("supported", "refuted", "not_enough_info")
    jeff_ok, jev_ok, jeff_c, jev_c = [], [], [], []
    cm_j = {g: {p: 0 for p in labels} for g in labels}
    cm_v = {g: {p: 0 for p in labels} for g in labels}
    for k in keys:
        g = gold[k]
        ja, jc = top(jeff[k]["probs"])
        va, vc = top(jev[k]["probs"])
        jeff_ok.append(ja == g)
        jev_ok.append(va == g)
        jeff_c.append(jc)
        jev_c.append(vc)
        cm_j[g][ja] += 1
        cm_v[g][va] += 1
    n = len(keys)
    jeff_acc, jev_acc = sum(jeff_ok) / n, sum(jev_ok) / n
    jeff_ece, jev_ece = ece(jeff_c, jeff_ok), ece(jev_c, jev_ok)
    rec = lambda cm, lab: cm[lab][lab] / sum(cm[lab].values())
    jeff_rec = {
        "supported": rec(cm_j, "supported"),
        "refuted": rec(cm_j, "refuted"),
        "not enough info": rec(cm_j, "not_enough_info"),
    }
    jev_rec = {
        "supported": rec(cm_v, "supported"),
        "refuted": rec(cm_v, "refuted"),
        "not enough info": rec(cm_v, "not_enough_info"),
    }
    print(f"n={n} jeff_acc={jeff_acc:.6f} jev_acc={jev_acc:.6f} jeff_ece={jeff_ece:.6f} jev_ece={jev_ece:.6f}")
    plot_headline(jeff_acc, jev_acc, jeff_ece, jev_ece)
    plot_recall(jeff_rec, jev_rec)
    plot_reliability(jeff_c, jeff_ok, jev_c, jev_ok, jeff_ece, jev_ece)


if __name__ == "__main__":
    main()
