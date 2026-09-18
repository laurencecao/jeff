"""Calibration, computed ONE way for both models.

A calibration comparison is only meaningful if both models' confidence is
measured the SAME way. Ours sets confidence = max(prob). Live Jev reports its
own `confidence` statistic, which is NOT max(prob) -- measured, it differs by
up to 0.33 (mean 0.04). Comparing our max-prob ECE against Jev's
own-statistic ECE is not apples-to-apples, and it made the reliability plot
internally inconsistent (bar from one definition, curve from the other).

This script reports both definitions for both models so the choice is visible.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

N_BINS = 10


def load_preds(path: Path) -> dict[tuple[str, str], dict]:
    out = {}
    for line in path.read_text().splitlines():
        d = json.loads(line)
        out[(d["row_id"], d["question_id"])] = d
    return out


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def ece(conf: np.ndarray, corr: np.ndarray, n_bins: int = N_BINS) -> float:
    """Expected calibration error over equal-width bins, population-weighted."""
    n = len(conf)
    e = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        m = (conf >= lo) & ((conf < hi) if b < n_bins - 1 else (conf <= hi))
        if m.sum() == 0:
            continue
        e += m.sum() / n * abs(corr[m].mean() - conf[m].mean())
    return float(e)


def collect(preds: dict, rows: list[dict], use_stored_conf: bool):
    conf, corr = [], []
    for r in rows:
        qid = next(iter(r["questions"]))
        gold = max(r["labels"][qid].items(), key=lambda kv: kv[1])[0]
        p = preds.get((r["row_id"], qid))
        if p is None:
            continue
        top = max(p["probs"].items(), key=lambda kv: kv[1])
        c = p["confidence"] if (use_stored_conf and "confidence" in p) else top[1]
        conf.append(float(c))
        corr.append(1.0 if top[0] == gold else 0.0)
    return np.array(conf), np.array(corr)


def report(tag: str, preds_path: Path, rows_path: Path) -> None:
    preds = load_preds(preds_path)
    rows = load_rows(rows_path)
    # Report the number of rows actually SCORED, not the size of the gold file:
    # ground_truth.jsonl holds 1,987 rows but only 199 of them have val preds.
    n_scored = sum(
        1 for r in rows if (r["row_id"], next(iter(r["questions"]))) in preds
    )
    print(f"\n=== {tag}  (scored rows={n_scored} of {len(rows)} in the gold file) ===")
    for use_stored in (False, True):
        conf, corr = collect(preds, rows, use_stored)
        acc = corr.mean()
        d = np.abs(conf - np.array([max(preds[(r["row_id"], next(iter(r["questions"])))]["probs"].values())
                                    for r in rows
                                    if (r["row_id"], next(iter(r["questions"]))) in preds]))
        label = "stored confidence" if use_stored else "max(prob)"
        print(f"  {label:18} ECE={ece(conf, corr):.5f}  acc={acc:.5f}  "
              f"conf range [{conf.min():.4f}, {conf.max():.4f}]")
        if use_stored:
            print(f"  {'':18} vs max(prob): max diff={d.max():.4f} mean={d.mean():.4f}")


def main() -> None:
    scale_rows = ROOT / "data/factcheck/eval_large.jsonl"
    val_rows = ROOT / "data/factcheck/ground_truth.jsonl"

    report("SCALE: ours (lora_4b_multi)", ROOT / "data/factcheck/preds_ours_large.jsonl", scale_rows)
    report("SCALE: live Jev", ROOT / "data/factcheck/preds_jev_large.jsonl", scale_rows)

    # val: ours = the run just measured; jev = its stored val preds
    report("VAL n=199: ours (lora_4b_multi)", ROOT / "data/factcheck/preds_autoresearch_val.jsonl", val_rows)
    report("VAL n=199: live Jev", ROOT / "data/factcheck/preds_jev_val.jsonl", val_rows)


if __name__ == "__main__":
    main()
