"""Paired A/B between two prediction files on the scale split.

Compares two arms on the SAME rows and reports the paired McNemar test plus a
bootstrap CI on the difference, because an unpaired accuracy delta is the wrong
instrument: the arms answer the same questions.

Also reports the stratified breakdown that matters for this project -- recall by
gold class, split by passage count -- so a gain can be attributed to the
mechanism it is supposed to fix rather than to noise elsewhere.

Usage:
    uv run python -m scripts.ab_scale_preds \
        data/factcheck/preds_ours_large.jsonl \
        data/factcheck/preds_soft_large.jsonl
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from math import erfc, sqrt
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["supported", "refuted", "not_enough_info"]


def load_gold(path: Path) -> tuple[dict[str, str], dict[str, int]]:
    gold, npas = {}, {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        qid = next(iter(r["questions"]))
        gold[r["row_id"]] = max(r["labels"][qid].items(), key=lambda kv: kv[1])[0]
        ev = (r.get("state") or {}).get("evidence") or []
        npas[r["row_id"]] = len(ev)
    return gold, npas


def load_preds(path: Path) -> dict[str, dict[str, float]]:
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        out[d["row_id"]] = d["probs"]
    return out


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    n = len(conf)
    e = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        m = (conf >= lo) & ((conf < hi) if b < n_bins - 1 else (conf <= hi))
        if m.sum() == 0:
            continue
        e += m.sum() / n * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    a_path, b_path = Path(sys.argv[1]), Path(sys.argv[2])
    gold, npas = load_gold(ROOT / "data/factcheck/eval_large.jsonl")
    A, B = load_preds(a_path), load_preds(b_path)

    ids = [r for r in gold if r in A and r in B and all(l in A[r] and l in B[r] for l in LABELS)]
    if not ids:
        raise SystemExit("no overlapping rows")

    def top(p):
        return max(p.items(), key=lambda kv: kv[1])

    a_ok = np.array([top(A[r])[0] == gold[r] for r in ids])
    b_ok = np.array([top(B[r])[0] == gold[r] for r in ids])
    n = len(ids)

    print(f"A = {a_path.name}   acc={a_ok.mean():.6f} ({a_ok.sum()}/{n})")
    print(f"B = {b_path.name}   acc={b_ok.mean():.6f} ({b_ok.sum()}/{n})")
    print(f"delta (B-A) = {b_ok.mean() - a_ok.mean():+.6f}  = {int(b_ok.sum() - a_ok.sum()):+d} rows")
    print()

    b_ = int((b_ok & ~a_ok).sum())
    c_ = int((~b_ok & a_ok).sum())
    if b_ + c_:
        z = abs(b_ - c_) / sqrt(b_ + c_)
        p = erfc(z / sqrt(2))
    else:
        z, p = 0.0, 1.0
    print(f"paired McNemar: B-right/A-wrong={b_}  A-right/B-wrong={c_}  "
          f"discordant={b_ + c_}")
    print(f"  z={z:.2f}  p={p:.4f}  -> {'SIGNIFICANT' if p < 0.05 else 'not significant'}")
    d = b_ok.astype(int) - a_ok.astype(int)
    rng = np.random.default_rng(11)
    bs = [d[rng.integers(0, n, n)].mean() for _ in range(6000)]
    print(f"  paired diff bootstrap 95% CI [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]")
    print()

    for tag, P in (("A", A), ("B", B)):
        conf = np.array([top(P[r])[1] for r in ids])
        ok = np.array([top(P[r])[0] == gold[r] for r in ids])
        print(f"{tag} ECE (max-prob conf) = {ece(conf, ok):.5f}")
    print()

    print("recall by gold class (all rows):")
    print(f"  {'gold':>16}{'n':>7}{'A':>9}{'B':>9}{'delta':>9}")
    for lab in LABELS:
        s = [r for r in ids if gold[r] == lab]
        if not s:
            continue
        ra = np.mean([top(A[r])[0] == lab for r in s])
        rb = np.mean([top(B[r])[0] == lab for r in s])
        print(f"  {lab:>16}{len(s):>7}{ra:>9.4f}{rb:>9.4f}{rb-ra:>+9.4f}")
    print()

    print("single-passage rows only (the stratum that carries the deficit):")
    one = [r for r in ids if npas[r] == 1]
    if one:
        print(f"  n={len(one)}")
        for lab in LABELS:
            s = [r for r in one if gold[r] == lab]
            if not s:
                continue
            ra = np.mean([top(A[r])[0] == lab for r in s])
            rb = np.mean([top(B[r])[0] == lab for r in s])
            print(f"  {lab:>16}{len(s):>7}{ra:>9.4f}{rb:>9.4f}{rb-ra:>+9.4f}"
                  f"   rows={int((rb-ra)*len(s)):+d}")
    print()

    # Where did B move relative to A, regardless of correctness?
    moved = Counter()
    for r in ids:
        ta, tb = top(A[r])[0], top(B[r])[0]
        if ta != tb:
            moved[(ta, tb)] += 1
    print("label flips A -> B (top-20):")
    for (ta, tb), k in moved.most_common(20):
        fixed = sum(1 for r in ids if top(A[r])[0] == ta and top(B[r])[0] == tb and gold[r] == tb)
        broke = sum(1 for r in ids if top(A[r])[0] == ta and top(B[r])[0] == tb and gold[r] == ta)
        print(f"  {ta:16} -> {tb:16} {k:5}   (fixed {fixed:4}, broke {broke:4}, neither {k - fixed - broke:4})")


if __name__ == "__main__":
    main()
