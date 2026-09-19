"""Fail-closed audit of decision-model prediction files against gold labels.

Recomputes every reported number from the raw JSONL artifacts rather than
trusting a summary: accuracy, macro F1, NLL, multiclass Brier, four ECE
variants, coverage-targeted selective risk, per-source and per-passage-count
recall, an exact paired McNemar test, and a fixed-seed paired bootstrap CI.

Fail-closed contract: any malformed input (duplicate keys, missing/extra keys,
bad label set, non-finite/negative/unnormalized probabilities, non-one-hot
gold) aborts with a nonzero exit and writes NOTHING. Outputs are computed fully
in memory and then written via tmp-file + rename, so a failed audit can never
leave a partial or stale-looking result behind.

Rows are keyed by (row_id, question_id). The audited question defaults to
"verdict"; a prediction row whose question_id is not the audited one is an
unexpected key and fails the audit.

Usage:
    uv run python -m scripts.audit_decision_results \
        --gold data/factcheck/eval_large.jsonl \
        --preds data/factcheck/preds_ours_large.jsonl \
                data/factcheck/preds_jev_large.jsonl \
        --json-out results/researchmax_gap_audit.json \
        --md-out results/researchmax_gap_audit.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_GOLD = "data/factcheck/eval_large.jsonl"
DEFAULT_PREDS = [
    "data/factcheck/preds_ours_large.jsonl",
    "data/factcheck/preds_jev_large.jsonl",
]
DEFAULT_QUESTION = "verdict"
DEFAULT_JSON_OUT = "results/researchmax_gap_audit.json"
DEFAULT_MD_OUT = "results/researchmax_gap_audit.md"
# Auto-attached as a historical (verbatim, unverified) summary when present.
DEFAULT_HISTORICAL = "results/jev_large.json"

NLL_EPSILON = 1e-15
PROB_SUM_TOL = 1e-6
CONF_TOL = 1e-9
N_BINS = 10
COVERAGE_TARGETS = (0.5, 0.8, 0.9)
BOOTSTRAP_SEED = 11
BOOTSTRAP_RESAMPLES = 6000


class AuditError(Exception):
    """Any input or computation that cannot be audited exactly."""


# --------------------------------------------------------------------------
# Loading + validation (fail closed)
# --------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_jsonl(path: Path):
    if not path.is_file():
        raise AuditError(f"missing file: {path}")
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            yield lineno, json.loads(line)
        except json.JSONDecodeError as e:
            raise AuditError(f"{path}:{lineno}: invalid JSON: {e}") from e


def _check_label_set(dist: dict, labels: list[str], where: str) -> None:
    if not isinstance(dist, dict):
        raise AuditError(f"{where}: probability/label map is not an object")
    if set(dist) != set(labels):
        raise AuditError(
            f"{where}: label set {sorted(dist)} != expected {sorted(labels)}")


def _check_prob_vector(dist: dict, labels: list[str], where: str) -> np.ndarray:
    _check_label_set(dist, labels, where)
    p = np.array([dist[l] for l in labels], dtype=float)
    if not np.all(np.isfinite(p)):
        raise AuditError(f"{where}: non-finite probability in {dist}")
    if np.any(p < 0):
        raise AuditError(f"{where}: negative probability in {dist}")
    s = float(p.sum())
    if abs(s - 1.0) > PROB_SUM_TOL:
        raise AuditError(
            f"{where}: probabilities sum to {s!r}, not 1 "
            f"(tolerance {PROB_SUM_TOL})")
    return p


def load_gold(path: Path, question: str) -> tuple[dict, list[str]]:
    """Return ({(row_id, qid): row_info}, labels). Fail closed.

    row_info: {label, probs(np one-hot), source, n_passages}.
    """
    gold: dict = {}
    labels: list[str] | None = None
    for lineno, r in _iter_jsonl(path):
        if not isinstance(r, dict) or "row_id" not in r:
            raise AuditError(f"{path}:{lineno}: gold row missing row_id")
        row_id = r["row_id"]
        qs = r.get("questions")
        ls = r.get("labels")
        if not isinstance(qs, dict) or not isinstance(ls, dict):
            raise AuditError(f"{path}:{lineno}: gold row missing questions/labels")
        if set(qs) != set(ls):
            raise AuditError(
                f"{path}:{lineno}: questions keys {sorted(qs)} != "
                f"labels keys {sorted(ls)}")
        if labels is None:
            if question not in ls:
                raise AuditError(
                    f"{path}:{lineno}: audited question {question!r} absent "
                    f"from first gold row (has {sorted(ls)})")
            labels = sorted(ls[question])
        for qid, dist in ls.items():
            if qid not in qs:
                raise AuditError(f"{path}:{lineno}: label qid {qid!r} not in questions")
            where = f"{path}:{lineno} ({row_id},{qid})"
            p = _check_prob_vector(dist, labels, where)
            if float(p.max()) != 1.0 or int((p == 1.0).sum()) != 1:
                raise AuditError(f"{where}: gold distribution is not one-hot: {dist}")
            key = (row_id, qid)
            if key in gold:
                raise AuditError(f"{where}: duplicate gold key {key}")
            ev = (r.get("state") or {}).get("evidence") or []
            gold[key] = {
                "label": labels[int(p.argmax())],
                "probs": p,
                "source": r.get("source") or "unknown",
                "n_passages": len(ev),
            }
    if labels is None or not gold:
        raise AuditError(f"{path}: no gold rows")
    expected = {k for k in gold if k[1] == question}
    if not expected:
        raise AuditError(f"{path}: no gold rows for question {question!r}")
    return gold, labels


def load_preds(path: Path, labels: list[str]) -> dict:
    """Return {(row_id, qid): {probs(np), confidence, argmax}}. Fail closed."""
    preds: dict = {}
    for lineno, d in _iter_jsonl(path):
        if not isinstance(d, dict) or "row_id" not in d or "question_id" not in d:
            raise AuditError(f"{path}:{lineno}: pred row missing row_id/question_id")
        key = (d["row_id"], d["question_id"])
        where = f"{path}:{lineno} {key}"
        if key in preds:
            raise AuditError(f"{where}: duplicate prediction key")
        p = _check_prob_vector(d.get("probs"), labels, where)
        conf = d.get("confidence")
        if not isinstance(conf, (int, float)) or isinstance(conf, bool) \
                or not math.isfinite(conf):
            raise AuditError(f"{where}: missing/non-finite confidence {conf!r}")
        if conf < -CONF_TOL or conf > 1.0 + CONF_TOL:
            raise AuditError(f"{where}: confidence {conf!r} outside [0,1]")
        top_label, top_prob = max(d["probs"].items(), key=lambda kv: kv[1])
        preds[key] = {
            "probs": p,
            "confidence": min(max(float(conf), 0.0), 1.0),
            # insertion-order top-1: identical policy to the existing eval
            # (max(probs.items(), key=kv[1]) -> first key wins exact ties)
            "top_label": top_label,
            "top_prob": float(top_prob),
            "n_tied_at_top": sum(1 for v in d["probs"].values()
                                 if v == top_prob),
        }
    if not preds:
        raise AuditError(f"{path}: no prediction rows")
    return preds


def align(gold: dict, preds: dict, question: str, pred_name: str) -> list:
    """Exact key coverage: pred keys for `question` must equal gold keys."""
    expected = {k for k in gold if k[1] == question}
    got = set(preds)
    other_q = {k for k in got if k[1] != question}
    if other_q:
        raise AuditError(
            f"{pred_name}: {len(other_q)} prediction rows carry a question_id "
            f"other than {question!r} (e.g. {min(other_q)})")
    missing = sorted(expected - got)
    extra = sorted(got - expected)
    if missing or extra:
        raise AuditError(
            f"{pred_name}: key coverage mismatch vs gold for question "
            f"{question!r}: {len(missing)} missing, {len(extra)} extra "
            f"(first missing={missing[:3]}, first extra={extra[:3]})")
    return sorted(expected)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def argmax_label(p: np.ndarray, labels: list[str]) -> tuple[str, float]:
    """Sorted-order argmax (alternate tie-break sensitivity only).

    The audited top-1 uses probs insertion order, matching the existing
    eval; this helper exists so the sensitivity number and tests share one
    implementation of the alternative."""
    i = int(np.argmax(p))
    return labels[i], float(p[i])


def ece_uniform(conf: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS) -> float:
    """10 equal-width bins on [0,1]; last bin closed on the right."""
    n = len(conf)
    e = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        m = (conf >= lo) & ((conf < hi) if b < n_bins - 1 else (conf <= hi))
        if m.sum() == 0:
            continue
        e += m.sum() / n * abs(float(correct[m].mean()) - float(conf[m].mean()))
    return float(e)


def ece_adaptive(conf: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS) -> float:
    """Equal-count bins over ascending stable rank.

    Tie handling: np.argsort(kind='stable') keeps input order inside a tied
    confidence value, and np.array_split cuts contiguous rank ranges (earlier
    bins get the remainder), so a tied value MAY straddle a boundary. No
    attempt is made to keep ties together; that is the stated convention.
    """
    n = len(conf)
    order = np.argsort(conf, kind="stable")
    e = 0.0
    for b in np.array_split(order, n_bins):
        if len(b) == 0:
            continue
        e += len(b) / n * abs(float(correct[b].mean()) - float(conf[b].mean()))
    return float(e)


def selective_risk(conf: np.ndarray, correct: np.ndarray,
                   targets=COVERAGE_TARGETS) -> dict:
    """Coverage-targeted selective risk with conservative tie groups.

    For target c: k=ceil(c*n), tau = kth-largest confidence, then EVERY row
    with conf >= tau is kept -- all rows tied at the threshold are included,
    so realized coverage can exceed c but never falls below it.
    """
    n = len(conf)
    order = np.argsort(-conf, kind="stable")
    out = {}
    for c in targets:
        k = int(math.ceil(c * n))
        tau = float(conf[order[k - 1]])
        sel = conf >= tau
        ns = int(sel.sum())
        out[f"{c:g}"] = {
            "target_coverage": c,
            "confidence_threshold": tau,
            "n_selected": ns,
            "coverage": ns / n,
            "accuracy": float(correct[sel].mean()),
            "risk": float(1.0 - correct[sel].mean()),
        }
    return out


def _per_label_prf(gold_lab: np.ndarray, pred_lab: np.ndarray, labels: list[str]):
    per, f1s = {}, []
    for l in labels:
        tp = int(((pred_lab == l) & (gold_lab == l)).sum())
        fp = int(((pred_lab == l) & (gold_lab != l)).sum())
        fn = int(((pred_lab != l) & (gold_lab == l)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[l] = {"precision": prec, "recall": rec, "f1": f1,
                  "support": int((gold_lab == l).sum())}
        f1s.append(f1)
    return per, float(np.mean(f1s))


def _recall_by(strata: dict, gold_lab: np.ndarray, pred_lab: np.ndarray,
               labels: list[str]) -> dict:
    out = {}
    for name, idx in sorted(strata.items(), key=lambda kv: str(kv[0])):
        idx = np.asarray(idx)
        g, p = gold_lab[idx], pred_lab[idx]
        rec = {}
        for l in labels:
            m = g == l
            if m.sum():
                rec[l] = {"n": int(m.sum()),
                          "recall": float((p[m] == l).mean())}
        out[str(name)] = {
            "n": int(len(idx)),
            "accuracy": float((g == p).mean()),
            "recall_by_class": rec,
        }
    return out


def arm_metrics(keys: list, gold: dict, preds: dict, labels: list[str]) -> dict:
    n = len(keys)
    P = np.stack([preds[k]["probs"] for k in keys])
    conf_stored = np.array([preds[k]["confidence"] for k in keys])
    gold_lab = np.array([gold[k]["label"] for k in keys])
    gold_idx = np.array([labels.index(g) for g in gold_lab])
    pred_lab = np.array([preds[k]["top_label"] for k in keys])
    pred_idx = np.array([labels.index(l) for l in pred_lab])
    correct = pred_idx == gold_idx
    conf_top = np.array([preds[k]["top_prob"] for k in keys])
    n_ties = int(sum(preds[k]["n_tied_at_top"] > 1 for k in keys))
    # sensitivity: what accuracy would be under sorted-label tie-break
    alt_idx = P.argmax(axis=1)
    acc_alt = float((alt_idx == gold_idx).mean())

    Y = np.zeros_like(P)
    Y[np.arange(n), gold_idx] = 1.0

    per_label, macro_f1 = _per_label_prf(gold_lab, pred_lab, labels)

    confusion = {g: {p: 0 for p in labels} for g in labels}
    for g, p in zip(gold_lab, pred_lab):
        confusion[g][p] += 1

    ovr_ece = {}
    for ci, l in enumerate(labels):
        ovr_ece[l] = ece_uniform(P[:, ci], (gold_idx == ci).astype(float))

    by_source, by_npas = {}, {}
    for i, k in enumerate(keys):
        by_source.setdefault(gold[k]["source"], []).append(i)
        by_npas.setdefault(gold[k]["n_passages"], []).append(i)

    return {
        "n": n,
        "n_correct": int(correct.sum()),
        "accuracy": float(correct.mean()),
        "n_exact_argmax_ties": n_ties,
        "accuracy_alt_sorted_tiebreak": acc_alt,
        "tiebreak_policy": "insertion order of the probs object "
                           "(max(probs.items(), key=kv[1])), matching the "
                           "existing eval; sorted-label argmax reported "
                           "separately as accuracy_alt_sorted_tiebreak",
        "macro_f1": macro_f1,
        "per_label": per_label,
        "nll": float((-np.log(np.maximum(
            P[np.arange(n), gold_idx], NLL_EPSILON))).mean()),
        "nll_epsilon": NLL_EPSILON,
        "brier_multiclass": float(((P - Y) ** 2).sum(axis=1).mean()),
        "ece": {
            "top_label_10bin": ece_uniform(conf_top, correct.astype(float)),
            "stored_confidence_10bin": ece_uniform(
                conf_stored, correct.astype(float)),
            "adaptive_equal_count_10bin": ece_adaptive(
                conf_top, correct.astype(float)),
            "classwise_ovr_10bin": ovr_ece,
            "classwise_ovr_macro": float(np.mean(list(ovr_ece.values()))),
        },
        "selective_risk": selective_risk(conf_stored, correct),
        "confusion": confusion,
        "by_source": _recall_by(by_source, gold_lab, pred_lab, labels),
        "by_passage_count": _recall_by(by_npas, gold_lab, pred_lab, labels),
        "_correct": correct,  # internal; stripped before output
    }


def paired_compare(keys: list, gold: dict, labels: list[str],
                   name_a: str, A: dict, name_b: str, B: dict) -> dict:
    """Exact McNemar + fixed-seed paired bootstrap on the same rows."""
    a_ok = np.array([A[k]["top_label"] == gold[k]["label"] for k in keys])
    b_ok = np.array([B[k]["top_label"] == gold[k]["label"] for k in keys])
    n = len(keys)
    b_only = int((b_ok & ~a_ok).sum())  # B right, A wrong
    a_only = int((~b_ok & a_ok).sum())  # A right, B wrong
    disc = b_only + a_only
    p_exact = float(binomtest(min(b_only, a_only), disc, 0.5).pvalue) \
        if disc else 1.0

    d = b_ok.astype(int) - a_ok.astype(int)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bs = np.array([d[rng.integers(0, n, n)].mean()
                   for _ in range(BOOTSTRAP_RESAMPLES)])
    return {
        "a": name_a, "b": name_b, "n": n,
        "accuracy_a": float(a_ok.mean()), "accuracy_b": float(b_ok.mean()),
        "delta_b_minus_a": float(b_ok.mean() - a_ok.mean()),
        "mcnemar": {
            "b_only_correct": b_only, "a_only_correct": a_only,
            "discordant": disc, "p_exact_two_sided": p_exact,
            "test": "scipy.stats.binomtest(min(b,c), b+c, 0.5), two-sided",
        },
        "bootstrap": {
            "resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED,
            "statistic": "mean(b_ok - a_ok) over paired row resamples",
            "ci95": [float(np.percentile(bs, 2.5)),
                     float(np.percentile(bs, 97.5))],
        },
    }


# --------------------------------------------------------------------------
# Payload + rendering
# --------------------------------------------------------------------------

FORMULAS = {
    "key": "(row_id, question_id); pred keys for the audited question must "
           "equal gold keys exactly; duplicates in either file abort",
    "labels": "sorted union taken from the first gold row's labels[question]; "
              "every gold/pred probability map must match it exactly",
    "gold": "labels[qid] must be finite, nonnegative, sum to 1 within "
            f"{PROB_SUM_TOL}, and be exactly one-hot",
    "probs": "finite, nonnegative, sum to 1 within "
             f"{PROB_SUM_TOL}; top-1 = max(probs.items(), key=kv[1]) so "
             "exact ties resolve to the FIRST INSERTED key (existing eval "
             "policy); n_exact_argmax_ties and accuracy_alt_sorted_tiebreak "
             "report tie prevalence and the sorted-order counterfactual",
    "confidence": "stored confidence required, finite, within "
                  f"[0,1] +/- {CONF_TOL}, clipped for binning",
    "accuracy": "mean(argmax(probs) == gold_label)",
    "macro_f1": "mean over classes of 2PR/(P+R); empty class contributes 0",
    "nll": f"mean(-log(max(p_gold, {NLL_EPSILON})))",
    "brier_multiclass": "mean over rows of sum_c (p_c - y_c)^2",
    "ece_top_label_10bin": "10 equal-width bins on [0,1] over max(probs); "
                           "sum_b (n_b/N) * |acc_b - conf_b|",
    "ece_stored_confidence_10bin": "same binning over the stored 'confidence' "
                                   "field, reported separately",
    "ece_classwise_ovr_10bin": "per class c, 10-bin ECE of p_c vs "
                               "1[gold==c]; macro = mean over classes",
    "ece_adaptive_equal_count_10bin": "10 equal-count bins over ascending "
                                    "stable rank of max(probs); ties keep "
                                    "input order and may straddle a boundary",
    "selective_risk": "for target c: k=ceil(c*N), tau=kth-largest stored "
                      "confidence, keep ALL rows with conf>=tau (tie group "
                      "fully included, so realized coverage >= c); "
                      "risk = 1 - accuracy on kept rows",
    "mcnemar": "exact two-sided binomtest on discordant pairs "
               "(b_only_correct, a_only_correct) on identical rows",
    "bootstrap": f"{BOOTSTRAP_RESAMPLES} paired row resamples, seed "
                 f"{BOOTSTRAP_SEED}, percentile 95% CI on acc(B)-acc(A)",
}


def build_payload(gold_path: Path, pred_paths: dict, question: str,
                  historical: dict | None) -> dict:
    gold, labels = load_gold(gold_path, question)
    arms, keys = {}, None
    for name, p in pred_paths.items():
        preds = load_preds(p, labels)
        ks = align(gold, preds, question, name)
        if keys is None:
            keys = ks
        arms[name] = {"metrics": arm_metrics(ks, gold, preds, labels),
                      "_preds": preds}

    names = list(pred_paths)
    paired = None
    if len(names) >= 2:
        paired = paired_compare(keys, gold, labels, names[0],
                                arms[names[0]]["_preds"], names[1],
                                arms[names[1]]["_preds"])

    payload = {
        "audit": "researchmax_gap_audit",
        "generated_at": datetime.now(UTC).isoformat(),
        "question_id": question,
        "labels": labels,
        "inputs": {
            "gold": {"path": str(gold_path), "sha256": sha256_file(gold_path),
                     "rows": len(gold)},
            "predictions": {name: {"path": str(p), "sha256": sha256_file(p),
                                   "rows": sum(1 for _ in _iter_jsonl(p))}
                            for name, p in pred_paths.items()},
        },
        "versions": {"python": platform.python_version(),
                     "numpy": np.__version__, "scipy": scipy.__version__},
        "counts": {"n_rows": len(keys),
                   "gold_label_counts": dict(Counter(
                       gold[k]["label"] for k in keys))},
        "arms": {name: {k: v for k, v in a["metrics"].items() if k != "_correct"}
                 for name, a in arms.items()},
        "paired": paired,
        "formulas": FORMULAS,
    }
    if historical:
        payload["historical_summaries"] = {
            "note": "verbatim copies of previously published summaries; NOT "
                    "recomputed here and kept separate from the audited "
                    "numbers above",
            "files": historical,
        }
    return payload


def render_markdown(payload: dict) -> str:
    L = []
    L.append("# researchMax gap audit — decision results\n")
    L.append(f"Generated: `{payload['generated_at']}`  ")
    L.append(f"question: `{payload['question_id']}`  ")
    L.append(f"n = {payload['counts']['n_rows']} rows  ")
    L.append(f"labels: {', '.join(payload['labels'])}\n")

    L.append("## Inputs\n")
    L.append("| file | sha256 | rows |")
    L.append("|---|---|---|")
    g = payload["inputs"]["gold"]
    L.append(f"| gold: `{g['path']}` | `{g['sha256'][:16]}…` | {g['rows']} |")
    for name, p in payload["inputs"]["predictions"].items():
        L.append(f"| {name}: `{p['path']}` | `{p['sha256'][:16]}…` | {p['rows']} |")
    v = payload["versions"]
    L.append(f"\npython {v['python']} · numpy {v['numpy']} · scipy {v['scipy']}\n")

    arms = payload["arms"]
    L.append("## Headline metrics (recomputed)\n")
    L.append("| arm | n | accuracy | macro F1 | NLL | Brier |")
    L.append("|---|---|---|---|---|---|")
    for name, m in arms.items():
        L.append(f"| {name} | {m['n']} | {m['accuracy']:.6f} "
                 f"({m['n_correct']}/{m['n']}) | {m['macro_f1']:.6f} | "
                 f"{m['nll']:.6f} | {m['brier_multiclass']:.6f} |")
    for name, m in arms.items():
        if m["n_exact_argmax_ties"]:
            L.append(f"\n_{name}: {m['n_exact_argmax_ties']} exact argmax "
                     f"tie(s); accuracy under sorted-label tie-break would "
                     f"be {m['accuracy_alt_sorted_tiebreak']:.6f} "
                     f"(insertion-order policy used above)._")

    L.append("\n## Calibration (ECE, lower is better)\n")
    L.append("| arm | top-label 10-bin | stored-conf 10-bin | "
             "adaptive equal-count | classwise OvR macro |")
    L.append("|---|---|---|---|---|")
    for name, m in arms.items():
        e = m["ece"]
        L.append(f"| {name} | {e['top_label_10bin']:.5f} | "
                 f"{e['stored_confidence_10bin']:.5f} | "
                 f"{e['adaptive_equal_count_10bin']:.5f} | "
                 f"{e['classwise_ovr_macro']:.5f} |")
    L.append("\nClasswise one-vs-rest ECE:\n")
    L.append("| arm | " + " | ".join(payload["labels"]) + " |")
    L.append("|---|" + "---|" * len(payload["labels"]))
    for name, m in arms.items():
        ovr = m["ece"]["classwise_ovr_10bin"]
        L.append(f"| {name} | " + " | ".join(
            f"{ovr[l]:.5f}" for l in payload["labels"]) + " |")

    L.append("\n## Selective risk (stored confidence, ties fully included)\n")
    L.append("| arm | target coverage | threshold | realized coverage | "
             "n | risk |")
    L.append("|---|---|---|---|---|---|")
    for name, m in arms.items():
        for c, s in m["selective_risk"].items():
            L.append(f"| {name} | {c} | {s['confidence_threshold']:.4f} | "
                     f"{s['coverage']:.4f} | {s['n_selected']} | "
                     f"{s['risk']:.4f} |")

    L.append("\n## Recall by gold class\n")
    L.append("| arm | class | n | recall |")
    L.append("|---|---|---|---|")
    for name, m in arms.items():
        for l in payload["labels"]:
            pl = m["per_label"][l]
            L.append(f"| {name} | {l} | {pl['support']} | {pl['recall']:.4f} |")

    L.append("\n## Accuracy by source\n")
    L.append("| arm | source | n | accuracy |")
    L.append("|---|---|---|---|")
    for name, m in arms.items():
        for src, s in m["by_source"].items():
            L.append(f"| {name} | {src} | {s['n']} | {s['accuracy']:.4f} |")

    L.append("\n## Accuracy by passage count\n")
    L.append("| arm | passages | n | accuracy |")
    L.append("|---|---|---|---|")
    for name, m in arms.items():
        for np_, s in m["by_passage_count"].items():
            L.append(f"| {name} | {np_} | {s['n']} | {s['accuracy']:.4f} |")

    if payload.get("paired"):
        pr = payload["paired"]
        mc, bs = pr["mcnemar"], pr["bootstrap"]
        L.append("\n## Paired comparison (same rows)\n")
        L.append(f"- A = `{pr['a']}` acc {pr['accuracy_a']:.6f}; "
                 f"B = `{pr['b']}` acc {pr['accuracy_b']:.6f}; "
                 f"Δ(B−A) = {pr['delta_b_minus_a']:+.6f}")
        L.append(f"- McNemar exact: B-only {mc['b_only_correct']}, "
                 f"A-only {mc['a_only_correct']}, discordant "
                 f"{mc['discordant']}, p = {mc['p_exact_two_sided']:.4g}")
        L.append(f"- Paired bootstrap 95% CI on Δ: "
                 f"[{bs['ci95'][0]:+.4f}, {bs['ci95'][1]:+.4f}] "
                 f"({bs['resamples']} resamples, seed {bs['seed']})")

    L.append("\n## Confusion (gold × predicted)\n")
    for name, m in arms.items():
        L.append(f"### {name}\n")
        L.append("| gold \\ pred | " + " | ".join(payload["labels"]) + " |")
        L.append("|---|" + "---|" * len(payload["labels"]))
        for gl in payload["labels"]:
            row = m["confusion"][gl]
            L.append(f"| {gl} | " + " | ".join(
                str(row[pl]) for pl in payload["labels"]) + " |")
        L.append("")

    if payload.get("historical_summaries"):
        hs = payload["historical_summaries"]
        L.append("\n## Historical published summaries (verbatim, NOT recomputed)\n")
        L.append(hs["note"] + "\n")
        for path, doc in hs["files"].items():
            L.append(f"### `{path}`\n")
            L.append("```json")
            L.append(json.dumps(doc, indent=2, sort_keys=True))
            L.append("```\n")

    L.append("\n## Formula definitions\n")
    for k, f in payload["formulas"].items():
        L.append(f"- **{k}**: {f}")
    L.append("")
    return "\n".join(L)


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".",
                               suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else ROOT / q


def _arm_name(path: Path) -> str:
    stem = path.stem
    return stem.removeprefix("preds_")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gold", default=DEFAULT_GOLD)
    ap.add_argument("--preds", nargs="+", default=DEFAULT_PREDS)
    ap.add_argument("--question", default=DEFAULT_QUESTION)
    ap.add_argument("--json-out", default=DEFAULT_JSON_OUT)
    ap.add_argument("--md-out", default=DEFAULT_MD_OUT)
    ap.add_argument("--historical", nargs="*", default=None,
                    help="previously published summary JSONs to embed "
                         "verbatim; default: auto-attach "
                         f"{DEFAULT_HISTORICAL} if it exists")
    args = ap.parse_args(argv)

    try:
        gold_path = _resolve(args.gold)
        pred_paths = {}
        for p in args.preds:
            pp = _resolve(p)
            name = _arm_name(pp)
            if name in pred_paths:
                raise AuditError(f"duplicate arm name {name!r}")
            pred_paths[name] = pp

        hist_args = args.historical
        if not hist_args:
            # flag absent or bare --historical: auto-attach the default
            # published summary when it exists and is readable
            default_h = ROOT / DEFAULT_HISTORICAL
            hist_args = [str(default_h)] if default_h.is_file() else []
        historical = None
        if hist_args:
            historical = {}
            for hp in hist_args:
                p = _resolve(hp)
                try:
                    historical[str(p)] = json.loads(p.read_text())
                except (OSError, json.JSONDecodeError) as e:
                    raise AuditError(f"historical summary {p}: unreadable: {e}")

        payload = build_payload(gold_path, pred_paths, args.question, historical)
        json_text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        md_text = render_markdown(payload)
        _atomic_write(_resolve(args.json_out), json_text)
        _atomic_write(_resolve(args.md_out), md_text)
    except AuditError as e:
        print(f"AUDIT FAILED: {e}", file=sys.stderr)
        return 2
    print(f"wrote {args.json_out} and {args.md_out} "
          f"({payload['counts']['n_rows']} rows, "
          f"{len(payload['arms'])} arms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
