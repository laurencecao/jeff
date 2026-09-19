"""Unit tests for scripts/audit_decision_results.py — CPU only, no models.

Covers the fail-closed contract (empty/missing/extra/duplicate keys, bad label
sets, non-finite/negative/unnormalized probabilities, non-one-hot gold),
known-value metrics (perfect and uniform predictors), conservative tie
handling in coverage-targeted selective risk, and paired McNemar/bootstrap
counts.

Run: uv run python -m scripts.test_audit_decision_results

"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

from scripts import audit_decision_results as aud


class _Raises:
    def __init__(self, exc, match=None):
        self.exc, self.match = exc, match

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is None:
            raise AssertionError(f"expected {self.exc.__name__}, none raised")
        if not issubclass(et, self.exc):
            return False
        if self.match and self.match not in str(ev):
            raise AssertionError(
                f"{self.exc.__name__} message {ev!r} lacks {self.match!r}")
        return True


def raises(exc, match=None):
    return _Raises(exc, match)


def approx(a, b, rel=1e-9, abs_=1e-12):
    assert math.isclose(a, b, rel_tol=rel, abs_tol=abs_), f"{a!r} != {b!r}"

LABELS = ["not_enough_info", "refuted", "supported"]  # sorted


def _gold_row(row_id, label, qid="verdict", source="fever", n_ev=1):
    dist = {l: 0.0 for l in LABELS}
    dist[label] = 1.0
    return {
        "row_id": row_id, "source": source,
        "state": {"claim": "c", "evidence": [{"title": "t", "text": "x"}] * n_ev},
        "questions": {qid: {"kind": "choice", "instructions": "i",
                            "criteria": {l: l for l in LABELS}}},
        "labels": {qid: dist},
    }


def _pred_row(row_id, probs, conf=None, qid="verdict"):
    if conf is None:
        conf = max(probs.values())
    return {"row_id": row_id, "question_id": qid, "probs": probs,
            "confidence": conf, "model": "test", "latency_ms": 1.0}


def _write_jsonl(path: Path, rows) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _tmp(tmp_path, name="gold.jsonl"):
    return tmp_path / name


def _gold_file(tmp_path, rows):
    return _write_jsonl(_tmp(tmp_path, "gold.jsonl"), rows)


def _pred_file(tmp_path, rows, name="preds_a.jsonl"):
    return _write_jsonl(_tmp(tmp_path, name), rows)


# --------------------------------------------------------------------------
# Known-value metrics
# --------------------------------------------------------------------------

def test_perfect_predictions(tmp_path):
    gold = [_gold_row("r1", "supported"), _gold_row("r2", "refuted"),
            _gold_row("r3", "not_enough_info")]
    preds = [_pred_row("r1", {"supported": 1.0, "refuted": 0.0,
                              "not_enough_info": 0.0}),
             _pred_row("r2", {"supported": 0.0, "refuted": 1.0,
                              "not_enough_info": 0.0}),
             _pred_row("r3", {"supported": 0.0, "refuted": 0.0,
                              "not_enough_info": 1.0})]
    payload = aud.build_payload(_gold_file(tmp_path, gold),
                                {"a": _pred_file(tmp_path, preds)},
                                "verdict", None)
    m = payload["arms"]["a"]
    assert m["n"] == 3 and m["n_correct"] == 3
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    # p_gold = 1 -> clipped to 1 -> -log(1) = 0
    assert m["nll"] == (0.0)
    assert m["brier_multiclass"] == (0.0)
    assert m["ece"]["top_label_10bin"] == (0.0)
    assert m["ece"]["stored_confidence_10bin"] == (0.0)
    assert m["ece"]["adaptive_equal_count_10bin"] == (0.0)
    assert m["ece"]["classwise_ovr_macro"] == (0.0)
    for s in m["selective_risk"].values():
        assert s["risk"] == 0.0 and s["coverage"] == 1.0


def test_uniform_predictions(tmp_path):
    """Uniform 1/3 probs: top-1 picks first INSERTED key (eval policy).

    LABELS is sorted, so first-inserted == first-sorted here; the
    differentiating case is test_argmax_tie_insertion_order."""
    u = {l: 1.0 / 3.0 for l in LABELS}
    gold = [_gold_row("r1", "not_enough_info"), _gold_row("r2", "refuted"),
            _gold_row("r3", "supported")]
    preds = [_pred_row(r, dict(u), conf=1.0 / 3.0) for r in ("r1", "r2", "r3")]
    payload = aud.build_payload(_gold_file(tmp_path, gold),
                                {"a": _pred_file(tmp_path, preds)},
                                "verdict", None)
    m = payload["arms"]["a"]
    # argmax tie -> first sorted label = not_enough_info -> only r1 correct
    approx(m["accuracy"], 1.0 / 3.0)
    assert m["n_correct"] == 1
    # NLL = -log(1/3)
    approx(m["nll"], -math.log(1.0 / 3.0))
    # Brier per row: (1/3-1)^2 + 2*(1/3)^2 = 4/9 + 2/9 = 6/9
    assert m["brier_multiclass"] == (6.0 / 9.0)
    # all confidences equal 1/3 -> single occupied bin, |acc - conf| = 0
    assert m["ece"]["top_label_10bin"] == (0.0)
    # macro F1: only not_enough_info has tp=1,fp=2 -> P=1/3,R=1,F1=0.5;
    # refuted/supported F1=0 -> macro = 1/6
    assert m["macro_f1"] == (1.0 / 6.0)


def test_argmax_tie_insertion_order(tmp_path):
    """Exact tie: first-inserted key wins (eval policy), NOT sorted order.

    probs insertion order is supported,refuted,nei with supported==refuted;
    insertion-order top-1 = supported (correct), sorted argmax = refuted.
    """
    gold = [_gold_row("r1", "supported")]
    preds = [_pred_row("r1", {"supported": 0.5, "refuted": 0.5,
                              "not_enough_info": 0.0})]
    payload = aud.build_payload(_gold_file(tmp_path, gold),
                                {"a": _pred_file(tmp_path, preds)},
                                "verdict", None)
    m = payload["arms"]["a"]
    assert m["accuracy"] == 1.0
    assert m["n_exact_argmax_ties"] == 1
    assert m["accuracy_alt_sorted_tiebreak"] == 0.0


def test_nll_epsilon_clips_zero_prob(tmp_path):
    gold = [_gold_row("r1", "supported")]
    preds = [_pred_row("r1", {"supported": 0.0, "refuted": 1.0,
                              "not_enough_info": 0.0})]
    payload = aud.build_payload(_gold_file(tmp_path, gold),
                                {"a": _pred_file(tmp_path, preds)},
                                "verdict", None)
    m = payload["arms"]["a"]
    approx(m["nll"], -math.log(aud.NLL_EPSILON))
    assert m["nll_epsilon"] == aud.NLL_EPSILON


# --------------------------------------------------------------------------
# Fail-closed input validation
# --------------------------------------------------------------------------

def _good_gold(tmp_path):
    return _gold_file(tmp_path, [_gold_row("r1", "supported"),
                                 _gold_row("r2", "refuted")])


def _good_preds(tmp_path, name="preds_a.jsonl"):
    return _pred_file(tmp_path, [
        _pred_row("r1", {"supported": 0.9, "refuted": 0.05,
                         "not_enough_info": 0.05}),
        _pred_row("r2", {"supported": 0.1, "refuted": 0.8,
                         "not_enough_info": 0.1})], name)


def test_empty_pred_file_fails(tmp_path):
    p = _tmp(tmp_path, "preds_empty.jsonl")
    p.write_text("")
    with raises(aud.AuditError):
        aud.build_payload(_good_gold(tmp_path), {"a": p}, "verdict", None)


def test_missing_key_fails(tmp_path):
    p = _pred_file(tmp_path, [
        _pred_row("r1", {"supported": 1.0, "refuted": 0.0,
                         "not_enough_info": 0.0})])
    with raises(aud.AuditError, match="missing"):
        aud.build_payload(_good_gold(tmp_path), {"a": p}, "verdict", None)


def test_extra_key_fails(tmp_path):
    rows = [_pred_row("r1", {"supported": 1.0, "refuted": 0.0,
                             "not_enough_info": 0.0}),
            _pred_row("r2", {"supported": 0.0, "refuted": 1.0,
                             "not_enough_info": 0.0}),
            _pred_row("rX", {"supported": 0.0, "refuted": 1.0,
                             "not_enough_info": 0.0})]
    with raises(aud.AuditError, match="extra"):
        aud.build_payload(_good_gold(tmp_path),
                          {"a": _pred_file(tmp_path, rows)}, "verdict", None)


def test_duplicate_pred_key_fails(tmp_path):
    rows = [_pred_row("r1", {"supported": 1.0, "refuted": 0.0,
                             "not_enough_info": 0.0}),
            _pred_row("r1", {"supported": 0.0, "refuted": 1.0,
                             "not_enough_info": 0.0})]
    with raises(aud.AuditError, match="duplicate"):
        aud.build_payload(_good_gold(tmp_path),
                          {"a": _pred_file(tmp_path, rows)}, "verdict", None)


def test_duplicate_gold_key_fails(tmp_path):
    gold = [_gold_row("r1", "supported"), _gold_row("r1", "refuted")]
    with raises(aud.AuditError, match="duplicate"):
        aud.build_payload(_gold_file(tmp_path, gold),
                          {"a": _good_preds(tmp_path)}, "verdict", None)


def test_wrong_question_id_fails(tmp_path):
    rows = [_pred_row("r1", {"supported": 1.0, "refuted": 0.0,
                             "not_enough_info": 0.0}, qid="other"),
            _pred_row("r2", {"supported": 0.0, "refuted": 1.0,
                             "not_enough_info": 0.0})]
    with raises(aud.AuditError, match="question_id"):
        aud.build_payload(_good_gold(tmp_path),
                          {"a": _pred_file(tmp_path, rows)}, "verdict", None)


def test_nonfinite_prob_fails(tmp_path):
    rows = [_pred_row("r1", {"supported": float("nan"), "refuted": 0.0,
                             "not_enough_info": 0.0}),
            _pred_row("r2", {"supported": 0.0, "refuted": 1.0,
                             "not_enough_info": 0.0})]
    with raises(aud.AuditError, match="non-finite"):
        aud.build_payload(_good_gold(tmp_path),
                          {"a": _pred_file(tmp_path, rows)}, "verdict", None)


def test_negative_prob_fails(tmp_path):
    rows = [_pred_row("r1", {"supported": -0.5, "refuted": 1.5,
                             "not_enough_info": 0.0}),
            _pred_row("r2", {"supported": 0.0, "refuted": 1.0,
                             "not_enough_info": 0.0})]
    with raises(aud.AuditError, match="negative"):
        aud.build_payload(_good_gold(tmp_path),
                          {"a": _pred_file(tmp_path, rows)}, "verdict", None)


def test_unnormalized_prob_fails(tmp_path):
    rows = [_pred_row("r1", {"supported": 0.9, "refuted": 0.9,
                             "not_enough_info": 0.9}),
            _pred_row("r2", {"supported": 0.0, "refuted": 1.0,
                             "not_enough_info": 0.0})]
    with raises(aud.AuditError, match="sum"):
        aud.build_payload(_good_gold(tmp_path),
                          {"a": _pred_file(tmp_path, rows)}, "verdict", None)


def test_bad_label_set_fails(tmp_path):
    rows = [_pred_row("r1", {"supported": 1.0, "refuted": 0.0,
                             "WRONG": 0.0}),
            _pred_row("r2", {"supported": 0.0, "refuted": 1.0,
                             "not_enough_info": 0.0})]
    with raises(aud.AuditError, match="label set"):
        aud.build_payload(_good_gold(tmp_path),
                          {"a": _pred_file(tmp_path, rows)}, "verdict", None)


def test_non_onehot_gold_fails(tmp_path):
    bad = _gold_row("r1", "supported")
    bad["labels"]["verdict"] = {"supported": 0.7, "refuted": 0.3,
                                "not_enough_info": 0.0}
    gold = [bad, _gold_row("r2", "refuted")]
    with raises(aud.AuditError, match="one-hot"):
        aud.build_payload(_gold_file(tmp_path, gold),
                          {"a": _good_preds(tmp_path)}, "verdict", None)


def test_missing_confidence_fails(tmp_path):
    r = _pred_row("r1", {"supported": 1.0, "refuted": 0.0,
                         "not_enough_info": 0.0})
    del r["confidence"]
    rows = [r, _pred_row("r2", {"supported": 0.0, "refuted": 1.0,
                                "not_enough_info": 0.0})]
    with raises(aud.AuditError, match="confidence"):
        aud.build_payload(_good_gold(tmp_path),
                          {"a": _pred_file(tmp_path, rows)}, "verdict", None)


def test_cli_failure_writes_nothing(tmp_path):
    gold = _gold_file(tmp_path, [_gold_row("r1", "supported")])
    bad = _pred_file(tmp_path, [])  # empty
    jo, mo = tmp_path / "out.json", tmp_path / "out.md"
    rc = aud.main(["--gold", str(gold), "--preds", str(bad),
                   "--json-out", str(jo), "--md-out", str(mo),
                   "--historical"])
    assert rc != 0
    assert not jo.exists() and not mo.exists()


# --------------------------------------------------------------------------
# Selective risk tie handling
# --------------------------------------------------------------------------

def test_selective_risk_tie_group_fully_included():
    # 10 rows, target 0.5 -> k=5; confidences: five 0.9s then five 0.9s tied
    # at the boundary -> tau=0.9, all 10 included, coverage 1.0 >= 0.5.
    conf = np.array([0.9] * 10)
    correct = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=float)
    sr = aud.selective_risk(conf, correct, targets=(0.5,))
    s = sr["0.5"]
    assert s["n_selected"] == 10
    assert s["coverage"] == 1.0
    assert s["confidence_threshold"] == 0.9
    assert s["risk"] == 0.5


def test_selective_risk_no_ties_exact_coverage():
    conf = np.linspace(0.05, 0.95, 10)
    correct = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=float)
    sr = aud.selective_risk(conf, correct, targets=(0.5,))
    s = sr["0.5"]
    assert s["n_selected"] == 5 and s["coverage"] == 0.5
    assert s["risk"] == 0.0  # top-5 confidence rows are exactly the correct ones


# --------------------------------------------------------------------------
# Paired comparison
# --------------------------------------------------------------------------

def test_paired_mcnemar_and_bootstrap(tmp_path):
    """A right on r1,r2; B right on r2,r3 -> b_only=1, a_only=1, disc=2."""
    gold = [_gold_row("r1", "supported"), _gold_row("r2", "refuted"),
            _gold_row("r3", "not_enough_info")]
    win = {"supported": 1.0, "refuted": 0.0, "not_enough_info": 0.0}
    lose = {"supported": 0.0, "refuted": 1.0, "not_enough_info": 0.0}
    # A: r1 correct, r2 correct, r3 wrong
    a = [_pred_row("r1", win), _pred_row("r2",
         {"supported": 0.0, "refuted": 1.0, "not_enough_info": 0.0}),
         _pred_row("r3", lose)]
    # B: r1 wrong, r2 correct, r3 correct
    b = [_pred_row("r1", lose), _pred_row("r2",
         {"supported": 0.0, "refuted": 1.0, "not_enough_info": 0.0}),
         _pred_row("r3",
         {"supported": 0.0, "refuted": 0.0, "not_enough_info": 1.0})]
    payload = aud.build_payload(
        _gold_file(tmp_path, gold),
        {"a": _pred_file(tmp_path, a, "preds_a.jsonl"),
         "b": _pred_file(tmp_path, b, "preds_b.jsonl")},
        "verdict", None)
    pr = payload["paired"]
    assert pr["a"] == "a" and pr["b"] == "b" and pr["n"] == 3
    assert pr["accuracy_a"] == (2 / 3)
    assert pr["accuracy_b"] == (2 / 3)
    assert pr["delta_b_minus_a"] == 0.0
    mc = pr["mcnemar"]
    assert mc["b_only_correct"] == 1 and mc["a_only_correct"] == 1
    assert mc["discordant"] == 2
    # binomtest(1, 2, 0.5) two-sided = 1.0
    assert mc["p_exact_two_sided"] == 1.0
    bs = pr["bootstrap"]
    assert bs["seed"] == aud.BOOTSTRAP_SEED
    assert bs["resamples"] == aud.BOOTSTRAP_RESAMPLES
    lo, hi = bs["ci95"]
    assert lo <= 0.0 <= hi


def test_paired_mcnemar_asymmetric(tmp_path):
    """B fixes 4 rows A got wrong, breaks 0 -> binomtest(0,4,0.5)=0.125."""
    gold = [_gold_row(f"r{i}", "supported") for i in range(4)]
    right = {"supported": 1.0, "refuted": 0.0, "not_enough_info": 0.0}
    wrong = {"supported": 0.0, "refuted": 1.0, "not_enough_info": 0.0}
    a = [_pred_row(f"r{i}", wrong) for i in range(4)]
    b = [_pred_row(f"r{i}", right) for i in range(4)]
    payload = aud.build_payload(
        _gold_file(tmp_path, gold),
        {"a": _pred_file(tmp_path, a, "preds_a.jsonl"),
         "b": _pred_file(tmp_path, b, "preds_b.jsonl")},
        "verdict", None)
    mc = payload["paired"]["mcnemar"]
    assert mc["b_only_correct"] == 4 and mc["a_only_correct"] == 0
    assert mc["p_exact_two_sided"] == 0.125
    ci = payload["paired"]["bootstrap"]["ci95"]
    assert ci[0] > 0.9 and ci[1] <= 1.0


# --------------------------------------------------------------------------
# End-to-end CLI on the real artifacts (read-only; writes to tmp)
# --------------------------------------------------------------------------

def test_cli_real_artifacts(tmp_path):
    gold = aud.ROOT / "data/factcheck/eval_large.jsonl"
    a = aud.ROOT / "data/factcheck/preds_ours_large.jsonl"
    b = aud.ROOT / "data/factcheck/preds_jev_large.jsonl"
    if not (gold.is_file() and a.is_file() and b.is_file()):
        print("  SKIP test_cli_real_artifacts: scale artifacts absent"); return
    jo, mo = tmp_path / "audit.json", tmp_path / "audit.md"
    rc = aud.main(["--gold", str(gold), "--preds", str(a), str(b),
                   "--json-out", str(jo), "--md-out", str(mo),
                   "--historical"])
    assert rc == 0
    payload = json.loads(jo.read_text())
    assert payload["counts"]["n_rows"] == 9730
    assert set(payload["arms"]) == {"ours_large", "jev_large"}
    assert payload["paired"]["n"] == 9730
    for m in payload["arms"].values():
        assert 0.0 <= m["accuracy"] <= 1.0
        assert m["nll"] >= 0.0
        assert m["brier_multiclass"] >= 0.0
        for e in m["ece"].values():
            if isinstance(e, dict):
                assert all(0.0 <= v <= 1.0 for v in e.values())
            else:
                assert 0.0 <= e <= 1.0
        for s in m["selective_risk"].values():
            assert s["coverage"] >= s["target_coverage"]
    md = mo.read_text()
    assert "researchMax gap audit" in md
    assert "Historical published summaries" in md  # jev_large.json auto-attached
    # sha256 recorded for every input
    for info in [payload["inputs"]["gold"],
                 *payload["inputs"]["predictions"].values()]:
        assert len(info["sha256"]) == 64


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        tp = Path(td)
        tests = [(n, f) for n, f in sorted(globals().items())
                 if n.startswith("test_") and callable(f)]
        ok = True
        for name, fn in tests:
            try:
                if "tmp_path" in fn.__code__.co_varnames:
                    d = tp / name
                    d.mkdir()
                    fn(d)
                else:
                    fn()
                print(f"  PASS {name}")
            except Exception as e:  # noqa: BLE001 - report all, exit nonzero
                ok = False
                print(f"  FAIL {name}: {e}")
        print("\nRESULT:", "PASS" if ok else "FAIL")
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
