"""Autoresearch metric for jev_clf: how close is our Jev replacement to Jev?

Evaluates the current adapter on the VAL split against human ground-truth
labels and prints `METRIC <name>=<value>` lines.

Primary metric is VAL ACCURACY. The test split is sealed and only used for the
final claim — optimising against it would make "as good as Jev" a
self-fulfilling number.

Reference bar (val, n=199): live Jev 1.13.0 = 0.769 accuracy, ECE 0.114.

The decision rule is read from `configs/jev_clf_infer.yaml` (or
$JEVCLF_INFER_CONFIG), so a candidate is measurable in one harness run instead
of a retrain. The rule operates in PROBABILITY space:

    ensemble_schemas -> average the label distribution over question wordings
    temperature      -> p ** (1/T), renormalised   (changes ECE, not argmax)
    label_bias       -> p * bias, renormalised     (changes the decision)

Deterministic: fixed seed, offline, greedy evaluation.

    uv run python -m scripts.jev_clf_autoresearch
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jev_clf import schema as S  # noqa: E402
from jev_clf.eval import agreement as agreement_metric  # noqa: E402
from jev_clf.eval import ground_truth_metrics  # noqa: E402
from scripts.jev_clf_lm_eval import build_inputs, label_variants  # noqa: E402

BASE_MODEL = os.environ.get("JEVCLF_BASE", "Qwen/Qwen2.5-1.5B-Instruct")
ADAPTER = os.environ.get("JEVCLF_ADAPTER", str(ROOT / "artifacts/jev_clf/lora_lm"))
INFER_CONFIG = Path(
    os.environ.get("JEVCLF_INFER_CONFIG", str(ROOT / "configs" / "jev_clf_infer.yaml"))
)
SPLIT = "val"
BAR_ACCURACY = 0.769  # live Jev 1.13.0 on this split (n=199)

DEFAULTS = {
    "readout": "first_token",
    "max_length": 2048,
    "temperature": 1.0,
    "label_bias": {},
    "ensemble_schemas": False,
}


def load_infer_config() -> dict:
    cfg = dict(DEFAULTS)
    if INFER_CONFIG.exists():
        cfg.update(yaml.safe_load(INFER_CONFIG.read_text()) or {})
    return cfg


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def ensemble_questions(cfg: dict, default_question) -> list:
    """Question wordings to average over. `ensemble_schemas` is False, "all",
    or a list of schema ids. All wordings ask the same 3-label question."""
    setting = cfg.get("ensemble_schemas") or False
    if not setting:
        return [default_question]

    from jev_clf.gen import _SCHEMAS, _build_question  # noqa: PLC0415

    if setting == "all":
        wanted = [s["id"] for s in _SCHEMAS if s["kind"] == "choice"]
    elif isinstance(setting, (list, tuple)):
        wanted = list(setting)
    else:
        raise ValueError(f"unsupported ensemble_schemas: {setting!r}")

    by_id = {s["id"]: s for s in _SCHEMAS}
    out = []
    for sid in wanted:
        if sid not in by_id:
            raise ValueError(f"unknown schema id in ensemble_schemas: {sid!r}")
        out.append(next(iter(_build_question(by_id[sid]).values())))
    if not out:
        raise ValueError("ensemble_schemas selected no schemas")
    return out


def main() -> None:
    torch.manual_seed(0)
    cfg = load_infer_config()
    device = pick_device()
    dtype = torch.float32 if device == "cpu" else torch.bfloat16

    adapter_path = Path(
        str(ROOT / cfg["adapter"]) if cfg.get("adapter") else ADAPTER
    )
    if not adapter_path.exists():
        raise SystemExit(f"FAILED: adapter not found at {adapter_path}")

    base_model_name = cfg.get("base_model") or BASE_MODEL
    tok = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(base_model_name, dtype=dtype).to(device)

    from peft import PeftModel

    model = PeftModel.from_pretrained(model, str(adapter_path)).to(device).eval()
    tag = f"{BASE_MODEL}+{adapter_path.name}"

    max_length = int(cfg.get("max_length", 2048))
    temperature = float(cfg.get("temperature", 1.0) or 1.0)
    bias = cfg.get("label_bias") or {}

    def readout(state, question) -> dict[str, float]:
        """First-token distribution over the question's declared labels."""
        labels = S.label_space(question)
        variants = label_variants(tok, labels)
        text = build_inputs(tok, state, question)
        enc = tok(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            logits = model(**enc).logits[0, -1].float()
        sub = torch.tensor([logits[variants[lab][0]] for lab in labels])
        probs = torch.softmax(sub, dim=-1)
        return {lab: float(p) for lab, p in zip(labels, probs)}

    def decide(state, default_question) -> dict[str, float]:
        questions = ensemble_questions(cfg, default_question)
        acc: dict[str, float] = {}
        for q in questions:
            for lab, p in readout(state, q).items():
                acc[lab] = acc.get(lab, 0.0) + p
        probs = {lab: v / len(questions) for lab, v in acc.items()}

        if temperature != 1.0:
            probs = {lab: p ** (1.0 / temperature) for lab, p in probs.items()}
        if bias:
            probs = {lab: p * float(bias.get(lab, 1.0)) for lab, p in probs.items()}

        total = sum(probs.values())
        if total <= 0:
            raise SystemExit(f"FAILED: zero-mass distribution after decision rule: {probs}")
        return {lab: p / total for lab, p in probs.items()}

    def predict_all(rows: list[S.DecisionRow]) -> tuple[list[S.PredictionRow], float]:
        out: list[S.PredictionRow] = []
        t0 = time.perf_counter()
        for row in rows:
            qid, question = next(iter(row.questions.items()))
            probs = decide(row.state, question)
            out.append(
                S.PredictionRow(
                    row_id=row.row_id,
                    question_id=qid,
                    probs=probs,
                    confidence=max(probs.values()),
                    model=tag,
                )
            )
        return out, time.perf_counter() - t0

    rows = [r for r in S.read_rows(ROOT / "data/factcheck/ground_truth.jsonl") if r.split == SPLIT]
    if not rows:
        raise SystemExit("FAILED: no val rows found")
    preds, elapsed = predict_all(rows)
    gt = ground_truth_metrics(rows, preds)
    S.write_predictions(ROOT / f"data/factcheck/preds_autoresearch_{SPLIT}.jsonl", preds)

    agreement_top1 = None
    ev_rows = [r for r in S.read_rows(ROOT / "data/factcheck/eval_schemas.jsonl") if r.split == SPLIT]
    if ev_rows:
        ev_preds, _ = predict_all(ev_rows)
        agreement_top1 = agreement_metric(ev_rows, ev_preds)["top1_match"]

    print(f"METRIC val_accuracy={gt['accuracy']:.6f}")
    print(f"METRIC val_macro_f1={gt['macro_f1']:.6f}")
    print(f"METRIC val_ece={gt['ece']:.6f}")
    print(f"METRIC val_brier={gt['brier']:.6f}")
    if agreement_top1 is not None:
        print(f"METRIC val_agreement_top1={agreement_top1:.6f}")
    print(f"METRIC val_n={gt['n']}")
    print(f"METRIC wall_seconds={elapsed:.1f}")

    gap = BAR_ACCURACY - gt["accuracy"]
    print(f"ASI bar=jev-1.13.0_val_accuracy:{BAR_ACCURACY}")
    print(f"ASI gap_to_jev={gap:+.4f}")
    print(f"ASI device={device}")
    print(f"ASI base_model={BASE_MODEL}")
    print(f"ASI adapter={adapter_path.name}")
    print(f"ASI temperature={temperature}")
    print(f"ASI ensemble_schemas={cfg.get('ensemble_schemas')}")
    print(f"ASI label_bias={json.dumps(cfg.get('label_bias') or {})}")
    print(f"ASI beats_bar={gt['accuracy'] >= BAR_ACCURACY}")

    (ROOT / "results" / "jev_clf_autoresearch_last.json").write_text(
        json.dumps(
            {
                "model": tag,
                "split": SPLIT,
                "infer_config": cfg,
                "device": device,
                "ground_truth": gt,
                "agreement_top1": agreement_top1,
                "bar_accuracy": BAR_ACCURACY,
                "gap_to_jev": gap,
                "wall_seconds": elapsed,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
