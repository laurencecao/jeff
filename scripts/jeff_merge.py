"""Merge the two LoRA adapters and evaluate the result.

lora_4b        : Choice-only training - our accuracy champion (0.8174 @ n=9730)
lora_4b_multi  : Choice + Noul + Score training - primitive coverage

Both share the same base model, rank, alpha and target modules, so a weight-space
merge is safe. The question is whether the merged model keeps the accuracy AND
the primitive coverage, without a retrain.

    uv run python -m scripts.jeff_merge
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jev_clf import schema as S  # noqa: E402
from jev_clf.eval import ground_truth_metrics  # noqa: E402
from scripts.jev_clf_lm_eval import build_inputs, label_variants  # noqa: E402

BASE = "Qwen/Qwen3-4B-Instruct-2507"
A = ROOT / "artifacts/jev_clf/lora_4b"
B = ROOT / "artifacts/jev_clf/lora_4b_multi"
OUT = ROOT / "artifacts/jev_clf/lora_merged"

WEIGHT_A = 0.5
WEIGHT_B = 0.5


def main() -> None:
    print("loading base + adapter A (choice-only)...")
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16)
    pa = PeftModel.from_pretrained(model, str(A))

    print("loading adapter B (multi-primitive)...")
    pb = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16), str(B)
    )

    sa = pa.base_model.model.state_dict()
    sb = pb.base_model.model.state_dict()
    merged = {}
    for k in sa:
        if k in sb and sa[k].dtype.is_floating_point:
            merged[k] = (WEIGHT_A * sa[k].float() + WEIGHT_B * sb[k].float()).to(sa[k].dtype)
        else:
            merged[k] = sa[k]
    print(f"merged {len(merged)} tensors")

    pa.base_model.model.load_state_dict(merged, strict=True)
    pa.save_pretrained(str(OUT))
    tok.save_pretrained(str(OUT))
    print("wrote", OUT)

    # quick eval on the sealed test split
    rows = [r for r in S.read_rows(ROOT / "data/factcheck/ground_truth.jsonl") if r.split == "test"][:200]
    model.eval()
    preds = []
    for r in rows:
        qid, q = next(iter(r.questions.items()))
        labels = S.label_space(q)
        variants = label_variants(tok, labels)
        text = build_inputs(tok, r.state, q)
        enc = tok(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            logits = model(**enc).logits[0, -1].float()
        sub = torch.tensor([logits[variants[l][0]] for l in labels])
        probs = torch.softmax(sub, dim=-1)
        preds.append(S.PredictionRow(
            row_id=r.row_id, question_id=qid,
            probs={l: float(p) for l, p in zip(labels, probs)},
            confidence=float(probs.max()), model="jeff-merged",
        ))
    m = ground_truth_metrics(rows, preds)
    print(json.dumps({k: m[k] for k in ("n", "accuracy", "macro_f1", "ece", "brier")}, indent=2))


if __name__ == "__main__":
    main()
