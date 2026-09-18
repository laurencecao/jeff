"""Throwaway smoke: zero-shot label readout from a real LM's own token distribution.

Not part of the frozen contract — this exists to validate the readout approach
end-to-end (does a language model, asked to classify against label definitions
supplied in the prompt, put its probability mass on the right label token?) and
to get a real number on disk before the larger model finishes downloading.

Uses the locally cached Qwen/Qwen2.5-0.5B, so it costs no download and runs in
seconds on MPS. A 0.5B base model is a FLOOR, not the headline result.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jev_clf import schema as S  # noqa: E402
from jev_clf.eval import ground_truth_metrics  # noqa: E402
from jev_clf.model import state_to_text  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-0.5B"
SPLIT = "val"


def label_first_token_ids(tok, labels: list[str]) -> dict[str, int]:
    """First token id of each label, tried with and without a leading space."""
    out: dict[str, int] = {}
    for label in labels:
        for variant in (label, " " + label):
            ids = tok(variant, add_special_tokens=False)["input_ids"]
            if ids:
                out[label] = ids[0]
                break
    return out


def main() -> None:
    rows = [r for r in S.read_rows(ROOT / "data/factcheck/ground_truth.jsonl") if r.split == SPLIT]
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to("mps").eval()

    preds: list[S.PredictionRow] = []
    collisions = 0
    for row in rows:
        qid, question = next(iter(row.questions.items()))
        labels = S.label_space(question)
        ids = label_first_token_ids(tok, labels)
        if len(set(ids.values())) != len(labels):
            collisions += 1
        prompt = (
            state_to_text(row.state)
            + "\n"
            + S.question_to_text(question)
            + "\nAnswer with exactly one label:"
        )
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to("mps")
        with torch.no_grad():
            logits = model(**enc).logits[0, -1]
        sub = torch.tensor([logits[ids[label]] for label in labels], dtype=torch.float32)
        probs = torch.softmax(sub, dim=-1)
        preds.append(
            S.PredictionRow(
                row_id=row.row_id,
                question_id=qid,
                probs={label: float(p) for label, p in zip(labels, probs)},
                confidence=float(probs.max()),
                model=f"lm_zeroshot_smoke:{MODEL_ID}",
                meta={"readout": "first_token"},
            )
        )

    m = ground_truth_metrics(rows, preds)
    print(json.dumps({k: m[k] for k in ("n", "accuracy", "macro_f1", "ece", "brier")}, indent=2))
    print("label-token collisions:", collisions)
    dist = {}
    for p in preds:
        top = max(p.probs.items(), key=lambda kv: kv[1])[0]
        dist[top] = dist.get(top, 0) + 1
    print("predicted-label distribution:", dist)
    out = ROOT / "results" / "lm_zeroshot_smoke.json"
    out.write_text(json.dumps({"model": MODEL_ID, "split": SPLIT, "metrics": m,
                               "predicted_distribution": dist,
                               "label_token_collisions": collisions}, indent=2) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
