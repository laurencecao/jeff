"""Zero-shot / adapter label readout from a real language model.

Asks a causal LM to classify against label definitions supplied in the prompt,
then reads the classification out of the model's own next-token distribution —
restricted to the label tokens. No head is trained or bolted on: the classifier
IS the language model, which is what keeps this text-conditioned (a label set of
a different size or wording still works, because it arrives in the prompt).

Usage:
    uv run python -m scripts.jev_clf_lm_eval --model Qwen/Qwen2.5-1.5B-Instruct
    uv run python -m scripts.jev_clf_lm_eval --model Qwen/Qwen2.5-1.5B-Instruct \
        --adapter artifacts/jev_clf/lora_lm --agreement

Readout variants:
    first_token  softmax over the first token id of each label (default)
    sequence     score each label as a whole token sequence, softmax across labels
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jev_clf import schema as S  # noqa: E402
from jev_clf.eval import agreement as agreement_metric  # noqa: E402
from jev_clf.eval import ground_truth_metrics  # noqa: E402
from jev_clf.model import state_to_text  # noqa: E402

SYSTEM = (
    "You are a fact-checking classifier. You are given a claim and the evidence "
    "passages retrieved for it. You answer with exactly one verdict label."
)


def label_variants(tok, labels: list[str]) -> dict[str, list[int]]:
    """Token ids for each label, trying a leading space first (mid-sentence form)."""
    out: dict[str, list[int]] = {}
    for label in labels:
        ids = tok(" " + label, add_special_tokens=False)["input_ids"]
        if not ids:
            ids = tok(label, add_special_tokens=False)["input_ids"]
        out[label] = ids
    return out


def build_inputs(tok, state, question) -> str:
    user = (
        S.question_to_text(question)
        + "\n\nState:\n"
        + state_to_text(state)
        + "\n\nVerdict:"
    )
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    try:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"{SYSTEM}\n\n{user}\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--split", default="val")
    ap.add_argument("--readout", default="first_token", choices=["first_token", "sequence"])
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--agreement", action="store_true", help="also score eval_schemas val rows")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()

    tag = args.model
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter).to(device).eval()
        tag = f"{args.model}+{Path(args.adapter).name}"

    def score_rows(rows: list[S.DecisionRow], label: str) -> list[S.PredictionRow]:
        preds: list[S.PredictionRow] = []
        for row in rows:
            qid, question = next(iter(row.questions.items()))
            labels = S.label_space(question)
            variants = label_variants(tok, labels)
            text = build_inputs(tok, row.state, question)
            enc = tok(text, return_tensors="pt", truncation=True, max_length=2048).to(device)
            t0 = time.perf_counter()
            with torch.no_grad():
                logits = model(**enc).logits[0, -1].float()
            latency = (time.perf_counter() - t0) * 1000

            if args.readout == "first_token":
                sub = torch.tensor([logits[variants[label][0]] for label in labels])
            else:
                # score each label as a whole sequence starting at the last position
                scores = []
                for lab in labels:
                    seq = variants[lab]
                    enc_lab = tok(text, return_tensors="pt", truncation=True, max_length=2048)
                    ids = enc_lab["input_ids"][0].tolist()
                    full = torch.tensor([ids + seq], device=device)
                    with torch.no_grad():
                        lg = model(full).logits[0].float()
                    lp = 0.0
                    for j, tokid in enumerate(seq):
                        row_logits = lg[len(ids) - 1 + j]
                        lp += float(torch.log_softmax(row_logits, dim=-1)[tokid])
                    scores.append(lp)
                sub = torch.tensor(scores)
            probs = torch.softmax(sub, dim=-1)
            preds.append(
                S.PredictionRow(
                    row_id=row.row_id,
                    question_id=qid,
                    probs={lab: float(p) for lab, p in zip(labels, probs)},
                    confidence=float(probs.max()),
                    model=tag,
                    latency_ms=latency,
                    meta={"readout": args.readout, "adapter": args.adapter},
                )
            )
        return preds

    gt_rows = [r for r in S.read_rows(ROOT / "data/factcheck/ground_truth.jsonl") if r.split == args.split]
    if args.max_rows:
        gt_rows = gt_rows[: args.max_rows]
    preds = score_rows(gt_rows, "gt")
    m = ground_truth_metrics(gt_rows, preds)
    print(json.dumps({k: m[k] for k in ("n", "accuracy", "macro_f1", "ece", "brier")}, indent=2))

    result = {"model": tag, "readout": args.readout, "split": args.split, "ground_truth": m}

    if args.agreement:
        ev_rows = [
            r for r in S.read_rows(ROOT / "data/factcheck/eval_schemas.jsonl") if r.split == args.split
        ]
        if args.max_rows:
            ev_rows = ev_rows[: args.max_rows]
        ev_preds = score_rows(ev_rows, "ev")
        agg = agreement_metric(ev_rows, ev_preds)
        print("agreement-with-Jev on held-out schemas:", json.dumps(agg, indent=2))
        result["agreement_with_jev"] = agg

    out = Path(args.out) if args.out else ROOT / "results" / f"lm_eval_{Path(args.model).name}_{args.readout}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
