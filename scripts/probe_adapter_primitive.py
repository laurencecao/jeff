"""Probe: does lora_4b_multi give a live Score, and does the sequence readout
fix it if not?

Runs the demo's exact questions against each candidate adapter and prints the
Score distribution under both readouts. Read-only; writes only stdout.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jev_clf import schema as S  # noqa: E402
from scripts.jev_clf_lm_eval import build_inputs, label_variants  # noqa: E402

BASE = "Qwen/Qwen3-4B-Instruct-2507"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

STATE = {
    "claim": "The Eiffel Tower is located in Barcelona.",
    "evidence": [
        {
            "title": "evidence 1",
            "text": "The Eiffel Tower is a wrought-iron lattice tower on the "
            "Champ de Mars in Paris, France.",
        }
    ],
}

VERDICT = S.ChoiceQuestion(
    instructions="Read the claim and the evidence passages in `state`. Decide "
    "whether the evidence establishes the claim, contradicts it, or cannot decide it.",
    criteria={
        "supported": "The evidence passages together entail the claim.",
        "refuted": "The evidence passages together contradict the claim.",
        "not_enough_info": "No evidence, or too weak to decide.",
    },
)
NOUL = S.NoulQuestion(
    instructions="Does the evidence contain a date or a number?",
    criteria={"yes": "yes", "no": "no"},
)
SCORE = S.ScoreQuestion(
    instructions="How much evidence is provided for the claim?",
    criteria=[
        "no evidence at all",
        "a single weak or unrelated passage",
        "one relevant passage",
        "several relevant passages",
    ],
)


def probe(model, tok, q, tag):
    labels = S.label_space(q)
    variants = label_variants(tok, labels)
    text = build_inputs(tok, STATE, q)
    enc = tok(text, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)
    with torch.no_grad():
        logits = model(**enc).logits[0, -1].float()

    first = [float(logits[variants[l][0]]) for l in labels]
    p_first = torch.softmax(torch.tensor(first), dim=-1).tolist()

    seq_scores = []
    for lab in labels:
        seq = variants[lab]
        full = torch.tensor([enc["input_ids"][0].tolist() + seq], device=DEVICE)
        with torch.no_grad():
            out = model(input_ids=full).logits[0]
        lp = 0.0
        for k, tid in enumerate(seq):
            pos = full.shape[1] - len(seq) + k - 1
            lp += float(torch.log_softmax(out[pos].float(), dim=-1)[tid])
        seq_scores.append(lp)
    p_seq = torch.softmax(torch.tensor(seq_scores), dim=-1).tolist()

    dead = len({round(x, 6) for x in p_first}) == 1
    top_first = max(zip(labels, p_first), key=lambda kv: kv[1])
    top_seq = max(zip(labels, p_seq), key=lambda kv: kv[1])
    print(f"  [{tag}] {'DEAD ' if dead else '     '}first_token -> {top_first[0]} "
          f"({top_first[1]:.4f})" + ("  <-- uniform" if dead else ""))
    print(f"  [{tag}]            sequence    -> {top_seq[0]} ({top_seq[1]:.4f})")
    print(f"  [{tag}]            first_token dist = "
          f"{ {l: round(p, 4) for l, p in zip(labels, p_first)} }")
    print(f"  [{tag}]            sequence    dist = "
          f"{ {l: round(p, 4) for l, p in zip(labels, p_seq)} }")


def main() -> None:
    tok = AutoTokenizer.from_pretrained(BASE)
    base = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.bfloat16, device_map="cpu"
    )

    for name in ["lora_4b", "lora_4b_multi"]:
        print(f"\n=== adapter {name} ===", flush=True)
        model = PeftModel.from_pretrained(base, str(ROOT / f"artifacts/jev_clf/{name}"))
        model = model.to(DEVICE).eval()
        with torch.no_grad():
            probe(model, tok, VERDICT, "verdict")
            probe(model, tok, NOUL, "noul  ")
            probe(model, tok, SCORE, "score ")
        model = model.to("cpu")
        del model
        torch.mps.empty_cache() if DEVICE == "mps" else None

    print("\n[probe] DONE", flush=True)


if __name__ == "__main__":
    main()
