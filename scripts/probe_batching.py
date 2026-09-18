"""Is batched inference bit-equivalent to per-question inference?

The client currently does one forward pass per question. All questions share
the SAME prompt prefix, so they can be batched into one forward with
LEFT padding, taking each row's own last position.

This probe answers one question first: does batching change the numbers?
If it does, batching is not a safe speedup and the idea dies here.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jev_clf import schema as S  # noqa: E402
from jev_clf.client import SystemOneClient  # noqa: E402
from jev_clf.readout import label_token_variants  # noqa: E402

ADAPTER = str(ROOT / "artifacts/jev_clf/lora_4b_multi")

STATE = {
    "claim": "The Eiffel Tower is located in Barcelona.",
    "evidence": [
        {
            "title": "e1",
            "text": "The Eiffel Tower is a wrought-iron lattice tower on the "
            "Champ de Mars in Paris, France.",
        }
    ],
}

# Four DISTINCT questions -> four distinct prompts. (Batching identical prompts
# would be measuring nothing.)
QUESTIONS = {
    "verdict": S.ChoiceQuestion(
        instructions="Decide whether the evidence establishes the claim, contradicts it, or cannot decide it.",
        criteria={
            "supported": "The evidence entails the claim.",
            "refuted": "The evidence contradicts the claim.",
            "not_enough_info": "Too weak to decide.",
        },
    ),
    "truefalse": S.ChoiceQuestion(
        instructions="Is the claim true?",
        criteria={"true": "The claim is true.", "false": "The claim is false."},
    ),
    "has_date": S.NoulQuestion(
        instructions="Does the evidence contain a date or a number?",
        criteria={"yes": "yes", "no": "no"},
    ),
    "strength": S.ScoreQuestion(
        instructions="How much evidence is provided?",
        criteria=["none", "weak", "relevant", "several"],
    ),
}


def main() -> None:
    from jev_clf.readout import distribution as readout_dist
    from scripts.jev_clf_lm_eval import build_inputs

    client = SystemOneClient(adapter=ADAPTER)
    model, tok = client.model, client.tokenizer
    device = client.device

    # --- per-question (current behaviour) ---
    t0 = time.perf_counter()
    per_q: dict[str, dict[str, float]] = {}
    prompts: dict[str, str] = {}
    for qid, q in QUESTIONS.items():
        text = build_inputs(tok, STATE, q)
        prompts[qid] = text
        per_q[qid] = readout_dist(
            model, tok, text, S.label_space(q), device=device, max_length=2048
        )
    t_per = time.perf_counter() - t0

    # --- batched: LEFT padding, one forward ---
    texts = list(prompts.values())
    old_side = tok.padding_side
    tok.padding_side = "left"
    try:
        t0 = time.perf_counter()
        enc = tok(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits[:, -1, :].float()
        t_batch = time.perf_counter() - t0
    finally:
        tok.padding_side = old_side

    print(f"[batch] per-question total {t_per * 1000:.0f} ms")
    print(f"[batch] batched  total      {t_batch * 1000:.0f} ms  "
          f"({t_per / t_batch:.2f}x)")
    print("\n[batch] probability deltas (first_token readout):")
    worst = 0.0
    for i, (qid, q) in enumerate(QUESTIONS.items()):
        labels = S.label_space(q)
        variants = label_token_variants(tok, labels)
        sub = torch.tensor([logits[i][variants[l][0]] for l in labels])
        bp = torch.softmax(sub, dim=-1)
        bp = {l: float(p) for l, p in zip(labels, bp)}
        d = max(abs(bp[l] - per_q[qid][l]) for l in labels)
        worst = max(worst, d)
        print(f"  {qid:9} batched={ {a: round(b, 6) for a, b in bp.items()} }")
        print(f"  {'':9} per-q  ={ {a: round(b, 6) for a, b in per_q[qid].items()} }"
              f"   maxdelta={d:.3e}")
    print(f"\n[batch] WORST DELTA = {worst:.3e}")
    print(f"[batch] VERDICT: {'EQUIVALENT' if worst < 1e-5 else 'DIVERGES -> unsafe'}")


if __name__ == "__main__":
    main()
