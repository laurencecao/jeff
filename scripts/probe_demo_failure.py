"""Why did Jeff 1 answer 'supported' on the demo conjunction? An ablation.

The demo claim was: "The new training program made participants both faster and
more accurate than standard training." Evidence: new program faster (42s vs
55s) but accuracy TIED (91% both). Correct = refuted; Jeff 1 said supported.

This isolates the cause by varying ONE thing at a time from the exact demo
payload, so the answer is measured rather than asserted.

Candidate causes:
  H1 conjunction        — the claim joins two assertions with "and", and one
                          half is true, so the model latches onto the true half.
  H2 single passage     — both facts live in ONE passage, so "evidence exists"
                          collapses to "supported".
  H3 comparison wording — "more accurate than" is a comparative, and the
                          evidence states equality (91% vs 91%).

Each variant changes exactly one of those.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jev_clf.client import SystemOneClient  # noqa: E402
from jev_clf.eval import make_factcheck_choice  # noqa: E402

ADAPTER = str(ROOT / "artifacts/jev_clf/lora_4b_multi")

SPEED = ("Participants in the new program completed the task in an average of "
         "42 seconds, compared with 55 seconds for the standard-training group.")
ACC_EQ = "Both groups answered exactly 91% of questions correctly."
ACC_BETTER = ("The new-program group scored 96% correct on the final assessment, "
              "against 91% for the standard-training group.")

CONJ = ("The new training program made participants both faster and more accurate "
        "than standard training.")
SPEED_ONLY = "The new training program made participants faster than standard training."
ACC_ONLY = "The new training program made participants more accurate than standard training."

VARIANTS: list[tuple[str, str, str, list[str], str, str]] = [
    ("demo-exact", "THE DEMO PAYLOAD", CONJ, [SPEED, ACC_EQ], "refuted",
     "1 passage, conjunction, accuracy tied"),
    ("demo-split", "H2: same text, 2 passages", CONJ, [SPEED, "|", ACC_EQ], "refuted",
     "only chunking differs from demo-exact"),
    ("speed-only", "H1a: claim drops the accuracy half", SPEED_ONLY, [SPEED, ACC_EQ], "supported",
     "conjunction removed; true half alone -> supported is CORRECT"),
    ("acc-only", "H1b: claim keeps only the false half", ACC_ONLY, [SPEED, ACC_EQ], "refuted",
     "conjunction removed; false half alone -> does it catch it?"),
    ("conj-both-true", "H1 control: both halves true", CONJ, [SPEED, ACC_BETTER], "supported",
     "same conjunction, accuracy now really better -> supported is CORRECT"),
]


def main() -> None:
    client = SystemOneClient(adapter=ADAPTER)
    print(f"[why] ours = {client.model_id}\n", flush=True)
    print(f"  {'variant':16} {'what':34} {'gold':10} {'got':16} {'conf':>6}  ok", flush=True)
    print(f"  {'-'*16} {'-'*34} {'-'*10} {'-'*16} {'-'*6}  --", flush=True)

    for name, what, claim, parts, gold, note in VARIANTS:
        if "|" in parts:
            i = parts.index("|")
            ev = [{"title": "evidence 1", "text": " ".join(parts[:i])},
                  {"title": "evidence 2", "text": " ".join(parts[i+1:])}]
        elif len(parts) == 1:
            ev = [{"title": "evidence 1", "text": parts[0]}]
        else:
            ev = [{"title": "evidence 1", "text": " ".join(parts)}]
        with torch.no_grad():
            out = client.system_one({"claim": claim, "evidence": ev},
                                    {"verdict": make_factcheck_choice()})
        a = out.choices["verdict"]
        ok = a.choice == gold
        print(f"  {name:16} {what:34} {gold:10} {a.choice:16} {a.confidence:>6.3f}  "
              f"{'PASS' if ok else 'FAIL'}", flush=True)
        print(f"      claim:  {claim}", flush=True)
        print(f"      note:   {note}", flush=True)

    print(flush=True)
    print("[why] read the speed-only vs acc-only pair: if speed-only passes and", flush=True)
    print("      acc-only fails, the model treats the TRUE half as decisive.", flush=True)
    print("[why] read demo-exact vs demo-split: if the verdict changes, chunking", flush=True)
    print("      alone flips it, which is an instability rather than a reasoning gap.", flush=True)


if __name__ == "__main__":
    main()
