"""Why did the user's real case fail while a structurally identical probe passed?

The user's live payload (1 passage, both facts in ONE text field) returned
"supported" at 0.796. probe_conjunction's conj-falsified-1 -- the SAME claim with
the SAME two facts, but split into TWO passages -- returned "refuted" at 0.515.

That is a 0.28 confidence swing and an argmax flip from nothing but PASSAGE
GRANULARITY. If real, it explains the 1-passage not_enough_info deficit directly:
a single passage gives the model one slot to attend to, so "evidence exists"
collapses to "supported".

This holds the text constant and varies only how it is chunked.
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
ACC = ("Both groups answered exactly 91% of questions correctly.")

CLAIM = ("The new training program made participants both faster and more "
         "accurate than standard training.")

VARIANTS: list[tuple[str, list[dict[str, str]], str]] = [
    (
        "A: 1 passage, concatenated (THE USER'S EXACT SHAPE)",
        [{"title": "evidence 1", "text":
          "Researchers compared 100 participants who completed a new training "
          "program with 100 participants who received standard training. "
          + SPEED + " " + ACC}],
        "refuted",
    ),
    (
        "B: 2 passages, split at the accuracy sentence",
        [{"title": "Trial results", "text": SPEED},
         {"title": "Accuracy", "text": ACC}],
        "refuted",
    ),
    (
        "C: 2 passages, generic titles",
        [{"title": "evidence 1", "text": SPEED},
         {"title": "evidence 2", "text": ACC}],
        "refuted",
    ),
    (
        "D: 1 passage, speed ONLY (accuracy absent entirely)",
        [{"title": "evidence 1", "text": SPEED}],
        "not_enough_info",
    ),
    (
        "E: 2 passages, speed only + an unrelated second passage",
        [{"title": "evidence 1", "text": SPEED},
         {"title": "evidence 2", "text":
          "The study was funded by a regional education grant."}],
        "not_enough_info",
    ),
]


def main() -> None:
    client = SystemOneClient(adapter=ADAPTER)
    print(f"[gran] ours = {client.model_id}\n", flush=True)
    print(f"  {'variant':52} {'gold':16} {'got':16} {'conf':>6}  ok", flush=True)
    print(f"  {'-'*52} {'-'*16} {'-'*16} {'-'*6}  --", flush=True)

    n_ok = 0
    for name, evidence, gold in VARIANTS:
        state = {"claim": CLAIM, "evidence": evidence}
        with torch.no_grad():
            out = client.system_one(state, {"verdict": make_factcheck_choice()})
        a = out.choices["verdict"]
        ok = a.choice == gold
        n_ok += ok
        print(f"  {name:52} {gold:16} {a.choice:16} {a.confidence:>6.3f}  "
              f"{'PASS' if ok else 'FAIL'}", flush=True)

    print(f"\n[gran] {n_ok}/{len(VARIANTS)}", flush=True)
    print("[gran] A vs B differ ONLY in chunking; compare their confidence and argmax.", flush=True)
    print("[gran] live Jev 1.13.0 reference on the same 5 variants: run with --with-jev", flush=True)


if __name__ == "__main__":
    main()
