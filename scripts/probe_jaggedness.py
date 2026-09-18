"""Does Jeff pass the jaggedness suite that Jev scores 9/9 on?

Runs the hand-built adversarial probes (jev_clf.eval.jaggedness_suite) through
the local client and reports per-arm accuracy. Each probe's verdict is
checkable in code or by inspection, so this is a fair test of whether our
student inherits the teacher's failure modes (counting, numeric magnitude,
double negation, prompt injection, buried evidence).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jev_clf import schema as S  # noqa: E402,F401
from jev_clf.client import SystemOneClient  # noqa: E402
from jev_clf.eval import jaggedness_suite  # noqa: E402

ADAPTER = str(ROOT / "artifacts/jev_clf/lora_4b_multi")


def main() -> None:
    rows = jaggedness_suite()
    print(f"[jag] {len(rows)} probes", flush=True)

    client = SystemOneClient(adapter=ADAPTER)
    print(f"[jag] model = {client.model_id}", flush=True)

    n_ok = 0
    by_arm: dict[str, list[bool]] = {}
    for r in rows:
        qid, q = next(iter(r.questions.items()))
        gold = max(r.labels[qid].items(), key=lambda kv: kv[1])[0]
        with torch.no_grad():
            out = client.system_one(r.state, {qid: q})
        got = out.choices[qid].choice
        conf = out.choices[qid].confidence
        ok = got == gold
        n_ok += ok
        arm = r.meta.get("arm", "?")
        by_arm.setdefault(arm, []).append(ok)
        flag = "PASS" if ok else "FAIL"
        print(f"  {flag}  {r.row_id:16} arm={arm:11} gold={gold:16} got={got:16} "
              f"conf={conf:.3f}" + ("" if ok else f"   <- {r.meta.get('note','')}"))

    print(f"\n[jag] TOTAL {n_ok}/{len(rows)}")
    print("[jag] by arm:")
    for arm, oks in sorted(by_arm.items()):
        print(f"    {arm:12} {sum(oks)}/{len(oks)}")
    print("\n[jag] live Jev 1.13.0 reference: 9/9")


if __name__ == "__main__":
    main()
