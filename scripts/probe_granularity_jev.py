"""Live Jev 1.13.0 on the passage-granularity variants, for a stability check.

Our own run on these 5 variants (scripts/probe_passage_granularity.py) went
0/5, with confidence swinging 0.506 -> 0.873 across semantically near-identical
inputs. This asks whether the teacher is stable where we are not.

Read-only w.r.t. the repo: it prints a comparison table and writes nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jev_clf.client import SystemOneClient  # noqa: E402
from jev_clf.eval import make_factcheck_choice  # noqa: E402
from jev_clf.jev import JevTeacher  # noqa: E402
from scripts.probe_passage_granularity import CLAIM, VARIANTS  # noqa: E402

ADAPTER = str(ROOT / "artifacts/jev_clf/lora_4b_multi")


def main() -> None:
    ours = SystemOneClient(adapter=ADAPTER)
    jev = JevTeacher()
    print(f"[gran] ours = {ours.model_id}", flush=True)
    print(f"[gran] jev  = {jev.model}\n", flush=True)

    import torch

    print(f"  {'variant':54} {'gold':16} {'ours':16} {'c':>5} {'jev':16} {'c':>5}", flush=True)
    print(f"  {'-'*54} {'-'*16} {'-'*16} {'-'*5} {'-'*16} {'-'*5}", flush=True)

    o_ok = j_ok = 0
    for name, evidence, gold in VARIANTS:
        state = {"claim": CLAIM, "evidence": evidence}
        q = {"verdict": make_factcheck_choice()}
        with torch.no_grad():
            a = ours.system_one(state, q).choices["verdict"]
        jd = jev.ask(state, q).distributions["verdict"]
        jtop = max(jd.items(), key=lambda kv: kv[1])
        o_ok += a.choice == gold
        j_ok += jtop[0] == gold
        print(f"  {name:54} {gold:16} {a.choice:16} {a.confidence:>5.3f} "
              f"{jtop[0]:16} {jtop[1]:>5.3f}", flush=True)

    print(f"\n[gran] ours {o_ok}/{len(VARIANTS)}   live Jev {j_ok}/{len(VARIANTS)}", flush=True)
    print(f"[gran] jev network calls = {jev.spent_requests()}", flush=True)


if __name__ == "__main__":
    main()
