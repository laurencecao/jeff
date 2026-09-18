"""Regression gate: routing the Choice path through jev_clf/readout.py must not
change a single prediction.

Compares, on val rows, the probabilities the CLIENT produces (which now goes
through readout.distribution) against the stored predictions the BENCHMARK
harness produced with its own inline first-token readout
(data/factcheck/preds_autoresearch_val.jsonl, adapter lora_4b_multi).

Both used the same adapter, so any difference is caused by the refactor.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jev_clf import schema as S  # noqa: E402
from jev_clf.client import SystemOneClient  # noqa: E402

N_ROWS = 12
ADAPTER = str(ROOT / "artifacts/jev_clf/lora_4b_multi")
REF = ROOT / "data/factcheck/preds_autoresearch_val.jsonl"


def main() -> None:
    stored: dict[tuple[str, str], dict[str, float]] = {}
    for line in REF.read_text().splitlines():
        d = json.loads(line)
        stored[(d["row_id"], d["question_id"])] = d["probs"]

    rows = [r for r in S.read_rows(ROOT / "data/factcheck/ground_truth.jsonl") if r.split == "val"]
    rows = rows[:N_ROWS]

    print(f"[gate] loading client with {Path(ADAPTER).name}", flush=True)
    client = SystemOneClient(adapter=ADAPTER)
    print(f"[gate] client.model_id = {client.model_id}", flush=True)

    worst = 0.0
    n_checked = 0
    n_mismatch = 0
    for r in rows:
        qid, q = next(iter(r.questions.items()))
        ref = stored.get((r.row_id, qid))
        if ref is None:
            continue
        with torch.no_grad():
            out = client.system_one(r.state, {qid: q})
        got = out.choices[qid].probabilities
        mode = out.readout_modes[qid]

        diff = max(abs(got[k] - ref[k]) for k in ref)
        worst = max(worst, diff)
        n_checked += 1
        top_got = max(got.items(), key=lambda kv: kv[1])[0]
        top_ref = max(ref.items(), key=lambda kv: kv[1])[0]
        if top_got != top_ref:
            n_mismatch += 1
        print(f"  {r.row_id[:34]:36} mode={mode:11} argmax {top_got:16} "
              f"(ref {top_ref:16}) maxdelta={diff:.2e}")

    print(f"\n[gate] rows checked      = {n_checked}")
    print(f"[gate] argmax mismatches = {n_mismatch}")
    print(f"[gate] worst prob delta  = {worst:.3e}")
    print(f"[gate] VERDICT: {'PASS - choice path unchanged' if n_mismatch == 0 and worst < 1e-6 else 'FAIL'}")


if __name__ == "__main__":
    main()
