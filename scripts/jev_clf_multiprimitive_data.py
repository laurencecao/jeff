"""Generate Noul and Score training data so the model covers all three primitives.

WHY: all six training schemas are Choice. The Noul schemas (fc-n0/fc-n1) were
held out entirely to measure agreement-with-Jev on unseen wordings, and Score was
never generated. The measured consequence is that our Score output is uniform
(0.25 on each level) — no signal at all — while Jev returns a real graded answer.

This builds teacher-labelled Noul and Score examples over the SAME ground-truth
TRAIN states, using NEW schema wordings that do not collide with the held-out
agreement schemas, so that adding them cannot leak into any evaluation.

    uv run python -m scripts.jev_clf_multiprimitive_data --n-states 600
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jev_clf import schema as S  # noqa: E402
from jev_clf.jev import JevTeacher  # noqa: E402
from jev_clf.model import state_to_text  # noqa: E402
from scripts.jev_clf_lm_eval import SYSTEM  # noqa: E402

OUT = ROOT / "data/factcheck/sft_multi.jsonl"

# Held out for the agreement metric — training data must never use these.
HELD_OUT_SCHEMAS = {"fc-c1", "fc-c2", "fc-n0", "fc-n1"}

# NEW wordings, distinct from both the training Choice schemas and the held-out
# ones. Each is a (id, kind, instructions, criteria) tuple.
NOUL_SCHEMAS = [
    (
        "no-a0",
        "Does the evidence presented establish that the claim is true?",
        {
            "yes": "The evidence establishes the claim.",
            "no": "The evidence contradicts the claim, or does not establish it.",
        },
    ),
    (
        "no-a1",
        "Taken together, do the passages in `state` support the claim?",
        {
            "yes": "The passages support the claim.",
            "no": "The passages do not support the claim.",
        },
    ),
]

SCORE_SCHEMAS = [
    (
        "sc-a0",
        "How much evidence does `state` provide for the claim?",
        [
            "no evidence at all",
            "a single weak or unrelated passage",
            "one relevant passage",
            "several relevant passages",
        ],
    ),
    (
        "sc-a1",
        "How directly does the evidence address the claim's subject?",
        [
            "not at all",
            "only tangentially",
            "broadly related",
            "directly on point",
        ],
    ),
    (
        "sc-a2",
        "How much interpretation would a reader need to connect the evidence to the claim?",
        [
            "a great deal of interpretation",
            "substantial interpretation",
            "a little interpretation",
            "none, the connection is explicit",
        ],
    ),
]


def make_questions(rng: random.Random) -> list[tuple[str, S.Question]]:
    out: list[tuple[str, S.Question]] = []
    for sid, instr, criteria in NOUL_SCHEMAS:
        out.append((sid, S.NoulQuestion(instructions=instr, criteria=criteria)))
    for sid, instr, criteria in SCORE_SCHEMAS:
        out.append((sid, S.ScoreQuestion(instructions=instr, criteria=list(criteria))))
    rng.shuffle(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-states", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--budget-usd", type=float, default=0.50)
    args = ap.parse_args()

    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise SystemExit("TYPESAFE_API_KEY not set")

    rows = [r for r in S.read_rows(ROOT / "data/factcheck/ground_truth.jsonl") if r.split == "train"]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    states = rows[: args.n_states]
    print(f"states: {len(states)} (from {len(rows)} ground-truth train rows)")

    teacher = JevTeacher(api_key=key)
    out: list[dict] = []
    planned = 0

    for i, row in enumerate(states):
        for sid, question in make_questions(rng):
            if sid in HELD_OUT_SCHEMAS:
                raise SystemExit(f"refusing to train on a held-out schema: {sid}")
            planned += 1
            answer = teacher.ask(row.state, {sid: question})
            dist = answer.distributions[sid]
            labels = S.label_space(question)
            assert set(dist) == set(labels), (dist, labels)
            label = max(dist.items(), key=lambda kv: kv[1])[0]

            user = (
                S.question_to_text(question)
                + "\n\nState:\n"
                + state_to_text(row.state)
            )
            out.append(
                {
                    "row_id": f"{row.row_id}--{sid}",
                    "question_id": sid,
                    "split": "train",
                    "messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": label},
                    ],
                    "label": label,
                    "label_space": labels,
                    "target_probs": dist,
                    "schema_id": sid,
                    "source": "multiprimitive",
                    "weight": 1.0,
                    "label_source": answer.model,
                    "group_id": row.group_id,
                    "meta": {
                        "gt_row_id": row.row_id,
                        "kind": question.kind,
                        "teacher_confidence": answer.confidence.get(sid),
                    },
                }
            )
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(states)} states, {len(out)} rows, "
                  f"{teacher.spent_requests()} requests")

    assert not any(r["split"] == "test" for r in out), "test rows must never appear"
    assert not ({r["schema_id"] for r in out} & HELD_OUT_SCHEMAS), "held-out schema leaked"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")

    kinds: dict[str, int] = {}
    labels: dict[str, int] = {}
    for r in out:
        kinds[r["meta"]["kind"]] = kinds.get(r["meta"]["kind"], 0) + 1
        labels[r["label"]] = labels.get(r["label"], 0) + 1
    print(f"\nwrote {OUT} rows={len(out)} planned={planned}")
    print("  by kind  :", kinds)
    print("  by label :", labels)
    req = teacher.spent_requests()
    est = req * 450 * 42 / 1e9
    print(f"  requests : {req} (~${est:.4f} est.)")
    if est > args.budget_usd:
        raise SystemExit(f"over budget: ~${est:.4f} > ${args.budget_usd}")


if __name__ == "__main__":
    main()
