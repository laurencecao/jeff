"""Convert the SFT-format multiprimitive data into DecisionRow format for the
OptionScorer trainer.

SFT rows carry the prompt as chat messages and the label as text; the trainer
wants a `DecisionRow` with a real `state` and a target distribution. The state is
recovered from the ground-truth file via each row's `meta.gt_row_id` provenance.

    uv run python -m scripts.jeff_sft_to_decision
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jeff import schema as S  # noqa: E402
from jeff.model import state_to_text  # noqa: E402

PAIRS = [
    ("data/factcheck/sft_train_multi.jsonl", "data/factcheck/sft_train_multi_state.jsonl"),
    ("data/factcheck/sft_val.jsonl", "data/factcheck/sft_val_state.jsonl"),
]


def question_from_sft(row: dict) -> tuple[str, S.Question]:
    """Rebuild the question object from the SFT row's stored structure."""
    qid = row["question_id"]
    sid = row.get("schema_id", "")
    prompt = row["messages"][1]["content"].split("\n\nState:")[0].strip()
    labels = list(row["label_space"])

    if sid.startswith("no-") or row.get("meta", {}).get("kind") == "noul":
        return qid, S.NoulQuestion(instructions=prompt, criteria={
            "yes": "The evidence establishes the claim.",
            "no": "The evidence contradicts the claim, or does not establish it.",
        })
    if sid.startswith("sc-") or row.get("meta", {}).get("kind") == "score":
        # ScoreQuestion.criteria must be the level DESCRIPTIONS (ordered), not
        # the label names. Pull them back out of the original prompt text, which
        # rendered one "<index>: <description>" line per level.
        levels = re.findall(r"^(\d+):\s+(.+)$", prompt, flags=re.M)
        descriptions = [d for _, d in sorted(levels, key=lambda kv: int(kv[0]))]
        if len(descriptions) != len(labels):
            raise SystemExit(
                f"{row['row_id']}: parsed {len(descriptions)} level descriptions "
                f"for {len(labels)} labels — refusing to guess the order"
            )
        return qid, S.ScoreQuestion(instructions=prompt, criteria=descriptions)
    return qid, S.ChoiceQuestion(
        instructions=prompt, criteria=dict(S.FACTCHECK_CRITERIA)
    )


def main() -> None:
    gt = {r.row_id: r for r in S.read_rows(ROOT / "data/factcheck/ground_truth.jsonl")}

    for src, dest in PAIRS:
        sft = [json.loads(l) for l in (ROOT / src).read_text().splitlines() if l.strip()]
        out: list[S.DecisionRow] = []
        missing = 0
        for r in sft:
            gt_id = (r.get("meta") or {}).get("gt_row_id") or r["row_id"]
            source_row = gt.get(gt_id)
            if source_row is None:
                missing += 1
                continue
            qid, question = question_from_sft(r)
            out.append(
                S.DecisionRow(
                    row_id=r["row_id"],
                    source=r["source"],
                    split=r["split"],
                    group_id=r["group_id"],
                    state=source_row.state,
                    questions={qid: question},
                    labels={qid: {k: float(v) for k, v in r["target_probs"].items()}},
                    weight=r["weight"],
                    label_source=r["label_source"],
                    meta=r.get("meta", {}),
                )
            )
        n = S.write_rows(ROOT / dest, out)
        print(f"{src}: {len(sft)} SFT rows -> {n} DecisionRows ({missing} missing state)")
        if missing:
            print(f"  WARNING: {missing} rows dropped for missing state provenance")


if __name__ == "__main__":
    main()
