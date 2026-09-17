"""Alignment test: the LoRA training prompt must match the eval prompt exactly.

If they drift, the adapter is evaluated on a prompt format it never trained on
and every reported number is garbage. This reconstructs the eval prompt from the
source ground-truth row and compares it against what the training data actually
contains, instead of trusting that two functions render the same thing.

    uv run python -m scripts.test_prompt_alignment
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jev_clf import schema as S  # noqa: E402
from jev_clf.model import state_to_text  # noqa: E402
from scripts.jev_clf_lm_eval import SYSTEM, build_inputs, label_variants  # noqa: E402
from scripts.jev_clf_lora_train import build_example  # noqa: E402

TOKENIZER = "Qwen/Qwen2.5-1.5B-Instruct"
CFG = {
    "max_length": 2048,
    "data": {"prompt_format": "eval", "completion_leading_space": True},
}


def eval_user_turn(row: S.DecisionRow) -> str:
    qid, question = next(iter(row.questions.items()))
    return (
        S.question_to_text(question)
        + "\n\nState:\n"
        + state_to_text(row.state)
        + "\n\nVerdict:"
    )


def main() -> None:
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    sft = [json.loads(l) for l in (ROOT / "data/factcheck/sft_train.jsonl").read_text().splitlines() if l.strip()]
    gt = {r.row_id: r for r in S.read_rows(ROOT / "data/factcheck/ground_truth.jsonl")}

    # sample short, medium and the longest rows — drift hides in the long ones
    sft.sort(key=lambda r: -len(r["messages"][1]["content"]))
    sample = sft[:3] + sft[len(sft) // 2 : len(sft) // 2 + 3] + sft[-3:]

    failures = 0
    checked_user_turn = 0
    for row in sample:
        gt_id = (row.get("meta") or {}).get("gt_row_id") or row["row_id"]
        src = gt.get(gt_id)

        ex = build_example(tok, row, CFG)
        labels = ex["labels"]
        sup = [i for i, x in enumerate(labels) if x != -100]
        completion = ex["input_ids"][sup[0] : sup[0] + len(sup)]
        label = row["label"]
        variants = label_variants(tok, [label])[label]
        ok_comp = completion[: len(variants)] == variants
        print(f"{row['row_id'][:32]:32s} sup={len(sup):2d} label={label:17s} "
              f"completion={tok.decode(completion)!r:22s} first-token-ok={ok_comp}")
        if not ok_comp:
            failures += 1

        # Compare the RENDERED training prompt against the RENDERED eval prompt
        # by token id. Only fc-c0 rows have the canonical wording that the eval
        # harness uses; fc-c3..c7 carry deliberately different instruction
        # wordings (that variation is the text-conditioning training signal).
        prompt_len = len(ex["input_ids"]) - len(sup)
        train_prompt_ids = ex["input_ids"][:prompt_len]
        rendered_txt = tok.decode(train_prompt_ids)
        tail = rendered_txt[-80:]
        print(f"    rendered prompt tail: {tail!r}")
        if "assistant" not in tail:
            failures += 1
            print("    FAIL: rendered prompt does not end with the assistant cue")

        if src is not None and row.get("schema_id") == "fc-c0":
            qid = next(iter(src.questions))
            eval_text = build_inputs(tok, src.state, src.questions[qid])
            eval_ids = tok(eval_text, add_special_tokens=False)["input_ids"]
            same = eval_ids == train_prompt_ids
            checked_user_turn += 1
            print(f"    RENDERED eval-vs-train prompt ids identical: {same} "
                  f"({len(eval_ids)} vs {len(train_prompt_ids)})")
            if not same:
                failures += 1
                for i, (a, b) in enumerate(zip(eval_ids, train_prompt_ids)):
                    if a != b:
                        print(f"      first divergence at token {i}: {a} vs {b} "
                              f"({tok.decode([a])!r} vs {tok.decode([b])!r})")
                        break

        # full chat template comparison, which is what the model actually sees
        msgs = [{"role": "system", "content": SYSTEM}] + row["messages"][1:2]
        rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        print(f"    chat prompt matches stored messages: {rendered.startswith(tok.apply_chat_template([{'role':'system','content':SYSTEM}], tokenize=False, add_generation_prompt=False).rstrip())}")

    print(f"\nsampled={len(sample)} user-turn comparisons={checked_user_turn} FAILURES={failures}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
