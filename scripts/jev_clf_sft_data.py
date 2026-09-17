"""Build the SFT dataset for the LM-classifier arm of jev_clf.

The frozen-MiniLM arm trained a head on a sentence-similarity encoder; this
script instead renders each DecisionRow as a chat the LM is fine-tuned on:

    system:    fixed task instruction (label names are the answer space)
    user:      question_to_text(question)  -- instructions + one line per
               label with its definition, so the label set arrives as TEXT
               at call time -- then the serialized state (claim + evidence)
    assistant: exactly the label name, nothing else

Because the user turn carries `question_to_text`, a question with a
different label set — different wording OR a different SIZE (the 2-label
noul schemas render "yes: ..." / "no: ..." lines) — is representable with
zero format changes: the label list is generated from `label_space`, never
hardcoded.

Sources:
  * data/factcheck/distill_full.jsonl   7000 train rows, target = argmax of
    Jev's soft distribution; the full distribution is kept in
    `target_probs` so soft-target training stays possible.
  * data/factcheck/ground_truth.jsonl   train rows (1589), target = the
    human label, rendered under the canonical fc-c0 wording; plus every
    ALT_EVERY-th train row duplicated under an alternative TRAIN schema
    (fc-c3..fc-c7) so the model does not overfit one phrasing. Held-out
    schemas (fc-c1, fc-c2, fc-n0, fc-n1) are never used.
  * sft_val.jsonl = ground_truth val split (199), canonical wording, for
    loss tracking only.

Usage (from the repo root):

    uv run python -m scripts.jev_clf_sft_data
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from jev_clf import gen  # noqa: E402
from jev_clf.model import state_to_text  # noqa: E402
from jev_clf.schema import (  # noqa: E402
    FACTCHECK_QUESTION_ID,
    argmax_label,
    label_space,
    normalize,
    one_hot,
    question_to_text,
    read_rows,
)

DISTILL = REPO_ROOT / "data" / "factcheck" / "distill_full.jsonl"
GROUND_TRUTH = REPO_ROOT / "data" / "factcheck" / "ground_truth.jsonl"
EVAL_SCHEMAS = REPO_ROOT / "data" / "factcheck" / "eval_schemas.jsonl"
OUT_TRAIN = REPO_ROOT / "data" / "factcheck" / "sft_train.jsonl"
OUT_VAL = REPO_ROOT / "data" / "factcheck" / "sft_val.jsonl"

# Schemas whose wording may appear in training data. The held-out schemas
# (fc-c1, fc-c2, fc-n0, fc-n1) are eval-only and are asserted absent below.
TRAIN_SCHEMAS = ("fc-c0", "fc-c3", "fc-c4", "fc-c5", "fc-c6", "fc-c7")
HELD_OUT_SCHEMAS = {"fc-c1", "fc-c2", "fc-n0", "fc-n1"}
CANONICAL_SCHEMA = "fc-c0"  # the wording embedded in ground_truth.jsonl
# Alternative train schemas for the wording-augmented ground-truth rows.
ALT_SCHEMAS = tuple(s for s in TRAIN_SCHEMAS if s != CANONICAL_SCHEMA)
ALT_EVERY = 3  # every 3rd ground-truth train row gets one extra wording

SYSTEM_PROMPT = (
    "You are a fact-checking classifier. The user gives you a question, a "
    "list of answer labels with their definitions, and a state containing a "
    "claim and evidence passages. Answer with exactly one of the listed "
    "label names and nothing else."
)

DEFAULT_TOKENIZER = "Qwen/Qwen2.5-0.5B"


# ---------------------------------------------------------------------------
# Row construction
# ---------------------------------------------------------------------------

def _user_text(question, state: Any) -> str:
    """Question (instructions + label: definition lines) then the state."""
    return f"{question_to_text(question)}\n\nState:\n{state_to_text(state)}"


def _sft_row(
    *,
    row_id: str,
    question_id: str,
    question,
    state: Any,
    label: str,
    target_probs: dict[str, float],
    split: str,
    schema_id: str,
    source: str,
    weight: float,
    label_source: str,
    group_id: str,
    meta: dict[str, Any],
) -> dict[str, Any]:
    space = label_space(question)
    assert label in space, f"{row_id}: label {label!r} not in {space}"
    assert set(target_probs) == set(space), (
        f"{row_id}: target_probs keys {sorted(target_probs)} != {space}"
    )
    return {
        "row_id": row_id,
        "question_id": question_id,
        "split": split,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _user_text(question, state)},
            {"role": "assistant", "content": label},
        ],
        "label": label,
        "label_space": space,
        "target_probs": target_probs,
        "schema_id": schema_id,
        "source": source,
        "weight": weight,
        "label_source": label_source,
        "group_id": group_id,
        "meta": meta,
    }


def _single_question(row) -> tuple[str, Any]:
    assert len(row.questions) == 1, f"{row.row_id}: {len(row.questions)} questions"
    return next(iter(row.questions.items()))


def build_train(alt_every: int = ALT_EVERY) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    # --- Jev distillation rows: keep the teacher's soft distribution ---
    for row in read_rows(DISTILL):
        assert row.split == "train", f"{row.row_id}: split {row.split}"
        qid, question = _single_question(row)
        target = normalize(row.targets(qid))
        out.append(_sft_row(
            row_id=row.row_id,
            question_id=qid,
            question=question,
            state=row.state,
            label=argmax_label(target),
            target_probs=target,
            split="train",
            schema_id=row.meta["schema_id"],
            source=row.source,
            weight=row.weight,
            label_source=row.label_source,
            group_id=row.group_id,
            meta={**row.meta, "augmented": False},
        ))

    # --- Ground-truth train rows: human label, canonical + alt wordings ---
    gt_train = [r for r in read_rows(GROUND_TRUTH) if r.split == "train"]
    for i, row in enumerate(gt_train):
        qid, question = _single_question(row)
        gt_label = argmax_label(row.targets(qid))
        out.append(_sft_row(
            row_id=row.row_id,
            question_id=qid,
            question=question,  # embedded wording == fc-c0 (asserted below)
            state=row.state,
            label=gt_label,
            target_probs=one_hot(label_space(question), gt_label),
            split="train",
            schema_id=CANONICAL_SCHEMA,
            source=row.source,
            weight=row.weight,
            label_source=row.label_source,
            group_id=row.group_id,
            meta={**row.meta, "augmented": False},
        ))
        if i % alt_every == 0:
            sid = ALT_SCHEMAS[(i // alt_every) % len(ALT_SCHEMAS)]
            alt_q = gen._build_question(gen._SCHEMA_BY_ID[sid])[qid]
            out.append(_sft_row(
                row_id=f"{row.row_id}--{sid}",
                question_id=qid,
                question=alt_q,
                state=row.state,
                label=gt_label,
                target_probs=one_hot(label_space(alt_q), gt_label),
                split="train",
                schema_id=sid,
                source=row.source,
                weight=row.weight,
                label_source=row.label_source,
                group_id=row.group_id,
                meta={**row.meta, "augmented": True,
                      "canonical_row_id": row.row_id},
            ))
    return out


def build_val() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in read_rows(GROUND_TRUTH):
        if row.split != "val":
            continue
        qid, question = _single_question(row)
        gt_label = argmax_label(row.targets(qid))
        out.append(_sft_row(
            row_id=row.row_id,
            question_id=qid,
            question=question,
            state=row.state,
            label=gt_label,
            target_probs=one_hot(label_space(question), gt_label),
            split="val",
            schema_id=CANONICAL_SCHEMA,
            source=row.source,
            weight=row.weight,
            label_source=row.label_source,
            group_id=row.group_id,
            meta={**row.meta, "augmented": False},
        ))
    return out


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _question_for_schema(schema_id: str):
    return gen._build_question(gen._SCHEMA_BY_ID[schema_id])[
        FACTCHECK_QUESTION_ID
    ]


def verify(train: list[dict], val: list[dict]) -> None:
    """Assert the leak/balance/contract invariants on the written rows."""
    eval_ids = {r.row_id for r in read_rows(EVAL_SCHEMAS)}
    eval_pairs = {
        (r.meta.get("gt_row_id"), r.meta["schema_id"])
        for r in read_rows(EVAL_SCHEMAS)
    }
    all_rows = train + val
    checks: list[str] = []

    test_ids = [r["row_id"] for r in all_rows if r["split"] == "test"]
    assert not test_ids, f"test rows in SFT data: {test_ids[:5]}"
    checks.append("no split=='test' rows")

    leaked = {r["row_id"] for r in all_rows} & eval_ids
    assert not leaked, f"eval_schemas row ids leaked: {sorted(leaked)[:5]}"
    checks.append("zero row_id overlap with eval_schemas.jsonl")

    bad_schema = [r["row_id"] for r in all_rows
                  if r["schema_id"] in HELD_OUT_SCHEMAS]
    assert not bad_schema, f"held-out schema used: {bad_schema[:5]}"
    checks.append("no held-out schema_id (fc-c1/fc-c2/fc-n0/fc-n1)")

    leaked_pairs = [
        r["row_id"] for r in all_rows
        if (r["meta"].get("canonical_row_id") or r["meta"].get("gt_row_id")
            or r["row_id"], r["schema_id"]) in eval_pairs
    ]
    assert not leaked_pairs, f"(state, schema) pairs in eval: {leaked_pairs[:5]}"
    checks.append("no (source-row, schema) pair appears in eval_schemas")

    overlap = {r["row_id"] for r in train} & {r["row_id"] for r in val}
    assert not overlap, f"train/val row_id overlap: {sorted(overlap)[:5]}"
    checks.append("train/val row_ids disjoint")

    for r in all_rows:
        q = _question_for_schema(r["schema_id"])
        space = label_space(q)
        assert r["label_space"] == space, (
            f"{r['row_id']}: label_space {r['label_space']} != {space}"
        )
        assert r["label"] in space, (
            f"{r['row_id']}: label {r['label']!r} not in {space}"
        )
        roles = [m["role"] for m in r["messages"]]
        assert roles == ["system", "user", "assistant"], (
            f"{r['row_id']}: roles {roles}"
        )
        assert r["messages"][2]["content"] == r["label"], (
            f"{r['row_id']}: assistant turn != label"
        )
    checks.append("every label in schema.label_space; assistant turn == label")

    for name, rows in (("train", train), ("val", val)):
        counts = Counter(r["label"] for r in rows)
        for label, n in counts.items():
            share = n / len(rows)
            assert 0.10 <= share <= 0.60, (
                f"{name} label {label}: share {share:.3f} outside [0.10, 0.60]"
            )
        checks.append(
            f"{name} balance: "
            + ", ".join(f"{l}={n} ({n / len(rows):.1%})"
                        for l, n in sorted(counts.items()))
        )

    for c in checks:
        print(f"  [ok] {c}")


# ---------------------------------------------------------------------------
# Token-length statistics
# ---------------------------------------------------------------------------

def token_stats(rows: list[dict], tokenizer_name: str) -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    full, prompt = [], []
    for r in rows:
        msgs = r["messages"]
        # tokenize=False -> rendered string; count ids explicitly (in
        # transformers 5.x tokenize=True returns a BatchEncoding, not ids)
        full.append(len(tok(tok.apply_chat_template(
            msgs, tokenize=False))["input_ids"]))
        prompt.append(len(tok(tok.apply_chat_template(
            msgs[:-1], tokenize=False, add_generation_prompt=True)
        )["input_ids"]))
    for name, lens in (("full conversation", full),
                       ("prompt only (sys+user)", prompt)):
        s = sorted(lens)
        n = len(s)
        q = lambda p: s[min(n - 1, int(p * n))]  # noqa: E731
        print(f"  {name}: min={s[0]} median={q(0.5)} p90={q(0.9)} "
              f"p99={q(0.99)} max={s[-1]}")
        for limit in (2048, 4096):
            over = sum(1 for x in s if x > limit) / n
            print(f"    fraction > {limit} tokens: {over:.4%}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER,
                    help="HF tokenizer for length stats")
    ap.add_argument("--alt-every", type=int, default=ALT_EVERY,
                    help="every Nth gt-train row gets an alt-wording copy")
    args = ap.parse_args()

    train, val = build_train(args.alt_every), build_val()
    for path, rows in ((OUT_TRAIN, train), (OUT_VAL, val)):
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {len(rows)} rows -> {path.relative_to(REPO_ROOT)}")

    print("\n== assertions ==")
    verify(train, val)

    print(f"\n== token lengths ({args.tokenizer}) ==")
    token_stats(train, args.tokenizer)

    print("\n== example rows (verbatim) ==")
    for r in (train[0], val[0]):
        print(f"--- {r['row_id']} [{r['schema_id']}] label={r['label']!r} ---")
        for m in r["messages"]:
            print(f"[{m['role']}]\n{m['content']}\n")

    print("== different-size label set (demonstration, not written) ==")
    noul = gen._build_question(gen._SCHEMA_BY_ID["fc-n0"])[
        FACTCHECK_QUESTION_ID]
    print("[user]\n" + _user_text(noul, {"claim": "<claim>",
                                        "evidence": [{"text": "<passage>"}]})
          + "\n[assistant]\nyes   # or no — label_space = "
          + str(label_space(noul)))


if __name__ == "__main__":
    main()
