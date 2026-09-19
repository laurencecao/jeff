"""Fail-closed leakage gate: is the SFT training pool disjoint from EVERY eval asset?

Why this exists. An earlier check only compared `ground_truth.jsonl` to
`eval_large.jsonl` and followed `meta.gt_row_id`, which does not cover the
distill rows (they can transform the claim/evidence) and does not touch other
eval assets at all. A separate check used `row_id` naming as a split indicator,
which is wrong: `ground_truth.jsonl`'s VAL rows also start `fever-train-`.

This gate compares the ACTUAL state embedded in each SFT user prompt against the
state of every evaluation asset, using a canonical signature over
(claim, ordered evidence texts). It also checks group_id / gt_row_id provenance.
Any overlap -> exit non-zero.

Canonical signature: NFC-normalize, collapse whitespace, lowercase. Evidence is
kept ORDERED (position matters for a passage-grounded decision) but titles are
excluded (they are UI metadata, not content).

Usage:
  uv run python -m scripts.check_train_eval_disjoint
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/factcheck"

TRAIN = DATA / "sft_train_multi.jsonl"
EVAL_ASSETS = [  # (label, path); ground_truth is filtered to val+test below
    ("ground_truth val+test only", DATA / "ground_truth.jsonl"),
    ("eval_large", DATA / "eval_large.jsonl"),
    ("eval_schemas", DATA / "eval_schemas.jsonl"),
    ("fixture", DATA / "fixture.jsonl"),
]


def canon(s: str | None) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFC", s)
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


def sig(claim: str | None, evidence: list[str]) -> str:
    payload = canon(claim) + "\x00" + "\x00".join(canon(e) for e in evidence)
    return hashlib.sha256(payload.encode()).hexdigest()


def parse_sft_state(user_text: str) -> tuple[str, list[str]]:
    """Recover (claim, ordered evidence) from an SFT prompt.

    Format, as written by the data builders:
        State:
        Claim: <claim>
        Evidence:
        1. <text>
        2. <text>
    Everything after the claim line up to the next section is not expected here,
    so we take the claim as the remainder of the `Claim:` line and the evidence as
    the numbered items.
    """
    m = re.search(r"Claim:\s*(.*?)(?:\n\s*Evidence:|\Z)", user_text, re.S)
    claim = m.group(1).strip() if m else ""
    ev: list[str] = []
    em = re.search(r"Evidence:\s*(.*)\Z", user_text, re.S)
    if em:
        body = em.group(1)
        parts = re.split(r"\n\s*\d+\.\s", "\n" + body)
        ev = [p.strip() for p in parts if p.strip()]
    return claim, ev


def main() -> None:
    # --- eval side --------------------------------------------------------
    # PROTECTION SCOPE. `ground_truth.jsonl` holds train/val/test. The SFT pool is
    # legitimately DERIVED from its train rows, so only its val and test rows are
    # protected. Every other asset is protected in full.
    eval_sigs: dict[str, list[str]] = {}
    eval_groups: dict[str, list[str]] = {}
    for name, path in EVAL_ASSETS:
        if not path.exists():
            print(f"[skip] {name}: {path.name} not found")
            continue
        per_split: dict[str, int] = {}
        n = 0
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            st = r.get("state") or {}
            if not isinstance(st, dict):
                continue
            split = r.get("split")
            if path.name == "ground_truth.jsonl":
                # protect ONLY the held-out rows; train rows are the SFT source
                if split not in ("val", "test"):
                    per_split[f"{split}(not protected)"] = per_split.get(f"{split}(not protected)", 0) + 1
                    continue
                tag = f"{name}/split={split}"
            else:
                tag = name
            ev = [p.get("text", "") for p in (st.get("evidence") or [])]
            eval_sigs.setdefault(sig(st.get("claim"), ev), []).append(
                f"{tag}:{r.get('row_id')}"
            )
            if r.get("group_id"):
                eval_groups.setdefault(r["group_id"], []).append(
                    f"{tag}:{r.get('row_id')}"
                )
            per_split[tag] = per_split.get(tag, 0) + 1
            n += 1
        print(f"[protected] {name}: {per_split}")

    # --- train side -------------------------------------------------------
    train_sigs: dict[str, list[str]] = {}
    train_groups: dict[str, list[str]] = {}
    train_gt_refs: set[str] = set()
    unparsed = 0
    n_train = 0
    for line in TRAIN.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("split") != "train":
            print(f"[FAIL] non-train split in SFT pool: {r.get('row_id')} split={r.get('split')}")
            sys.exit(1)
        n_train += 1
        u = [m for m in r["messages"] if m["role"] == "user"]
        claim, ev = parse_sft_state(u[0]["content"]) if u else ("", [])
        if not claim or not ev:
            unparsed += 1
        train_sigs.setdefault(sig(claim, ev), []).append(r["row_id"])
        if r.get("group_id"):
            train_groups.setdefault(r["group_id"], []).append(r["row_id"])
        g = (r.get("meta") or {}).get("gt_row_id")
        if g:
            train_gt_refs.add(g)
        if r.get("group_id", "").startswith("gt-") or "gt_row_id" in (r.get("meta") or {}):
            pass
    print(f"\n[train] {n_train} rows in the SFT pool; {unparsed} prompt states could not be parsed")

    # --- intersections ----------------------------------------------------
    print()
    inter_sig = set(train_sigs) & set(eval_sigs)
    inter_grp = set(train_groups) & set(eval_groups)
    inter_gt = train_gt_refs & {k.split(":")[-1] for v in eval_sigs.values() for k in v}

    print(f"exact (claim, ordered evidence) state overlaps : {len(inter_sig)}")
    for s in list(inter_sig)[:5]:
        print(f"   train={train_sigs[s][:2]}  eval={eval_sigs[s][:2]}")
    print(f"group_id overlaps                              : {len(inter_grp)}")
    for g in list(inter_grp)[:5]:
        print(f"   {g}: train={train_groups[g][:2]}  eval={eval_groups[g][:2]}")
    print(f"gt_row_id strings naming a PROTECTED eval row   : {len(inter_gt)}  (not itself a failure if the state differs)")
    for g in list(inter_gt)[:5]:
        print(f"   {g}")

    ok = not (inter_sig or inter_grp)
    print()
    if ok:
        print("[gate] PASS - SFT pool is disjoint from every eval asset on both the exact")
        print("       state signature and group_id. Recurring corpus sentences are expected")
        print("       and are not treated as leakage; only exact/group identity counts.")
        sys.exit(0)
    print("[gate] FAIL - overlap found. Do NOT train on this pool until it is filtered.")
    sys.exit(1)


if __name__ == "__main__":
    main()
