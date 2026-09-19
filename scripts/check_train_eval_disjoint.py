"""Fail-closed leakage gate: is the SFT training pool disjoint from EVERY eval asset?

Why this exists. An earlier check only compared `ground_truth.jsonl` to
`eval_large.jsonl` and followed `meta.gt_row_id`, which does not cover the
distill rows (they can transform the claim/evidence) and does not touch other
eval assets at all. A separate check used `row_id` naming as a split indicator,
which is wrong: `ground_truth.jsonl`'s VAL rows also start `fever-train-`.

This gate compares the ACTUAL state each row presents to the model. Both sides
go through the SAME rendering and parsing pipeline:

  * eval rows:  state dict -> jev_clf.model.state_to_text -> parse_state_text
  * train rows: user prompt -> strip the "State:" header -> parse_state_text

so a state that renders identically on both sides cannot slip through an
asymmetric signature (an earlier version hashed eval `text` only while the SFT
prompt renders `title: text`, so identical states never intersected).

The gate FAILS CLOSED: a missing/empty eval asset, an eval asset with no
protected rows, an empty or non-train SFT pool, an unparseable prompt or
state, a malformed id/group, an unknown ground_truth split, or any overlap
all exit non-zero. PASS means every check ran and found nothing.

Leakage signatures (any hit fails):
  1. exact state: same claim + same evidence lines in the same order — the
     rendered model input is identical.
  2. evidence-set: same claim + the same full evidence set in ANY order —
     reordering must not bypass isolation. This is a superset of (1) and is
     reported separately.
  3. group_id shared between the pool and a protected row.
  4. meta.gt_row_id that names a protected row (unambiguous: ground_truth
     row_ids do not collide with other assets' row_ids).

NOT leakage: reused corpus snippets, shared row_id naming, or a different
claim over overlapping evidence — those are expected and allowed.

Canonical signature: NFC-normalize, collapse whitespace, lowercase. Evidence
lines are compared as rendered (`title: text`), because that is what the model
reads. An explicitly empty evidence list renders as `(none)` and parses to an
empty list — a legitimate state, not a parse failure.

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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jev_clf.model import state_to_text  # noqa: E402

DATA = ROOT / "data/factcheck"

TRAIN = DATA / "sft_train_multi.jsonl"

# The one eval asset whose train rows are the legitimate SFT source: only its
# val/test rows are protected. Every other asset is protected in full.
TRAIN_SOURCE_NAME = "ground_truth.jsonl"
GT_SPLITS = ("train", "val", "test")

EVAL_ASSETS = [  # (label, path)
    ("ground_truth val+test only", DATA / "ground_truth.jsonl"),
    ("eval_large", DATA / "eval_large.jsonl"),
    ("eval_schemas", DATA / "eval_schemas.jsonl"),
    ("fixture", DATA / "fixture.jsonl"),
]


def canon(s: str | None) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFC", s)
    s = s.replace("’", "'").replace("‘", "'")
    s = s.replace("“", '"').replace("”", '"')
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


def sig(claim: str | None, evidence: list[str]) -> str:
    """Ordered signature: identical claim + identical evidence lines in order."""
    payload = canon(claim) + "\x00" + "\x00".join(canon(e) for e in evidence)
    return hashlib.sha256(payload.encode()).hexdigest()


def set_sig(claim: str | None, evidence: list[str]) -> str:
    """Order-insensitive signature: same claim + same full evidence multiset."""
    payload = canon(claim) + "\x00" + "\x00".join(sorted(canon(e) for e in evidence))
    return hashlib.sha256(payload.encode()).hexdigest()


_STATE_HEADER = re.compile(r"(?:^|\n)[ \t]*State:[ \t]*\n")
_CLAIM_THEN_EVIDENCE = re.compile(r"\s*Claim:[ \t]*(.*?)\n[ \t]*Evidence:[ \t]*\n?", re.S)
_EVIDENCE_ITEM = re.compile(r"\n[ \t]*\d+\.[ \t]+")
_EMPTY_MARKER = "(none)"  # what state_to_text emits for an empty evidence list


def parse_state_text(text: str) -> tuple[str, list[str]] | None:
    """Parse a ``state_to_text`` rendering into (claim, rendered evidence lines).

    Returns None when the text does not follow the convention — no leading
    ``Claim:`` line, no ``Evidence:`` marker, or an empty claim. An explicit
    empty evidence list renders as ``(none)`` and parses to ``[]``: that is a
    legitimate empty list, NOT a parse failure.
    """
    m = _CLAIM_THEN_EVIDENCE.match(text)
    if not m:
        return None
    claim = m.group(1).strip()
    if not claim:
        return None
    body = text[m.end():]
    parts = _EVIDENCE_ITEM.split("\n" + body)
    ev = [p.strip() for p in parts if p.strip()]
    if ev == [_EMPTY_MARKER]:
        ev = []
    return claim, ev


def parse_sft_state(user_text: str) -> tuple[str, list[str]] | None:
    """Recover (claim, rendered evidence) from an SFT user prompt.

    The builders emit ``<question text>\\n\\nState:\\n`` + ``state_to_text(state)``,
    so the state is everything after the ``State:`` header, parsed with the same
    parser used for eval-side renderings.
    """
    m = _STATE_HEADER.search(user_text)
    if not m:
        return None
    return parse_state_text(user_text[m.end():])


def _eval_state_signature(state) -> tuple[str, list[str]] | None:
    """Fingerprint a protected eval row's state through the SAME pipeline the
    train side is checked with: render with state_to_text, then parse."""
    if not isinstance(state, dict) or not isinstance(state.get("claim"), str):
        return None
    ev = state.get("evidence")
    if ev is not None and not isinstance(ev, list):
        return None
    return parse_state_text(state_to_text(state))


def run_gate(train_path=TRAIN, eval_assets=EVAL_ASSETS) -> int:
    """Run the leakage gate. Prints the report; returns 0 on PASS, 1 on FAIL."""
    failures: list[str] = []

    def fail(msg: str) -> None:
        failures.append(msg)
        print(f"[FAIL] {msg}")

    # --- eval side --------------------------------------------------------
    eval_sigs: dict[str, list[str]] = {}
    eval_set_sigs: dict[str, list[str]] = {}
    eval_groups: dict[str, list[str]] = {}
    protected_ids: set[str] = set()
    for name, path in eval_assets:
        path = Path(path)
        if not path.exists():
            fail(f"{name}: {path} not found")
            continue
        lines = [l for l in path.read_text().splitlines() if l.strip()]
        if not lines:
            fail(f"{name}: {path.name} is empty — no protected rows")
            continue
        is_gt = path.name == TRAIN_SOURCE_NAME
        per_split: dict[str, int] = {}
        n = 0
        for line in lines:
            r = json.loads(line)
            rid = r.get("row_id")
            if not isinstance(rid, str) or not rid:
                fail(f"{name}: row with missing/empty row_id")
                continue
            split = r.get("split")
            if is_gt:
                # protect ONLY the held-out rows; train rows are the SFT source
                if split not in GT_SPLITS:
                    fail(f"{name}: {rid} has unknown/missing split {split!r}")
                    continue
                if split == "train":
                    k = "train(not protected)"
                    per_split[k] = per_split.get(k, 0) + 1
                    continue
                tag = f"{name}/split={split}"
            else:
                tag = name
            protected_ids.add(rid)
            parsed = _eval_state_signature(r.get("state"))
            if parsed is None:
                fail(f"{name}: {rid} protected state does not match the claim/evidence schema")
                continue
            claim, ev = parsed
            eval_sigs.setdefault(sig(claim, ev), []).append(f"{tag}:{rid}")
            eval_set_sigs.setdefault(set_sig(claim, ev), []).append(f"{tag}:{rid}")
            gid = r.get("group_id")
            if gid is not None:
                if not isinstance(gid, str) or not gid:
                    fail(f"{name}: {rid} has empty group_id")
                else:
                    eval_groups.setdefault(gid, []).append(f"{tag}:{rid}")
            per_split[tag] = per_split.get(tag, 0) + 1
            n += 1
        print(f"[protected] {name}: {per_split}")
        if n == 0:
            fail(f"{name}: 0 protected rows — nothing verified")

    # --- train side -------------------------------------------------------
    train_path = Path(train_path)
    train_sigs: dict[str, list[str]] = {}
    train_set_sigs: dict[str, list[str]] = {}
    train_groups: dict[str, list[str]] = {}
    train_gt_refs: set[str] = set()
    unparsed = 0
    n_train = 0
    if not train_path.exists():
        fail(f"SFT pool {train_path} not found")
    else:
        for line in train_path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            rid = r.get("row_id")
            if not isinstance(rid, str) or not rid:
                fail("train row with missing/empty row_id")
                continue
            if r.get("split") != "train":
                fail(f"non-train split in SFT pool: {rid} split={r.get('split')!r}")
            n_train += 1
            users = [m for m in (r.get("messages") or []) if m.get("role") == "user"]
            parsed = parse_sft_state(users[0].get("content") or "") if users else None
            if parsed is None:
                unparsed += 1
                fail(f"unparseable SFT prompt state: {rid}")
                continue
            claim, ev = parsed
            train_sigs.setdefault(sig(claim, ev), []).append(rid)
            train_set_sigs.setdefault(set_sig(claim, ev), []).append(rid)
            gid = r.get("group_id")
            if gid is not None:
                if not isinstance(gid, str) or not gid:
                    fail(f"train {rid}: empty group_id")
                else:
                    train_groups.setdefault(gid, []).append(rid)
            g = (r.get("meta") or {}).get("gt_row_id")
            if g is not None:
                if not isinstance(g, str) or not g:
                    fail(f"train {rid}: empty meta.gt_row_id")
                else:
                    train_gt_refs.add(g)
        print(f"\n[train] {n_train} rows in the SFT pool; {unparsed} prompt states could not be parsed")
        if n_train == 0:
            fail("SFT pool is empty — nothing to check")

    # --- intersections ----------------------------------------------------
    print()
    inter_sig = set(train_sigs) & set(eval_sigs)
    inter_set = set(train_set_sigs) & set(eval_set_sigs)
    inter_grp = set(train_groups) & set(eval_groups)
    inter_gt = train_gt_refs & protected_ids

    print(f"exact state overlaps (identical rendered model input)      : {len(inter_sig)}")
    for s in list(inter_sig)[:5]:
        print(f"   train={train_sigs[s][:2]}  eval={eval_sigs[s][:2]}")
    print(f"evidence-set overlaps (same claim + same evidence, any order): {len(inter_set)}")
    for s in list(inter_set)[:5]:
        print(f"   train={train_set_sigs[s][:2]}  eval={eval_set_sigs[s][:2]}")
    print(f"group_id overlaps                                          : {len(inter_grp)}")
    for g in list(inter_grp)[:5]:
        print(f"   {g}: train={train_groups[g][:2]}  eval={eval_groups[g][:2]}")
    print(f"gt_row_id strings naming a PROTECTED eval row              : {len(inter_gt)}")
    for g in list(inter_gt)[:5]:
        print(f"   {g}")
    n_gt_ok = len(train_gt_refs - protected_ids)
    print(f"gt_row_id refs to unprotected (train-split) source rows    : {n_gt_ok}  (expected provenance)")

    for s in inter_sig:
        fail(f"exact state overlap: train={train_sigs[s][:2]} eval={eval_sigs[s][:2]}")
    for s in inter_set - inter_sig:
        fail(f"reordered evidence-set overlap: train={train_set_sigs[s][:2]} eval={eval_set_sigs[s][:2]}")
    for g in inter_grp:
        fail(f"group_id overlap: {g} train={train_groups[g][:2]} eval={eval_groups[g][:2]}")
    for g in inter_gt:
        fail(f"meta.gt_row_id names a protected eval row: {g}")

    print()
    if not failures:
        print("[gate] PASS - SFT pool is disjoint from every eval asset on the exact state,")
        print("       order-insensitive evidence-set, group_id, and gt_row_id signatures.")
        print("       Recurring corpus sentences are expected and are not treated as leakage.")
        return 0
    print(f"[gate] FAIL - {len(failures)} problem(s). Do NOT train on this pool until resolved.")
    return 1


def main() -> None:
    sys.exit(run_gate())


if __name__ == "__main__":
    main()
