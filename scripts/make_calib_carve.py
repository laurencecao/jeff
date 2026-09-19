"""Create the group-disjoint calibration carve from LEGAL train states only.

Why this exists: post-hoc calibration (temperature / isotonic) must be FIT on data
that is disjoint from the rows used for training AND from every reported holdout.
`train.py` fits on the sealed val split, which is fine for the legacy smoke-test
loop but would leak here -- fitting on val and then reporting val ECE is circular.

The carve comes from ground_truth.jsonl split='train'. It is group-disjoint from
the remaining train groups, so no claim group straddles the boundary (a group
shares a claim across different evidence sets, so splitting inside a group would
leak the claim).

Writes data/factcheck/calib_carve.jsonl (the carved rows) and
data/factcheck/calib_carve_ids.json (the id list + the excluded groups), so
downstream code can EXCLUDE these rows from training rather than trust prose.

Usage: uv run python -m scripts.make_calib_carve [--frac 0.15]
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GT = ROOT / "data/factcheck/ground_truth.jsonl"
OUT = ROOT / "data/factcheck/calib_carve.jsonl"
IDS = ROOT / "data/factcheck/calib_carve_ids.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = [json.loads(l) for l in GT.read_text().splitlines() if l.strip()]
    train = [r for r in rows if r.get("split") == "train"]
    if not train:
        sys.exit("no split='train' rows found")

    by_group: dict[str, list[dict]] = collections.defaultdict(list)
    for r in train:
        by_group[r["group_id"]].append(r)

    groups = sorted(by_group)
    random.Random(args.seed).shuffle(groups)

    target = int(len(train) * args.frac)
    carved: list[dict] = []
    carve_groups: list[str] = []
    for g in groups:
        if len(carved) >= target:
            break
        carved.extend(by_group[g])
        carve_groups.append(g)

    carved_ids = [r["row_id"] for r in carved]
    rest_groups = [g for g in groups if g not in set(carve_groups)]
    overlap = set(carve_groups) & set(rest_groups)
    assert not overlap, f"carve is not group-disjoint: {overlap[:3]}"

    # refuse to carve anything that is not train-only
    bad = [r["row_id"] for r in carved if r.get("split") != "train"]
    assert not bad, f"non-train row in carve: {bad[:3]}"

    mix = collections.Counter(
        max(next(iter(r["labels"].values())).items(), key=lambda kv: kv[1])[0]
        for r in carved
    )

    OUT.write_text("".join(json.dumps(r) + "\n" for r in carved))
    IDS.write_text(json.dumps({
        "seed": args.seed,
        "frac": args.frac,
        "n_rows": len(carved),
        "n_groups": len(carve_groups),
        "carved_row_ids": carved_ids,
        "carved_group_ids": carve_groups,
        "class_mix": dict(mix),
        "source": "ground_truth.jsonl split='train'",
        "invariant": "group-disjoint from all remaining train groups; train-only",
    }, indent=2) + "\n")

    print(f"carved {len(carved)} rows across {len(carve_groups)} groups "
          f"({len(carved)/len(train):.1%} of {len(train)} train rows)")
    print(f"class mix: {dict(mix)}")
    print(f"group-disjoint from the remaining {len(rest_groups)} groups: True")
    print(f"wrote {OUT.relative_to(ROOT)} and {IDS.relative_to(ROOT)}")

    # --- ENFORCEMENT CHECK ------------------------------------------------
    # Writing ids excludes nothing by itself. Verify that filtering the ACTUAL
    # training pool by these groups removes rows and leaves it disjoint.
    sft = ROOT / "data/factcheck/sft_train_multi.jsonl"
    if sft.exists():
        pool = [json.loads(l) for l in sft.read_text().splitlines() if l.strip()]
        cg = set(carve_groups)
        removed = [r for r in pool if r.get("group_id") in cg]
        kept = [r for r in pool if r.get("group_id") not in cg]
        # a carved group can reach the SFT pool through gt_row_id too
        carve_rows = set(carved_ids)
        via_gt = [r for r in kept
                  if (r.get("meta") or {}).get("gt_row_id") in carve_rows]
        print()
        print(f"[enforce] SFT pool rows                     : {len(pool)}")
        print(f"[enforce] removed by group_id               : {len(removed)}")
        print(f"[enforce] remaining after group filter      : {len(kept)}")
        print(f"[enforce] still reaching a carve row via gt_row_id: {len(via_gt)}")
        leak = set(cg) & {r.get("group_id") for r in kept}
        print(f"[enforce] carve groups still present in kept: {len(leak)}")
        if leak or via_gt:
            print("[enforce] FAIL - exclusion is not complete")
            sys.exit(1)
        print("[enforce] PASS - filtered pool contains no carved group and no direct "
              "reference to a carved row")


if __name__ == "__main__":
    main()
