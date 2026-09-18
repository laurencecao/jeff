"""Is partial-support / conjunctive failure SYSTEMATIC, and does live Jev pass it?

Motivation. On the 9,730-row scale set our entire deficit to Jev is
not_enough_info recall (+281 errors there, against -212 in our favour on
supported). One concrete failure was observed: claim "the new training program
made participants both faster and more accurate", evidence showing faster
(42s vs 55s) but accuracy TIED (both 91%), correctly "refuted" -- we answered
"supported" at 0.796.

That single case cannot distinguish "systematic defect" from "one unlucky row".
This probe builds a controlled family around it: a conjunctive claim where one
half is supported and the other is falsified or simply absent. It reports our
accuracy and -- crucially -- live Jev's on the SAME rows, because the fix only
matters if the teacher is right where we are wrong.

Arms:
  conj-falsified   A and B true-in-claim, B contradicted by evidence -> refuted
  conj-absent      A and B true-in-claim, B has no evidence at all   -> not_enough_info
  conj-both        control: both halves fully supported              -> supported
  partial-scale    claim overstates a number ("more than doubled")   -> refuted

The conj-both control matters: if we fail only the negative arms, the defect is
over-claiming. If we fail the control too, something else is wrong.

Usage:
    uv run python -m scripts.probe_conjunction              # ours only
    uv run python -m scripts.probe_conjunction --with-jev   # + live Jev (costs API calls)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jev_clf import schema as S  # noqa: E402
from jev_clf.client import SystemOneClient  # noqa: E402
from jev_clf.eval import FACTCHECK_LABELS, make_factcheck_choice  # noqa: E402

ADAPTER = str(ROOT / "artifacts/jev_clf/lora_4b_multi")

# Each entry: (row_id, arm, claim, evidence passages, gold verdict, note)
PROBES: list[tuple[str, str, str, list[dict[str, str]], str, str]] = [
    # -- conjunction, second half FALSIFIED ---------------------------------
    (
        "conj-falsified-1", "conj-falsified",
        "The new training program made participants both faster and more accurate "
        "than standard training.",
        [
            {"title": "Trial results", "text":
             "Participants using the new training program completed the task in an "
             "average of 42 seconds, compared with 55 seconds for the standard "
             "training group."},
            {"title": "Accuracy", "text":
             "Both the new-program group and the standard-training group scored "
             "exactly 91% correct on the final assessment."},
        ],
        "refuted",
        "faster (42s vs 55s) but accuracy TIED (91% both) -> 'more accurate' is false",
    ),
    (
        "conj-falsified-2", "conj-falsified",
        "The drug lowered blood pressure and reduced the rate of heart attacks.",
        [
            {"title": "Efficacy", "text":
             "Systolic blood pressure fell by 12 mmHg in the treatment arm versus "
             "2 mmHg on placebo."},
            {"title": "Cardiac events", "text":
             "Heart attack rates were 4.1% on the drug and 3.9% on placebo, a "
             "difference that was not statistically significant."},
        ],
        "refuted",
        "BP lowered (true) but heart-attack rate not reduced (3.9% vs 4.1%)",
    ),
    (
        "conj-falsified-3", "conj-falsified",
        "The candidate won the popular vote and carried the state of Ohio.",
        [
            {"title": "National count", "text":
             "With 98% of precincts reporting, the candidate led the national "
             "popular vote by 3.2 million ballots."},
            {"title": "Ohio result", "text":
             "In Ohio the candidate finished second, trailing by roughly 81,000 votes."},
        ],
        "refuted",
        "won popular vote (true) but LOST Ohio -> conjunctive claim false",
    ),
    # -- conjunction, second half has NO evidence ---------------------------
    (
        "conj-absent-1", "conj-absent",
        "The reform reduced waiting times and increased patient satisfaction.",
        [
            {"title": "Waiting times", "text":
             "Median waiting time fell from 18 weeks to 11 weeks in the two years "
             "after the reform took effect."},
        ],
        "not_enough_info",
        "waiting times reduced (supported); satisfaction is NEVER mentioned -> cannot decide the conjunction",
    ),
    (
        "conj-absent-2", "conj-absent",
        "The software upgrade improved performance and fixed the data-loss bug.",
        [
            {"title": "Benchmarks", "text":
             "Throughput rose from 1,200 to 3,400 requests per second after the "
             "upgrade was deployed."},
        ],
        "not_enough_info",
        "performance improved (supported); the data-loss bug is never mentioned",
    ),
    # -- control: both halves fully supported -------------------------------
    (
        "conj-both-1", "conj-both",
        "The new training program made participants faster and more accurate than "
        "standard training.",
        [
            {"title": "Trial results", "text":
             "Participants using the new training program completed the task in an "
             "average of 42 seconds, compared with 55 seconds for the standard "
             "training group."},
            {"title": "Accuracy", "text":
             "The new-program group scored 96% correct on the final assessment, "
             "against 91% for the standard-training group."},
        ],
        "supported",
        "CONTROL: both halves true (42s vs 55s, 96% vs 91%) -> supported",
    ),
    (
        "conj-both-2", "conj-both",
        "The candidate won the popular vote and carried the state of Ohio.",
        [
            {"title": "National count", "text":
             "The candidate led the national popular vote by 3.2 million ballots."},
            {"title": "Ohio result", "text":
             "In Ohio the candidate won by roughly 81,000 votes."},
        ],
        "supported",
        "CONTROL: both halves true -> supported",
    ),
    # -- partial magnitude: claim overstates a measured number --------------
    (
        "partial-scale-1", "partial-scale",
        "Enrollment more than doubled after the campaign.",
        [
            {"title": "Enrollment", "text":
             "Enrollment rose from 4,100 students in 2019 to 6,700 in 2022."},
        ],
        "refuted",
        "4,100 -> 6,700 is +63%, not more than double",
    ),
    (
        "partial-scale-2", "partial-scale",
        "The population of Springfield more than doubled between 2010 and 2020.",
        [
            {"title": "Springfield", "text":
             "At the 2020 census, Springfield had a population of 512,400, up from "
             "498,700 in 2010."},
        ],
        "refuted",
        "498,700 -> 512,400 is +2.7%, nowhere near doubling",
    ),
]


def build_rows():
    rows = []
    for row_id, arm, claim, evidence, gold, note in PROBES:
        rows.append(
            S.DecisionRow(
                row_id=row_id,
                source="adversarial",
                split="test",
                group_id=f"conj-{row_id}",
                state={"claim": claim, "evidence": evidence},
                questions={"verdict": make_factcheck_choice()},
                labels={"verdict": S.one_hot(list(FACTCHECK_LABELS), gold)},
                weight=1.0,
                label_source="ground_truth",
                meta={"arm": arm, "note": note},
            )
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-jev", action="store_true",
                    help="also score live Jev 1.13.0 on the same rows (uses API calls)")
    args = ap.parse_args()

    rows = build_rows()
    print(f"[conj] {len(rows)} probes\n", flush=True)

    client = SystemOneClient(adapter=ADAPTER)
    print(f"[conj] ours = {client.model_id}\n", flush=True)

    ours_ok: dict[str, list[bool]] = {}
    ours_rec: dict[str, tuple[str, str, float]] = {}
    import torch

    for r in rows:
        gold = max(r.labels["verdict"].items(), key=lambda kv: kv[1])[0]
        with torch.no_grad():
            out = client.system_one(r.state, {"verdict": make_factcheck_choice()})
        got = out.choices["verdict"].choice
        conf = out.choices["verdict"].confidence
        ok = got == gold
        ours_ok.setdefault(r.meta["arm"], []).append(ok)
        ours_rec[r.row_id] = (gold, got, conf)

    jev_rec: dict[str, tuple[str, float]] = {}
    if args.with_jev:
        from jev_clf.jev import JevTeacher
        teacher = JevTeacher()
        print("[conj] querying live Jev ...\n", flush=True)
        for r in rows:
            ans = teacher.ask(r.state, {"verdict": make_factcheck_choice()})
            dist = ans.distributions["verdict"]
            top = max(dist.items(), key=lambda kv: kv[1])
            jev_rec[r.row_id] = (top[0], top[1])
        print(f"[conj] jev network calls = {teacher.spent_requests()}\n", flush=True)

    width = 18
    hdr = f"  {'probe':{width}} {'arm':15} {'gold':16} {'ours':16} {'conf':>6}"
    if args.with_jev:
        hdr += f" {'jev':16} {'conf':>6}"
    print(hdr)
    print(f"  {'-'*width} {'-'*15} {'-'*16} {'-'*16} {'-'*6}" + (f" {'-'*16} {'-'*6}" if args.with_jev else ""))

    for r in rows:
        gold, got, conf = ours_rec[r.row_id]
        line = f"  {r.row_id:{width}} {r.meta['arm']:15} {gold:16} {got:16} {conf:>6.3f}"
        if args.with_jev:
            jg, jc = jev_rec.get(r.row_id, ("-", 0.0))
            line += f" {jg:16} {jc:>6.3f}"
        print(line)

    print(f"\n[conj] OURS by arm:")
    t_ok = t_n = 0
    for arm in sorted(ours_ok):
        oks = ours_ok[arm]
        t_ok += sum(oks); t_n += len(oks)
        print(f"    {arm:16} {sum(oks)}/{len(oks)}")
    print(f"    {'TOTAL':16} {t_ok}/{t_n}")

    if args.with_jev:
        print(f"\n[conj] LIVE JEV by arm:")
        j_ok: dict[str, list[bool]] = {}
        for r in rows:
            gold = ours_rec[r.row_id][0]
            jg = jev_rec[r.row_id][0]
            j_ok.setdefault(r.meta["arm"], []).append(jg == gold)
        jt_ok = jt_n = 0
        for arm in sorted(j_ok):
            oks = j_ok[arm]
            jt_ok += sum(oks); jt_n += len(oks)
            print(f"    {arm:16} {sum(oks)}/{len(oks)}")
        print(f"    {'TOTAL':16} {jt_ok}/{jt_n}")
        print(f"\n[conj] HEAD TO HEAD: ours {t_ok}/{t_n}   jev {jt_ok}/{jt_n}")

    dest = ROOT / "results" / "probe_conjunction.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(
        {
            "ours": {k: {"gold": v[0], "got": v[1], "confidence": v[2]} for k, v in ours_rec.items()},
            "jev": {k: {"got": v[0], "confidence": v[1]} for k, v in jev_rec.items()},
            "ours_by_arm": {k: [sum(v), len(v)] for k, v in ours_ok.items()},
        },
        indent=2,
    ))
    print(f"\n[conj] wrote {dest}")


if __name__ == "__main__":
    main()
