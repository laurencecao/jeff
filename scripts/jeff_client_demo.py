"""Prove the drop-in surface, and measure the one architectural gap.

Three things:
  1. a mixed multi-primitive call (Choice + Noul + Score x2) on one state,
     printed in Jev's response shape;
  2. the SAME questions asked of live Jev, for a like-for-like comparison —
     including the Score primitive, which our model was never trained on;
  3. latency as the number of questions per state grows, ours vs Jev's. Jev
     advertises flat latency in question count; an autoregressive readout cannot
     match that, and this quantifies the difference rather than asserting it.

    uv run python -m scripts.jeff_client_demo
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jeff import schema as S  # noqa: E402
from jeff.client import Choice, Noul, Score, SystemOneClient  # noqa: E402
from jeff.jev import JevTeacher  # noqa: E402

N_STATES = 3
LATENCY_KS = (1, 2, 4, 8)
LATENCY_REPS = 3


def questions_for(state, n: int) -> dict:
    """A mixed set of n questions over one state: verdict first, then extras."""
    qs: dict = {
        "verdict": Choice(
            instructions=(
                "Read the claim and the evidence passages in `state`. Decide "
                "whether the evidence establishes the claim, contradicts it, or "
                "cannot decide it."
            ),
            criteria=dict(S.FACTCHECK_CRITERIA),
        )
    }
    extras = [
        Noul(instructions="Does the evidence mention any organisation or institution?"),
        Score(
            instructions="How much evidence is provided for the claim?",
            criteria=[
                "no evidence at all",
                "a single weak or unrelated passage",
                "one relevant passage",
                "several relevant passages",
            ],
        ),
        Noul(instructions="Is the claim about a scientific or technical subject?"),
        Noul(instructions="Would a careful reader describe this claim as controversial?"),
        Score(
            instructions="How clearly does the evidence state the claim's subject?",
            criteria=["not at all", "vaguely", "clearly", "explicitly named"],
        ),
        Noul(instructions="Does the evidence contain a date or a number?"),
        Noul(instructions="Is the evidence longer than the claim itself?"),
    ]
    for q in extras[: n - 1]:
        qs[f"q{len(qs)}"] = q
    return qs


def main() -> None:
    states = [r.state for r in S.read_rows(ROOT / "data/factcheck/ground_truth.jsonl") if r.split == "val"][:N_STATES]

    print("=== loading our drop-in client ===")
    t0 = time.perf_counter()
    client = SystemOneClient()
    print(f"  model={client.model_id} device={client.device} load={time.perf_counter()-t0:.1f}s")

    print("\n=== 1. mixed multi-primitive call, our client ===")
    for state in states[:1]:
        qs = {
            "verdict": questions_for(state, 1)["verdict"],
            "has_org": Noul(instructions="Does the evidence mention any organisation or institution?"),
            "evidence_strength": Score(
                instructions="How much evidence is provided for the claim?",
                criteria=[
                    "no evidence at all",
                    "a single weak or unrelated passage",
                    "one relevant passage",
                    "several relevant passages",
                ],
            ),
        }
        res = client.system_one(state, qs)
        c = res.choices["verdict"]
        print(f"  choice verdict = {c.choice}  confidence={c.confidence:.3f}")
        print(f"    probabilities = {{{', '.join(f'{k}: {v:.3f}' for k, v in c.probabilities.items())}}}")
        print(f"  noul has_org = {res.nouls['has_org'].noul:.3f}   (no confidence field, as Jev)")
        s = res.scores["evidence_strength"]
        print(f"  score evidence_strength = {s.score:.3f}  confidence={s.confidence:.3f}")
        print(f"    probabilities = {{{', '.join(f'{k}: {v:.3f}' for k, v in s.probabilities.items())}}}")
        print(f"  forward passes = {res.n_forward_passes} (one per question)")

    print("\n=== 2. same questions, live Jev (note: Score is untrained for us) ===")
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        print("  (TYPESAFE_API_KEY unset — skipping the Jev side)")
    else:
        teacher = JevTeacher(api_key=key)
        state = states[0]
        qs = {
            "verdict": questions_for(state, 1)["verdict"],
            "has_org": Noul(instructions="Does the evidence mention any organisation or institution?"),
            "evidence_strength": Score(
                instructions="How much evidence is provided for the claim?",
                criteria=[
                    "no evidence at all",
                    "a single weak or unrelated passage",
                    "one relevant passage",
                    "several relevant passages",
                ],
            ),
        }
        a = teacher.ask(state, qs)
        ours = client.system_one(state, qs)
        print(f"  verdict   jev={max(a.distributions['verdict'], key=a.distributions['verdict'].get)}"
              f"  ours={ours.choices['verdict'].choice}")
        print(f"  noul      jev={a.distributions['has_org'].get('yes', float('nan')):.3f}"
              f"  ours={ours.nouls['has_org'].noul:.3f}")
        jd = a.distributions["evidence_strength"]
        jscore = sum(int(k) * v for k, v in jd.items())
        print(f"  score     jev={jscore:.3f}  ours={ours.scores['evidence_strength'].score:.3f}"
              "   <- ours is untrained on Score")

    print("\n=== 3. latency vs number of questions per state (the architectural gap) ===")
    teacher = JevTeacher(api_key=key) if key else None
    print(f"  {'k':>3}  {'ours ms':>9}  {'ours x':>7}  {'jev ms':>9}  {'jev x':>7}")
    ours_base = None
    jev_base = None
    rows = []
    for k in LATENCY_KS:
        ours_lat, jev_lat = [], []
        for rep in range(LATENCY_REPS):
            qs = questions_for(states[0], k)
            t = time.perf_counter()
            client.system_one(states[0], qs)
            ours_lat.append((time.perf_counter() - t) * 1000)
            if teacher is not None:
                # unique wording per rep so the teacher cache cannot serve these
                qs = {qid: q for qid, q in questions_for(states[0], k).items()}
                qs[f"rep{rep}"] = Noul(instructions=f"Probe k={k} rep={rep}: is the claim plausible?")
                t = time.perf_counter()
                teacher.ask(states[0], qs)
                jev_lat.append((time.perf_counter() - t) * 1000)
        o = statistics.median(ours_lat)
        j = statistics.median(jev_lat) if jev_lat else float("nan")
        ours_base = ours_base or o
        jev_base = jev_base or j
        print(f"  {k:>3}  {o:>9.1f}  {o/ours_base:>6.2f}x  {j:>9.1f}  {j/jev_base:>6.2f}x")
        rows.append({"k": k, "ours_ms": o, "ours_x": o / ours_base,
                     "jev_ms": j, "jev_x": j / jev_base if jev_lat else None})

    dest = ROOT / "results" / "client_demo_latency.json"
    dest.write_text(json.dumps({"states": N_STATES, "rows": rows}, indent=2) + "\n")
    print("\nwrote", dest)


if __name__ == "__main__":
    main()
