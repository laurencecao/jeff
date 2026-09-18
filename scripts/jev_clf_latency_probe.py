"""Does the option-scorer architecture give flat latency in question count?

Jev's claim is that adding questions barely changes response time, because every
option is scored as a query against ONE encoded state. Our autoregressive
readout cannot do that (one full pass per question), which measured 6.39x at
k=8 while Jev was 0.78x.

The OptionScorer encodes the state once and scores every option as a query, so
it SHOULD be flat. This measures it before we invest in retraining it with a
real language-model encoder.

    uv run python -m scripts.jev_clf_latency_probe
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jev_clf import schema as S  # noqa: E402
from jev_clf.client import Choice, Noul, Score  # noqa: E402
from jev_clf.model import OptionScorer  # noqa: E402

KS = (1, 2, 4, 8, 16)
REPS = 5
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


def questions_for(n: int) -> dict:
    """n mixed-primitive questions over one state."""
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
            criteria=["none", "weak", "one relevant", "several relevant"],
        ),
        Noul(instructions="Is the claim about a scientific or technical subject?"),
        Noul(instructions="Would a careful reader call this claim controversial?"),
        Score(
            instructions="How clearly is the subject named?",
            criteria=["not at all", "vaguely", "broadly", "explicitly"],
        ),
        Noul(instructions="Does the evidence contain a date or a number?"),
        Noul(instructions="Is the evidence longer than the claim itself?"),
        Score(
            instructions="How much interpretation is needed to link evidence to claim?",
            criteria=["a great deal", "substantial", "a little", "none"],
        ),
        Noul(instructions="Is the evidence longer than the claim?"),
        Noul(instructions="Does the evidence cite a source?"),
        Noul(instructions="Is the claim quantitative?"),
        Noul(instructions="Does the evidence contradict the claim?"),
        Noul(instructions="Is the evidence recent?"),
        Noul(instructions="Is the claim widely held?"),
        Noul(instructions="Does the evidence mention the claim's subject by name?"),
    ]
    for q in extras[: n - 1]:
        qs[f"q{len(qs)}"] = q
    return qs


def main() -> None:
    rows = [r for r in S.read_rows(ROOT / "data/factcheck/ground_truth.jsonl") if r.split == "val"]
    state = rows[0].state

    print(f"loading OptionScorer ({ENCODER}, frozen)...")
    scorer = OptionScorer(encoder_name=ENCODER, head_width=64, freeze_encoder=True)
    scorer.eval()
    print("loaded\n")

    print(f"{'k':>4}  {'ms':>8}  {'x':>6}   (lower is flat)")
    base = None
    out = []
    for k in KS:
        qs = questions_for(k)
        lat = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            with torch.no_grad() if (torch := __import__("torch")) else _null():
                scorer.forward([state], [qs])
            lat.append((time.perf_counter() - t0) * 1000)
        med = statistics.median(lat)
        base = base or med
        print(f"{k:>4}  {med:>8.1f}  {med/base:>5.2f}x")
        out.append({"k": k, "ms": med, "x": med / base})

    print(
        "\nIf roughly flat: the architecture gives Jev's single-pass property, and the\n"
        "autoregressive readout (6.39x at k=8) is the thing to replace."
    )

    import json

    dest = ROOT / "results" / "option_scorer_latency.json"
    dest.write_text(json.dumps({"encoder": ENCODER, "rows": out}, indent=2) + "\n")
    print("wrote", dest)


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    main()
