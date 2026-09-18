"""Verify the "many questions, one parallel forward pass" claim against the live API.

Jev's materials claim that all questions about a state are evaluated in parallel
and that adding questions barely changes response time. That is a concrete,
falsifiable statement about the API, so measure it: hold the state fixed, vary
the number of questions, and record latency and token usage.

    uv run python -m scripts.jev_parallel_probe
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

from typesafe_sdk import Choice, Noul, TypeSafeClient  # noqa: E402

from jeff import schema as S  # noqa: E402

MODEL = "jev-1.13.0"
KS = (1, 2, 4, 8, 16)
REPS = 3


def main() -> None:
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise SystemExit("TYPESAFE_API_KEY not set")
    rows = [r for r in S.read_rows(ROOT / "data/factcheck/ground_truth.jsonl") if r.split == "val"]
    state = rows[0].state
    question = rows[0].questions["verdict"]

    client = TypeSafeClient()
    out: list[dict] = []
    for k in KS:
        lat, tok = [], []
        for rep in range(REPS):
            qs: dict = {
                "verdict": Choice(
                    instructions=str(question.instructions),
                    criteria=dict(question.criteria),  # type: ignore[arg-type]
                )
            }
            for i in range(k - 1):
                # unique wording per (k, rep, i) so the cache cannot serve these
                qs[f"extra_{i}"] = Noul(
                    instructions=(
                        f"Probe k={k} rep={rep} index={i}: does the evidence "
                        f"mention any organisation or institution?"
                    )
                )
            t0 = time.perf_counter()
            r = client.system_one(state, qs)
            dt = (time.perf_counter() - t0) * 1000
            lat.append(dt)
            tok.append(r.usage.input_tokens)
        med = statistics.median(lat)
        out.append(
            {
                "n_questions": k,
                "median_latency_ms": round(med, 1),
                "latencies_ms": [round(x, 1) for x in lat],
                "median_input_tokens": statistics.median(tok),
            }
        )
        print(f"k={k:2d}  median {med:7.1f} ms   tokens {statistics.median(tok):.0f}")

    base = out[0]["median_latency_ms"]
    print("\nlatency growth vs k=1:")
    for row in out:
        print(f"  k={row['n_questions']:2d}  {row['median_latency_ms'] / base:.2f}x  "
              f"({row['median_latency_ms']:.1f} ms)")

    dest = ROOT / "results" / "jev_parallel_probe.json"
    dest.write_text(json.dumps({"model": MODEL, "reps": REPS, "results": out}, indent=2) + "\n")
    print("\nwrote", dest)


if __name__ == "__main__":
    main()
