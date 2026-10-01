"""Batched readout: does it agree with the per-question readout, and how fast?

Two layers, because the answer differs by layer:

1. MECHANICS (always runs, no download): a tiny randomly-initialised model
   with the real tokenizer. Batching must be *numerically identical* here —
   one padded batch row vs the same prompt scored alone, first-token and
   sequence readouts, mixed primitives in one call, chunking by batch_size,
   label order preserved, distributions normalised. This is what the padding
   + position_ids plumbing has to get right.

2. THE REAL ADAPTER (``--model``/``--adapter``): a 4B model does not
   reproduce a GEMM done in a different batch shape (and bfloat16) bit for
   bit, so the check there is "same ranking / same argmax, probabilities
   within a documented tolerance", plus the measured speedup. This is the
   honest boundary of the batched path: it is a throughput option, not a way
   to reproduce the benchmarked per-question numbers.

    uv run python -m scripts.test_readout_batch
    uv run python -m scripts.test_readout_batch --model Qwen/Qwen3-4B-Instruct-2507 \
        --adapter artifacts/jev_clf/lora_4b_multi
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jev_clf import readout  # noqa: E402
from jev_clf import schema as S  # noqa: E402

TOKENIZER_ID = os.environ.get("JEVCLF_TOKENIZER", "Qwen/Qwen3-4B-Instruct-2507")
TINY_TOL = 1e-4  # float32, same weights, two GEMM shapes


def tiny_model(tok):
    """Random-weight 2-layer model with the real tokenizer's vocab."""
    cfg = AutoConfig.from_pretrained(TOKENIZER_ID)
    cfg.num_hidden_layers = 2
    cfg.hidden_size = 64
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    cfg.head_dim = 16
    cfg.intermediate_size = 128
    cfg.max_position_embeddings = 4096
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(cfg).eval()


# ---------------------------------------------------------------------------
# mechanics
# ---------------------------------------------------------------------------


def test_padding_is_invariant(tok, model) -> int:
    """A row's last-token logits must not depend on the other rows' lengths.

    ``_left_padded`` keeps the caller's row order (only the chunker sorts, and
    it carries the original indices), so the comparison is index-to-index.
    """
    texts = ["short prompt", "a considerably longer prompt " * 6, "medium prompt " * 3]
    failures = 0
    ids = [tok(t, truncation=True, max_length=512)["input_ids"] for t in texts]
    with torch.no_grad():
        singles = [
            model(**readout._left_padded(tok, [s], "cpu")).logits[0, -1].float()
            for s in ids
        ]
        batched = model(**readout._left_padded(tok, ids, "cpu")).logits[:, -1, :].float()
    worst = 0.0
    for i in range(len(texts)):
        d = float((batched[i] - singles[i]).abs().max())
        worst = max(worst, d)
        if d > TINY_TOL:
            print(f"    [fail] row {i}: max |logit delta| = {d:.2e}")
            failures += 1
    print(f"  padding invariant ({len(texts)} rows of length "
          f"{[len(i) for i in ids]}): worst |logit delta| = {worst:.2e}")
    return failures


def test_batch_matches_serial(tok, model) -> int:
    """Batch over a mixed Choice / Noul / Score set == per-question readout."""
    state = {"claim": "The museum opened in 1998.", "evidence": ["It opened in 1998."]}
    questions = {
        "verdict": S.ChoiceQuestion(
            instructions="Judge the claim using the evidence.",
            criteria={
                "supported": "The evidence establishes the claim.",
                "refuted": "The evidence contradicts the claim.",
                "not_enough_info": "The evidence is insufficient.",
            },
        ),
        "truefalse": S.ChoiceQuestion(
            instructions="Is the claim true?",
            criteria={"true": "The claim is true.", "false": "The claim is false."},
        ),
        "has_date": S.NoulQuestion(instructions="Does the evidence include a year?"),
        "strength": S.ScoreQuestion(
            instructions="How strongly does the evidence settle the claim?",
            criteria=["none", "weak", "moderate", "strong"],  # shares first token
        ),
    }
    from scripts.jev_clf_lm_eval import build_inputs  # noqa: E402

    requests = [
        (build_inputs(tok, state, q), S.label_space(q)) for q in questions.values()
    ]
    serial: list[dict[str, float]] = []
    for text, labels in requests:
        serial.append(
            readout.distribution(model, tok, text, labels, device="cpu", max_length=512)
        )
    batched, passes = readout.distribution_batch(
        model, tok, requests, device="cpu", max_length=512, mode="auto"
    )

    failures = 0
    worst = 0.0
    for i, (qid, q) in enumerate(questions.items()):
        labels = S.label_space(q)
        if list(batched[i]) != labels:
            print(f"    [fail] {qid}: label order changed: {list(batched[i])} != {labels}")
            failures += 1
        total = sum(batched[i].values())
        if abs(total - 1.0) > 1e-5:
            print(f"    [fail] {qid}: probabilities sum to {total}")
            failures += 1
        d = max(abs(batched[i][l] - serial[i][l]) for l in labels)
        worst = max(worst, d)
        mode = readout.choose_mode(readout.label_token_variants(tok, labels))
        print(f"  {qid:10} mode={mode:12} maxdelta={d:.2e} "
              f"batched={ {a: round(b, 5) for a, b in batched[i].items()} }")
        if d > TINY_TOL:
            print(f"    [fail] {qid}: delta {d:.2e} > {TINY_TOL:.0e}")
            failures += 1

    # pass accounting: 3 first-token questions in one chunk -> 1 pass, the
    # 4-label Score -> one pass per label index.
    expected = 1 + 4
    if passes != expected:
        print(f"    [fail] passes={passes}, expected {expected}")
        failures += 1
    else:
        print(f"  pass count: {passes} (1 for the three first-token questions + "
              f"4 label positions for the Score question)")
    print(f"  worst probability delta = {worst:.2e}")
    return failures


def test_chunking_and_batch_size(tok, model) -> int:
    """batch_size must only change the number of forwards, not the answer."""
    texts = [f"prompt number {i} " * (i + 1) for i in range(7)]
    requests = [(t, ["yes", "no"]) for t in texts]
    ref, _ = readout.distribution_batch(
        model, tok, requests, device="cpu", max_length=512, batch_size=8
    )
    failures = 0
    for bs in (1, 2, 7):
        got, passes = readout.distribution_batch(
            model, tok, requests, device="cpu", max_length=512, batch_size=bs
        )
        d = max(abs(got[i]["yes"] - ref[i]["yes"]) for i in range(len(texts)))
        print(f"  batch_size={bs}: passes={passes} worst delta vs batch_size=8: {d:.2e}")
        if d > TINY_TOL:
            failures += 1
    return failures


def mechanics() -> int:
    print(f"tokenizer: {TOKENIZER_ID}")
    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = tiny_model(tok)
    failures = 0
    failures += test_padding_is_invariant(tok, model)
    failures += test_batch_matches_serial(tok, model)
    failures += test_chunking_and_batch_size(tok, model)
    return failures


# ---------------------------------------------------------------------------
# the real adapter
# ---------------------------------------------------------------------------


def real_model(model_id: str, adapter: str | None) -> int:
    from jev_clf.client import SystemOneClient  # noqa: E402

    client = SystemOneClient(base_model=model_id, adapter=adapter)
    state = {
        "claim": "The Eiffel Tower is located in Barcelona.",
        "evidence": [
            {
                "title": "e1",
                "text": "The Eiffel Tower is a wrought-iron lattice tower on the "
                        "Champ de Mars in Paris, France.",
            }
        ],
    }
    questions = {
        "verdict": S.ChoiceQuestion(
            instructions="Decide whether the evidence establishes the claim, contradicts it, or cannot decide it.",
            criteria=dict(S.FACTCHECK_CRITERIA),
        ),
        "has_date": S.NoulQuestion(instructions="Does the evidence contain a date or a number?"),
        "about_place": S.NoulQuestion(instructions="Is the claim about a location?"),
        "strength": S.ScoreQuestion(
            instructions="How much evidence is provided?",
            criteria=["none", "weak", "relevant", "several"],
        ),
    }

    t0 = time.perf_counter()
    serial = client.system_one(state, questions)
    t_serial = time.perf_counter() - t0
    t0 = time.perf_counter()
    batched = client.system_one(state, questions, batch_questions=True)
    t_batch = time.perf_counter() - t0

    print(f"\nmodel: {serial.model}")
    print(f"  serial : {t_serial * 1000:7.0f} ms  {serial.n_forward_passes} forwards")
    print(f"  batched: {t_batch * 1000:7.0f} ms  {batched.n_forward_passes} forwards"
          f"  ({t_serial / t_batch:.2f}x)")

    failures = 0
    for qid, question in questions.items():
        mode = serial.readout_modes[qid]
        if isinstance(question, S.NoulQuestion):
            a, b = serial.nouls[qid].noul, batched.nouls[qid].noul
            print(f"  {qid:12} mode={mode:12} serial={a:.6f} batched={b:.6f} "
                  f"delta={abs(a - b):.2e}")
        elif isinstance(question, S.ScoreQuestion):
            a, b = serial.scores[qid], batched.scores[qid]
            print(f"  {qid:12} mode={mode:12} argmax {max(a.probabilities, key=a.probabilities.get)}"
                  f" -> {max(b.probabilities, key=b.probabilities.get)}")
            for label in a.probabilities:
                print(f"      {label:16} serial={a.probabilities[label]:.6f} "
                      f"batched={b.probabilities[label]:.6f} "
                      f"delta={abs(a.probabilities[label] - b.probabilities[label]):.2e}")
        else:
            a, b = serial.choices[qid], batched.choices[qid]
            print(f"  {qid:12} mode={mode:12} argmax {a.choice} -> {b.choice}")
            for label in a.probabilities:
                print(f"      {label:16} serial={a.probabilities[label]:.6f} "
                      f"batched={b.probabilities[label]:.6f} "
                      f"delta={abs(a.probabilities[label] - b.probabilities[label]):.2e}")

    # What is actually promised: same decision, not the same float. Keep the
    # ranking check strict and the probability check loose and reported.
    for qid, question in questions.items():
        if isinstance(question, S.ScoreQuestion):
            sa = max(serial.scores[qid].probabilities, key=serial.scores[qid].probabilities.get)
            sb = max(batched.scores[qid].probabilities, key=batched.scores[qid].probabilities.get)
            if sa != sb:
                print(f"    [fail] {qid}: argmax moved {sa} -> {sb}")
                failures += 1
        elif not isinstance(question, S.NoulQuestion):
            if serial.choices[qid].choice != batched.choices[qid].choice:
                print(f"    [fail] {qid}: argmax moved "
                      f"{serial.choices[qid].choice} -> {batched.choices[qid].choice}")
                failures += 1
    return failures


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="also check a real model+adapter (downloads/loads 4B)")
    ap.add_argument("--adapter", default=None)
    args = ap.parse_args()

    print("=== 1. batched readout mechanics (tiny model, no download) ===")
    failures = mechanics()
    if args.model:
        print("\n=== 2. the real adapter (ranking + throughput, not float equality) ===")
        failures += real_model(args.model, args.adapter)

    print(f"\nFAILURES={failures}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
