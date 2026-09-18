"""Read the classification out of the model's own next-token distribution.

Two readouts over the label set, chosen per question:

* ``first_token`` — softmax over each label's first-token logit, from ONE
  forward pass of the prompt. Correct when every label starts with a
  DISTINCT token (choice labels: " supported" -> [7248], " refuted" ->
  [2053, 2774], " not_enough_info" -> [537, ...]; yes/no: " yes" -> [9834],
  " no" -> [902]). This is the readout the benchmark
  (scripts/jev_clf_autoresearch.py) measures, so it must stay the default
  for those question types.

* ``sequence`` — score each label as a whole token sequence: sum the
  log-softmaxes of the prompt-boundary logits that predict each label token,
  then softmax across the per-label totals. Required when the first-token
  readout degenerates: the single-digit score levels "0".."3" all tokenize
  as [220, 15..18], i.e. they share the bare-space first token 220, so the
  first-token readout reads the SAME logit four times (measured uniform
  0.25 — arithmetic noise, not a prediction). Scoring the full sequence
  breaks the tie.

All-or-nothing within a question: exactly one readout is used for all of a
question's labels, so the numbers fed into the final softmax share one
scale. Mixing readouts per label would compare uncalibrated values.
"""

from __future__ import annotations

import torch

MODES = ("auto", "first_token", "sequence")


def label_token_variants(tok, labels: list[str]) -> dict[str, list[int]]:
    """Token id list for each label, trying the leading-space (mid-sentence)
    form first, falling back to the bare label.

    Same recipe as scripts.jev_clf_lm_eval.label_variants (the benchmarked
    readout script), so the id lists it returns are identical to those.
    """
    out: dict[str, list[int]] = {}
    for label in labels:
        ids = tok(" " + label, add_special_tokens=False)["input_ids"]
        if not ids:
            ids = tok(label, add_special_tokens=False)["input_ids"]
        out[label] = ids
    return out


def choose_mode(variants: dict[str, list[int]]) -> str:
    """Resolve the ``auto`` rule for a label set.

    ``"first_token"`` iff every label has at least one token and all the
    FIRST tokens are DISTINCT; otherwise ``"sequence"``.

    Distinctness — not single-token-ness — is what makes the first-token
    readout meaningful: if two labels share a first token, their first-token
    logits are the same number (the score levels " 0".." 3" all start with
    the bare-space token 220). The readout only ever consumes
    ``variants[label][0]``, so multi-token labels are fine: with the default
    tokenizer, " refuted" -> [2053, 2774] and " not_enough_info" ->
    [537, 6205, 1384, 3109] still use the first-token readout over ids
    2053/537, exactly as the benchmark (scripts/jev_clf_autoresearch.py,
    which reads only the first id) does. A naive "exactly one token" gate
    would misroute those choice questions to the sequence readout and
    silently change benchmarked behaviour.
    """
    seqs = list(variants.values())
    if seqs and all(seqs) and len({s[0] for s in seqs}) == len(seqs):
        return "first_token"
    return "sequence"


def distribution(
    model,
    tok,
    text: str,
    labels: list[str],
    *,
    device: str,
    max_length: int = 2048,
    mode: str = "auto",
) -> dict[str, float]:
    """Probabilities over ``labels`` from the model's next-token
    distribution given ``text``.

    ``text`` is the ALREADY-BUILT prompt string; this module deliberately
    does not import the prompt builder, keeping it dependency-light.

    Cost: one forward pass over the prompt (the prompt is tokenised once and
    its logits computed once, shared across labels). The ``first_token``
    readout consumes those logits directly, so that is its whole cost. The
    ``sequence`` readout needs logits INSIDE the label tokens, so it runs
    the model once per label over prompt+label — expected, and documented
    here because it shows up in latency for multi-token label sets.
    """
    if not labels:
        raise ValueError("labels must be non-empty")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    variants = label_token_variants(tok, labels)
    resolved = choose_mode(variants) if mode == "auto" else mode

    enc = tok(
        text, return_tensors="pt", truncation=True, max_length=max_length
    ).to(device)
    with torch.no_grad():
        logits = model(**enc).logits[0, -1].float()

    if resolved == "first_token":
        # One shared prompt pass: softmax over each label's first-token logit.
        sub = torch.tensor([logits[variants[label][0]] for label in labels])
        probs = torch.softmax(sub, dim=-1)
        return {label: float(p) for label, p in zip(labels, probs)}

    # Sequence readout: one EXTRA forward per label over prompt + label.
    prompt_ids = enc["input_ids"][0].tolist()  # post-truncation prompt ids
    totals: list[float] = []
    for label in labels:
        seq = variants[label]
        full = prompt_ids + seq
        with torch.no_grad():
            out = model(input_ids=torch.tensor([full]).to(device)).logits[0]
        total = 0.0
        for k, tid in enumerate(seq):
            # Position math: the logits at position (N-1+k) of `full` predict
            # the token at position (N+k), and with N = len(prompt_ids) that
            # is exactly seq[k] — the k-th label token. So `total` is the
            # factorised log p(label | prompt).
            pos = (len(full) - len(seq)) - 1 + k  # == N - 1 + k
            total += float(torch.log_softmax(out[pos].float(), dim=-1)[tid])
        totals.append(total)
    probs = torch.softmax(torch.tensor(totals, dtype=torch.float32), dim=-1)
    return {label: float(p) for label, p in zip(labels, probs)}