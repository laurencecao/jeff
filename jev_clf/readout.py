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

import contextlib

import torch

MODES = ("auto", "first_token", "sequence")


def _locked(lock):
    """``with _locked(tok_lock):`` — a no-op when the caller passes no lock.

    The fast tokenizer is not thread-safe (concurrent ``tok(...)`` raises
    "Already borrowed" from the Rust side), so a client that wants to run
    several requests at once passes one lock and every tokenizer call in this
    module goes through it. The forward passes themselves stay outside the
    lock and are safe to run concurrently.
    """
    return lock if lock is not None else contextlib.nullcontext()



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
    tok_lock=None,
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

    ``tok_lock``: optional lock held around every tokenizer call, for callers
    that run several questions concurrently (the fast tokenizer is not
    thread-safe). Forward passes are never inside the lock.
    """
    if not labels:
        raise ValueError("labels must be non-empty")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    with _locked(tok_lock):
        variants = label_token_variants(tok, labels)
        resolved = choose_mode(variants) if mode == "auto" else mode
        enc = tok(text, return_tensors="pt", truncation=True, max_length=max_length)

    if resolved == "first_token":
        # One shared prompt pass: softmax over each label's first-token logit.
        with torch.no_grad():
            logits = model(**enc.to(device)).logits[0, -1].float()
        sub = torch.tensor([logits[variants[label][0]] for label in labels])
        probs = torch.softmax(sub, dim=-1)
        return {label: float(p) for label, p in zip(labels, probs)}

    # Sequence readout: one forward per label over prompt + label.
    # (The prompt-only pass is NOT run here — the first-token readout is the
    # only consumer of it, so computing it for a sequence question was wasted
    # work; see the forward-count note in jev_clf/client.py.)
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


def distribution_batch(
    model,
    tok,
    requests: list[tuple[str, list[str]]],
    *,
    device: str,
    max_length: int = 2048,
    mode: str = "auto",
    batch_size: int = 8,
    tok_lock=None,
) -> tuple[list[dict[str, float]], int]:
    """Batched sibling of :func:`distribution`: many prompts, few forwards.

    ``requests`` is ``[(text, labels), ...]`` and the result is
    ``(probs_list, n_passes)`` with one distribution per request, in order.

    Cost:
      * ``first_token`` questions cost ONE forward per ``batch_size`` chunk,
        whatever the number of prompts (the per-question path costs one
        forward per prompt).
      * ``sequence`` questions cost one forward per label *index* per chunk
        (the per-question path costs one forward per label per prompt).

    Measured warm on 4x A2 (15 GB), bfloat16, model sharded across all four by device_map="auto": 8 same-length Noul questions 1295 ms / 8 forwards ->
    729 ms / 1 forward. A mixed 4-question request (Choice + 2 Noul + Score)
    1153 ms / 7 forwards -> 1056 ms / 5 forwards. The pass count drops much
    more than the wall clock because a small batch on a sharded model is
    latency-bound, not throughput-bound.

    Prompts are LEFT-padded (the tokenizer's ``padding_side`` is set and
    restored under ``tok_lock``) and ``position_ids`` are recomputed from the
    attention mask, so every row is scored at its own last real token.

    NOT bit-identical to the per-question path: a padded batch is a different
    GEMM shape, so on bfloat16 the probabilities move by ~1e-4 (Choice) to
    ~4e-3 (a Noul yes/no that sits near 0). Ranking/argmax is stable in the
    measurements, but do not use this path to reproduce benchmarked
    per-question numbers. ``scripts/test_readout_batch.py`` measures the gap.
    """
    if not requests:
        return [], 0
    for _, labels in requests:
        if not labels:
            raise ValueError("labels must be non-empty")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    batch_size = max(1, int(batch_size))

    out: list[dict[str, float] | None] = [None] * len(requests)
    passes = 0

    with _locked(tok_lock):
        variants = [label_token_variants(tok, labels) for _, labels in requests]
        resolved = [choose_mode(v) if mode == "auto" else mode for v in variants]
        # Encode every prompt once; both readouts work from these ids.
        prompt_ids = [
            tok(text, truncation=True, max_length=max_length)["input_ids"]
            for text, _ in requests
        ]

    # ---- first-token questions: one chunked forward for all of them -------
    ft = [i for i, r in enumerate(resolved) if r == "first_token"]
    for chunk in _length_chunks(ft, prompt_ids, batch_size):
        batch = _left_padded(tok, [prompt_ids[i] for i in chunk], device)
        with torch.no_grad():
            logits = model(**batch).logits[:, -1, :].float()
        passes += 1
        for row, i in enumerate(chunk):
            labels = requests[i][1]
            var = variants[i]
            sub = torch.tensor([logits[row][var[label][0]] for label in labels])
            probs = torch.softmax(sub, dim=-1)
            out[i] = {label: float(p) for label, p in zip(labels, probs)}

    # ---- sequence questions: one chunked forward per label index ----------
    # A pass scores one label position for every row, so rows must agree on
    # their label list — and therefore on the label index being scored.
    groups: dict[tuple[str, ...], list[int]] = {}
    for i in [i for i, r in enumerate(resolved) if r == "sequence"]:
        groups.setdefault(tuple(requests[i][1]), []).append(i)

    for labels_tuple, rows in groups.items():
        labels = list(labels_tuple)
        totals: dict[int, list[float]] = {i: [0.0] * len(labels) for i in rows}
        for k, label in enumerate(labels):
            tokens = [variants[i][label] for i in rows]
            for chunk, chunk_tokens in _token_chunks(rows, tokens, batch_size):
                full = [prompt_ids[i] + t for i, t in zip(chunk, chunk_tokens)]
                batch = _left_padded(tok, full, device)
                with torch.no_grad():
                    logits = model(**batch).logits.float()
                passes += 1
                width = int(batch["input_ids"].shape[1])
                for row, (i, t) in enumerate(zip(chunk, chunk_tokens)):
                    boundary = width - len(t) - 1  # row's last prompt token
                    total = 0.0
                    for j, tid in enumerate(t):
                        total += float(
                            torch.log_softmax(logits[row][boundary + j], dim=-1)[tid]
                        )
                    totals[i][k] = total
        for i in rows:
            probs = torch.softmax(
                torch.tensor(totals[i], dtype=torch.float32), dim=-1
            )
            out[i] = {label: float(p) for label, p in zip(labels, probs)}

    missing = [i for i, p in enumerate(out) if p is None]
    if missing:
        raise RuntimeError(f"distribution_batch produced no answer for {missing}")
    return [p for p in out if p is not None], passes


# -- batch plumbing ---------------------------------------------------------


def _length_chunks(
    idx: list[int], prompts: list[list[int]], batch_size: int
) -> list[list[int]]:
    """Sort by prompt length, then chunk — keeps padding waste small."""
    order = sorted(idx, key=lambda i: len(prompts[i]))
    return [order[s:s + batch_size] for s in range(0, len(order), batch_size)]


def _token_chunks(
    idx: list[int], tokens: list[list[int]], batch_size: int
) -> list[tuple[list[int], list[list[int]]]]:
    order = sorted(range(len(idx)), key=lambda j: len(tokens[j]))
    out: list[tuple[list[int], list[list[int]]]] = []
    for s in range(0, len(order), batch_size):
        sel = order[s:s + batch_size]
        out.append(([idx[j] for j in sel], [tokens[j] for j in sel]))
    return out


def _left_padded(tok, id_seqs: list[list[int]], device: str) -> dict:
    """input_ids/attention_mask/position_ids for a causal LM, left-padded.

    Left padding puts every row's last real token at index -1, and position
    ids counted from each row's first real token keep the positions identical
    to a single-prompt pass (so RoPE sees the same distances).
    """
    width = max(len(s) for s in id_seqs)
    pad = tok.pad_token_id
    if pad is None:
        pad = tok.eos_token_id
    ids, mask = [], []
    for s in id_seqs:
        pad_n = width - len(s)
        ids.append([pad] * pad_n + list(s))
        mask.append([0] * pad_n + [1] * len(s))
    batch = {
        "input_ids": torch.tensor(ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(mask, dtype=torch.long, device=device),
    }
    batch["position_ids"] = (batch["attention_mask"].cumsum(-1) - 1).clamp_min(0)
    return batch

