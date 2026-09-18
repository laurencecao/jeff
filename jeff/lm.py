"""Zero-shot LM label readout for jeff — the classification is read OUT of
a real language model, not bolted onto a frozen featurizer.

A generative instruct LM already represents claim/evidence relations. Instead
of training a head, we render ``state`` + ``question`` through the model's own
chat template, end the turn with a cue naming the allowed label strings, and
read the answer distribution off the model's next-token logits. Labels and
their natural-language definitions still arrive in the prompt at call time, so
the text-conditioned contract is preserved: a differently-worded question or a
different number of labels works unchanged.

Two readout rules are implemented and both are reported by the eval driver:

- ``first_token``: softmax over the FIRST token id of each label's
  tokenization at the boundary position. Cheap (one boundary distribution per
  item) but lossy for multi-token labels: ``not_enough_info`` is judged only
  on P("not"), and two labels sharing a first token collide (each such label
  is credited the full shared mass — flagged via ``first_token_collisions``).
- ``sequence`` (DEFAULT): each label is scored as the whole token sequence it
  would occupy as the answer — sum of log-probs of its tokens at the positions
  right after the generation prompt — then softmaxed across labels. This is
  the faithful readout for multi-token labels: it prices the entire string
  the model would emit, not just its first BPE piece. No length normalization
  is applied: the label IS the sequence, and a longer label legitimately
  costs more probability mass.

Tokenization note (verified on Qwen BPE): for the chat templates used here,
``encode(prompt + label)`` has ``encode(prompt)`` as a strict prefix, so the
label's standalone token ids are exactly the ids the model would generate.
``_label_token_suffix`` asserts this per (prompt, label) pair and falls back
to the joint-encoding suffix if a boundary merge ever appears, so the scored
sequence is always the model's natural continuation.

Verified against transformers 5.17: ``from_pretrained(dtype=...)``,
``apply_chat_template``, and the ``logits_to_keep`` forward kwarg (which keeps
the vocab-sized logits tensor to the handful of positions we actually read —
important on a 36 GB unified-memory box).
"""
from __future__ import annotations

import time
from typing import Any, Iterable, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .model import state_to_text
from .schema import (
    DecisionRow,
    PredictionRow,
    Question,
    label_space,
    normalize,
    question_to_text,
)

READOUTS = ("sequence", "first_token")
DEFAULT_READOUT = "sequence"

_FALLBACK_TEMPLATE = "{content}\n\nAnswer:"


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


def build_prompt(state: Any, question: Question, tokenizer: Any) -> str:
    """Render state + question into the model's chat template.

    The user turn carries the flat state text (``jeff.model.state_to_text``),
    the question text (``schema.question_to_text``: instructions plus one
    ``label: definition`` line per label), and a cue naming the exact label
    strings. The string ends with the template's generation prompt, so the
    first scored position is where the model would emit the answer.
    """
    labels = label_space(question)
    cue = (
        "Answer with exactly one of the following labels and nothing else: "
        + ", ".join(labels)
    )
    content = f"{state_to_text(state)}\n\n{question_to_text(question)}\n\n{cue}"
    messages = [{"role": "user", "content": content}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:  # noqa: BLE001 - base models may ship no chat template
        return _FALLBACK_TEMPLATE.format(content=content)


def label_token_ids(tokenizer: Any, labels: Iterable[str]) -> dict[str, int]:
    """Map each label to the token id used for first-token readout.

    The id is the first token of the label's standalone encoding, which (per
    the module docstring's prefix property) is also the first token the model
    emits when the label is its answer. Labels that tokenize to nothing raise.
    """
    out: dict[str, int] = {}
    for label in labels:
        ids = tokenizer.encode(label, add_special_tokens=False)
        if not ids:
            raise ValueError(f"label {label!r} tokenizes to zero tokens")
        out[label] = ids[0]
    return out


# ---------------------------------------------------------------------------
# The labeler
# ---------------------------------------------------------------------------


class LMLabeler:
    """Zero-shot text-conditioned classifier on a generative LM.

    ``probs(states, questions)`` returns one ``{label: prob}`` dict per
    (state, question) pair, in ``label_space`` order, summing to 1.
    ``questions`` may be a single ``Question`` (broadcast to every state) or a
    sequence parallel to ``states``. ``readout`` selects the default rule;
    ``predict_variants`` computes both rules in a single pass for reporting.
    """

    def __init__(
        self,
        model_id: str,
        device: str | None = None,
        dtype: Any = None,
        readout: str = DEFAULT_READOUT,
        batch_size: int = 8,
    ) -> None:
        if readout not in READOUTS:
            raise ValueError(f"readout must be one of {READOUTS}, got {readout!r}")
        self.model_id = model_id
        self.readout = readout
        self.batch_size = batch_size
        self.name = f"lm:{model_id}:{readout}"
        self.device = device or (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype or "auto"
        )
        self.model.to(self.device)
        self.model.eval()

    # -- scoring internals --------------------------------------------------

    def _encode_prompt(self, state: Any, question: Question) -> list[int]:
        text = build_prompt(state, question, self.tokenizer)
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _label_token_suffix(self, prompt_ids: list[int], label: str) -> list[int]:
        """Token ids the model would generate for ``label`` after this prompt.

        Asserts the prefix property (joint encoding = prompt ids + standalone
        label ids); on a boundary merge, falls back to the joint suffix so the
        scored sequence is still the natural continuation.
        """
        standalone = self.tokenizer.encode(label, add_special_tokens=False)
        if not standalone:
            raise ValueError(f"label {label!r} tokenizes to zero tokens")
        joint = self.tokenizer.encode(
            self.tokenizer.decode(prompt_ids) + label, add_special_tokens=False
        )
        if joint[: len(prompt_ids)] == prompt_ids:
            return standalone
        return joint[len(prompt_ids) :]

    @torch.no_grad()
    def _score_group(
        self,
        prompt_ids_list: list[list[int]],
        label_ids_list: list[list[int]],
    ) -> list[dict[str, Any]]:
        """One batched forward per label; returns per-item raw scores.

        Every item in the group shares the same ordered label list. For label
        k we run ``prompt + label_k`` left-padded with
        ``logits_to_keep = len(label_k) + 1``: kept index 0 is the boundary
        (predicts the first label token) and kept index j predicts label token
        j. The boundary distribution is label-independent, so the first-token
        readout is captured once, from the k=0 pass.
        """
        n = len(prompt_ids_list)
        seq_logps: list[list[float]] = [[0.0] * len(label_ids_list) for _ in range(n)]
        boundary: list[torch.Tensor | None] = [None] * n
        first_ids = [lids[0] for lids in label_ids_list]

        # Sort by length so a batch pads to similar sizes.
        order = sorted(range(n), key=lambda i: len(prompt_ids_list[i]))
        for k, lids in enumerate(label_ids_list):
            for start in range(0, n, self.batch_size):
                idxs = order[start : start + self.batch_size]
                seqs = [prompt_ids_list[i] + lids for i in idxs]
                maxlen = max(len(s) for s in seqs)
                pad = self.tokenizer.pad_token_id
                input_ids = torch.tensor(
                    [[pad] * (maxlen - len(s)) + s for s in seqs],
                    dtype=torch.long,
                    device=self.device,
                )
                mask = torch.tensor(
                    [[0] * (maxlen - len(s)) + [1] * len(s) for s in seqs],
                    dtype=torch.long,
                    device=self.device,
                )
                position_ids = (mask.cumsum(-1) - 1).clamp_min(0)
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=mask,
                    position_ids=position_ids,
                    logits_to_keep=len(lids) + 1,
                )
                lp = torch.log_softmax(out.logits.float(), dim=-1)
                # kept index j predicts label token j (index 0 = boundary).
                tgt = torch.tensor(lids, device=self.device).view(1, -1, 1)
                token_lp = (
                    lp[:, : len(lids), :]
                    .gather(2, tgt.expand(len(idxs), -1, -1))
                    .squeeze(-1)
                )
                for b, i in enumerate(idxs):
                    seq_logps[i][k] = float(token_lp[b].sum())
                if k == 0:
                    for b, i in enumerate(idxs):
                        boundary[i] = lp[b, 0, :].cpu()  # log-probs at boundary
                del out, lp, token_lp, input_ids, mask, position_ids

        first_idx = torch.tensor(first_ids)
        items = []
        for i in range(n):
            b = boundary[i]
            assert b is not None
            items.append(
                {
                    "seq_logp": seq_logps[i],
                    "first_logp": b[first_idx],
                }
            )
        return items

    def _score_items(
        self, items: list[tuple[Any, Question]]
    ) -> list[dict[str, Any]]:
        """Score (state, question) pairs; groups by shared label list."""
        prompts: list[list[int]] = []
        label_ids: list[list[list[int]]] = []
        for state, question in items:
            labels = label_space(question)
            pids = self._encode_prompt(state, question)
            prompts.append(pids)
            label_ids.append(
                [self._label_token_suffix(pids, label) for label in labels]
            )

        # Group consecutive items sharing the same label-id lists.
        results: list[dict[str, Any] | None] = [None] * len(items)
        i = 0
        while i < len(items):
            j = i + 1
            while j < len(items) and label_ids[j] == label_ids[i]:
                j += 1
            group = self._score_group(prompts[i:j], label_ids[i])
            for off, scored in enumerate(group):
                lids = label_ids[i]
                first_ids = [l[0] for l in lids]
                collisions = sorted(
                    {tid for tid in first_ids if first_ids.count(tid) > 1}
                )
                scored["collisions"] = [
                    self.tokenizer.decode([tid]) for tid in collisions
                ]
                results[i + off] = scored
            i = j
        return [r for r in results if r is not None]

    @staticmethod
    def _probs_from(
        scored: dict[str, Any], labels: list[str], readout: str
    ) -> dict[str, float]:
        if readout == "sequence":
            logits = torch.tensor(scored["seq_logp"])
        else:
            logits = scored["first_logp"]
        probs = torch.softmax(logits, dim=-1).tolist()
        return {label: float(p) for label, p in zip(labels, probs)}

    def _normalize_items(
        self,
        states: Sequence[Any],
        questions: Question | Sequence[Question],
    ) -> list[tuple[Any, Question]]:
        states = list(states)
        if isinstance(questions, (str, bytes)):
            raise TypeError("questions must be a Question or a sequence of them")
        if isinstance(questions, Sequence):
            questions = list(questions)
            if len(questions) != len(states):
                raise ValueError(
                    f"questions ({len(questions)}) must broadcast or match "
                    f"states ({len(states)})"
                )
        else:
            questions = [questions] * len(states)
        return list(zip(states, questions))

    # -- public API ---------------------------------------------------------

    def probs(
        self,
        states: Sequence[Any],
        questions: Question | Sequence[Question],
    ) -> list[dict[str, float]]:
        """One ``{label: prob}`` per (state, question), ``label_space`` order."""
        items = self._normalize_items(states, questions)
        scored = self._score_items(items)
        return [
            self._probs_from(s, label_space(q), self.readout)
            for s, (_, q) in zip(scored, items)
        ]

    def probs_variants(
        self,
        states: Sequence[Any],
        questions: Question | Sequence[Question],
    ) -> list[dict[str, Any]]:
        """Both readouts in one pass: ``{"sequence": probs, "first_token": probs,
        "first_token_collisions": [...]}`` per item."""
        items = self._normalize_items(states, questions)
        scored = self._score_items(items)
        out = []
        for s, (_, q) in zip(scored, items):
            labels = label_space(q)
            out.append(
                {
                    "sequence": self._probs_from(s, labels, "sequence"),
                    "first_token": self._probs_from(s, labels, "first_token"),
                    "first_token_collisions": s["collisions"],
                }
            )
        return out

    def predict(
        self, rows: Iterable[DecisionRow], readout: str | None = None
    ) -> list[PredictionRow]:
        """``eval.throughput``/metrics-compatible row predictor."""
        return self.predict_variants(rows)[readout or self.readout]

    def predict_variants(
        self, rows: Iterable[DecisionRow]
    ) -> dict[str, list[PredictionRow]]:
        """One pass over rows; PredictionRows for BOTH readout rules."""
        rows = list(rows)
        items: list[tuple[Any, Question]] = []
        index: list[tuple[DecisionRow, str]] = []
        for row in rows:
            for qid, question in row.questions.items():
                items.append((row.state, question))
                index.append((row, qid))

        t0 = time.perf_counter()
        scored = self._score_items(items)
        total_ms = (time.perf_counter() - t0) * 1000.0
        per_item_ms = total_ms / max(len(items), 1)

        out: dict[str, list[PredictionRow]] = {r: [] for r in READOUTS}
        for (row, qid), s in zip(index, scored):
            labels = label_space(row.questions[qid])
            for readout in READOUTS:
                probs = normalize(self._probs_from(s, labels, readout))
                meta: dict[str, Any] = {
                    "lm_model": self.model_id,
                    "readout": readout,
                }
                if s["collisions"]:
                    meta["first_token_collisions"] = s["collisions"]
                out[readout].append(
                    PredictionRow(
                        row_id=row.row_id,
                        question_id=qid,
                        probs=probs,
                        confidence=max(probs.values()),
                        model=f"lm:{self.model_id}:{readout}",
                        latency_ms=per_item_ms,
                        meta=meta,
                    )
                )
        return out
