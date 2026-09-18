"""Baselines: live Jev, zero-shot NLI, and the open option scorers.

Every baseline answers the same ``DecisionRow``s the student sees and writes
the same ``PredictionRow`` shape, so ``jeff.eval`` is model-agnostic.

- ``JevClassifier`` — the teacher answering live; the ceiling for both
  agreement and ground-truth metrics.
- ``ZeroShotNLI`` — a small MNLI cross-encoder; entailment -> supported,
  contradiction -> refuted, neutral -> not_enough_info.
- ``OptionScorerBase`` — jevlike / jevbetter through
  ``schema.to_option_scorer_rows``. Lossy by design: no instructions, no
  per-label definitions, exactly one question per row.
"""

from __future__ import annotations

import importlib
import time
from typing import Any, Iterable

from jeff.schema import (
    FACTCHECK_LABELS,
    ChoiceQuestion,
    DecisionRow,
    NoulQuestion,
    PredictionRow,
    label_space,
    normalize,
    one_hot,
    to_option_scorer_rows,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _claim_evidence(state: Any, row_id: str) -> tuple[str, list[dict[str, str]]]:
    """Extract (claim, evidence) from a factcheck-shaped state."""
    if not isinstance(state, dict) or "claim" not in state:
        raise ValueError(
            f"row {row_id}: baseline expects state={{'claim':..,'evidence':[..]}}, "
            f"got {type(state).__name__}"
        )
    return str(state["claim"]), list(state.get("evidence") or [])


def _premise(evidence: list[dict[str, str]]) -> str:
    """Flatten evidence passages into one NLI premise."""
    parts = []
    for passage in evidence:
        title, text = passage.get("title"), passage.get("text", "")
        parts.append(f"{title}: {text}" if title else text)
    return "\n".join(parts)


def _map_nli_probs(
    question: ChoiceQuestion | NoulQuestion,
    p_entail: float,
    p_contra: float,
    p_neutral: float,
    row_id: str,
) -> dict[str, float]:
    """Map NLI outputs onto a question's label space."""
    labels = label_space(question)
    if isinstance(question, NoulQuestion):
        return normalize({"yes": p_entail, "no": p_contra + p_neutral})
    if isinstance(question, ChoiceQuestion) and set(labels) == set(FACTCHECK_LABELS):
        return normalize(
            {"supported": p_entail, "refuted": p_contra, "not_enough_info": p_neutral}
        )
    raise ValueError(
        f"row {row_id}: ZeroShotNLI can only answer the factcheck choice "
        f"{FACTCHECK_LABELS} or a noul; got labels {labels}"
    )


def _no_evidence_probs(question: ChoiceQuestion | NoulQuestion, row_id: str) -> dict[str, float]:
    """Evidence-zero rows refuse: not_enough_info (or 'no' for a noul)."""
    labels = label_space(question)
    if isinstance(question, NoulQuestion):
        return one_hot(labels, "no")
    if "not_enough_info" in labels:
        return one_hot(labels, "not_enough_info")
    raise ValueError(f"row {row_id}: no evidence and no not_enough_info label in {labels}")


# ---------------------------------------------------------------------------
# JevClassifier — the teacher, live
# ---------------------------------------------------------------------------


class JevClassifier:
    """Answers rows with live Jev via ``jeff.jev.JevTeacher``.

    This is the upper bound: agreement with it is trivially ~1.0 (modulo
    sampling noise), and its ground-truth accuracy is what the student is
    trying to approach at a fraction of the cost.
    """

    def __init__(
        self,
        teacher: Any = None,
        model: str = "jev-1.13.0",
        cache_path: str | None = None,
        api_key: str | None = None,
    ) -> None:
        if teacher is None:
            try:
                from jeff.jev import JevTeacher
            except ImportError as exc:
                raise ImportError(
                    "jeff.jev is not available yet — the JevTeacher slice "
                    "has not landed. Original error: " + str(exc)
                ) from exc
            kwargs: dict[str, Any] = {"model": model}
            if cache_path is not None:
                kwargs["cache_path"] = cache_path
            if api_key is not None:
                kwargs["api_key"] = api_key
            teacher = JevTeacher(**kwargs)
        self.teacher = teacher
        self.name = model

    def predict(self, rows: Iterable[DecisionRow]) -> list[PredictionRow]:
        preds: list[PredictionRow] = []
        for row in rows:
            answer = self.teacher.ask(row.state, row.questions)
            for question_id, dist in answer.distributions.items():
                preds.append(
                    PredictionRow(
                        row_id=row.row_id,
                        question_id=question_id,
                        probs=dist,
                        confidence=answer.confidence.get(question_id),
                        model=answer.model,
                        latency_ms=answer.latency_ms,
                        meta={"usage": answer.usage},
                    )
                )
        return preds

    def spent_requests(self) -> int:
        return self.teacher.spent_requests()


# ---------------------------------------------------------------------------
# ZeroShotNLI — entailment scoring with a small MNLI cross-encoder
# ---------------------------------------------------------------------------

DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-small"


class ZeroShotNLI:
    """Zero-shot baseline: premise = concatenated evidence, hypothesis = claim.

    Label map: entailment -> supported, contradiction -> refuted,
    neutral -> not_enough_info. Rows with no evidence are short-circuited to
    not_enough_info — an NLI model cannot honestly judge a claim with an
    empty premise, and the task contract says evidence-zero rows refuse.

    Verified against transformers 5.17: ``AutoTokenizer`` /
    ``AutoModelForSequenceClassification.from_pretrained`` and
    ``config.id2label`` are unchanged from 4.x.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_NLI_MODEL,
        device: str | None = None,
        batch_size: int = 8,
        max_length: int = 512,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.model_name = model_name
        self.name = f"nli:{model_name}"
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model.to(self.device)
        self.model.eval()

        id2label = {i: label.lower() for i, label in self.model.config.id2label.items()}
        try:
            self._i_entail = next(i for i, l in id2label.items() if "entail" in l)
            self._i_contra = next(i for i, l in id2label.items() if "contradict" in l)
            self._i_neutral = next(i for i, l in id2label.items() if "neutral" in l)
        except StopIteration as exc:
            raise ValueError(
                f"{model_name} does not expose entailment/contradiction/neutral "
                f"labels; id2label={id2label}"
            ) from exc

    def predict(self, rows: Iterable[DecisionRow]) -> list[PredictionRow]:
        torch = self._torch
        rows = list(rows)
        preds: list[PredictionRow] = []
        # (row, question_id, question, premise, hypothesis); None premise = shortcut
        work: list[tuple[DecisionRow, str, Any, str | None, str]] = []
        for row in rows:
            claim, evidence = _claim_evidence(row.state, row.row_id)
            premise = _premise(evidence) if evidence else None
            for question_id, question in row.questions.items():
                if not isinstance(question, (ChoiceQuestion, NoulQuestion)):
                    raise ValueError(
                        f"row {row.row_id}: ZeroShotNLI answers choice/noul "
                        f"questions, not {question.kind}"
                    )
                work.append((row, question_id, question, premise, claim))

        t0 = time.perf_counter()
        scored: list[dict[str, float] | None] = [None] * len(work)
        pending = [i for i, (_, _, _, premise, _) in enumerate(work) if premise is not None]
        for start in range(0, len(pending), self.batch_size):
            batch_idx = pending[start : start + self.batch_size]
            pairs = [(work[i][3], work[i][4]) for i in batch_idx]
            enc = self.tokenizer(
                [p for p, _ in pairs],
                [h for _, h in pairs],
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True,
            ).to(self.device)
            with torch.no_grad():
                probs = torch.softmax(self.model(**enc).logits, dim=-1).cpu()
            for i, row_probs in zip(batch_idx, probs.tolist()):
                row, question_id, question, _, _ = work[i]
                scored[i] = _map_nli_probs(
                    question,
                    row_probs[self._i_entail],
                    row_probs[self._i_contra],
                    row_probs[self._i_neutral],
                    row.row_id,
                )
        batch_ms = (time.perf_counter() - t0) * 1000.0
        per_item_ms = batch_ms / max(len(work), 1)

        for (row, question_id, question, premise, _), probs in zip(work, scored):
            meta: dict[str, Any] = {"nli_model": self.model_name}
            if probs is None:
                probs = _no_evidence_probs(question, row.row_id)
                meta["shortcut"] = "no_evidence"
            preds.append(
                PredictionRow(
                    row_id=row.row_id,
                    question_id=question_id,
                    probs=probs,
                    confidence=max(probs.values()),
                    model=self.name,
                    latency_ms=per_item_ms,
                    meta=meta,
                )
            )
        return preds


# ---------------------------------------------------------------------------
# OptionScorerBase — jevlike / jevbetter through the lossy adapter
# ---------------------------------------------------------------------------


class OptionScorerBase:
    """Adapter for the open option-scorer packages (jevlike, jevbetter).

    Those scorers take ``{"context", "options": [...], "label": int}`` — a
    fixed menu with no instructions and no per-label definitions. The loss
    is the point of the comparison: they cannot see the label definitions
    the student is trained on, and they cannot represent a multi-question
    row (``to_option_scorer_rows`` raises on one).

    The packages are not dependencies of this repo; the import error names
    exactly what is missing.
    """

    def __init__(self, package: str = "jevlike", **kwargs: Any) -> None:
        self.package = package
        self.name = f"option-scorer:{package}"
        try:
            self._module = importlib.import_module(package)
        except ImportError as exc:
            raise ImportError(
                f"OptionScorerBase needs the optional package {package!r}, which "
                f"is not installed. Install it into this environment (e.g. "
                f"`uv pip install {package}`) or pick another baseline. Note the "
                "comparison is lossy either way: option scorers receive no "
                "instructions, no per-label definitions, and exactly one "
                "question per row."
            ) from exc
        self._scorer = self._resolve(kwargs)

    def _resolve(self, kwargs: dict[str, Any]) -> Any:
        """Find a callable scorer on the imported module."""
        mod = self._module
        for attr in ("score_options", "score"):
            fn = getattr(mod, attr, None)
            if callable(fn):
                return fn
        for attr in ("OptionScorer", "Scorer"):
            cls = getattr(mod, attr, None)
            if cls is not None:
                return cls(**kwargs)
        raise ImportError(
            f"package {self.package!r} is installed but exposes none of "
            "score_options / score / OptionScorer / Scorer — cannot adapt it. "
            f"Found: {[n for n in dir(mod) if not n.startswith('_')]}"
        )

    def _score_one(self, context: str, options: list[str]) -> Any:
        scorer = self._scorer
        if hasattr(scorer, "score"):
            return scorer.score(context, options)
        if hasattr(scorer, "predict"):
            return scorer.predict(context, options)
        return scorer(context, options)

    def predict(self, rows: Iterable[DecisionRow]) -> list[PredictionRow]:
        rows = list(rows)
        adapted = to_option_scorer_rows(rows)  # raises on multi-question rows
        by_id = {row.row_id: row for row in rows}
        preds: list[PredictionRow] = []
        for item in adapted:
            row = by_id[item["row_id"]]
            question_id, question = next(iter(row.questions.items()))
            labels = label_space(question)
            t0 = time.perf_counter()
            raw = self._score_one(item["context"], item["options"])
            latency_ms = (time.perf_counter() - t0) * 1000.0
            if isinstance(raw, int):  # scorer returned a winning index
                probs = one_hot(labels, labels[raw])
            else:
                probs = normalize(dict(zip(labels, (float(x) for x in raw))))
            preds.append(
                PredictionRow(
                    row_id=row.row_id,
                    question_id=question_id,
                    probs=probs,
                    confidence=max(probs.values()),
                    model=self.name,
                    latency_ms=latency_ms,
                    meta={"package": self.package, "lossy": True},
                )
            )
        return preds
