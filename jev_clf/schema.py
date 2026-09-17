"""Frozen cross-slice contract for jev_clf — an independent, decision-only
fact-checking model.

A Jev-like model takes `state` plus typed questions whose answer space is
declared in natural language at call time, and returns a probability
distribution over that answer space. That is a *text-conditioned classifier*:
the labels are not fixed by training, they are named in the request.

jev_clf is our own such model, aimed at fact verification: given a claim and
the evidence passages that were retrieved for it, return a calibrated
distribution over {supported, refuted, not_enough_info}.

This module is the interface every other slice imports. Field names here are
load-bearing — if you must change one, update CONTRACTS_JEV_CLF.md in the same
commit.

Nothing in this file downloads a model, calls the network, or reads disk at
import time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Questions — the typed answer space, declared in language
# ---------------------------------------------------------------------------

QUESTION_KINDS = ("noul", "choice", "score")


class NoulQuestion(BaseModel):
    """A yes/no question. Answer space is always exactly ("yes", "no")."""

    kind: Literal["noul"] = "noul"
    instructions: str
    criteria: dict[str, str] | None = None  # optional descriptions: {"yes":..,"no":..}


class ChoiceQuestion(BaseModel):
    """Pick one of a named set. `criteria` keys ARE the labels.

    This mirrors TypeSafe's Choice, where the label set and its natural-language
    definitions arrive together in the request.
    """

    kind: Literal["choice"] = "choice"
    instructions: str
    criteria: dict[str, str | None]  # label -> its definition (None = undescribed)


class ScoreQuestion(BaseModel):
    """Ordered rubric. `criteria[i]` is the description of level i."""

    kind: Literal["score"] = "score"
    instructions: str
    criteria: list[str]


Question = Annotated[
    Union[NoulQuestion, ChoiceQuestion, ScoreQuestion], Field(discriminator="kind")
]
Questions = dict[str, Question]


def label_space(question: Question) -> list[str]:
    """The answer space of a question, in canonical order.

    Noul is the two-label case of Choice; Score's labels are its level indices
    as strings.
    """
    if isinstance(question, NoulQuestion):
        return ["yes", "no"]
    if isinstance(question, ChoiceQuestion):
        return list(question.criteria.keys())
    return [str(i) for i in range(len(question.criteria))]


def question_to_text(question: Question) -> str:
    """Render a question as the flat text a text-only encoder consumes.

    Labels and their definitions are part of the input, not just the output
    space — this is what makes the model text-conditioned rather than a fixed
    softmax head.
    """
    lines = [question.instructions]
    for label in label_space(question):
        definition = None
        if isinstance(question, NoulQuestion):
            definition = (question.criteria or {}).get(label)
        elif isinstance(question, ChoiceQuestion):
            definition = question.criteria[label]
        else:
            definition = question.criteria[int(label)]
        lines.append(f"{label}: {definition}" if definition else label)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The fact-checking task, as shipped
# ---------------------------------------------------------------------------

FACTCHECK_QUESTION_ID = "verdict"
FACTCHECK_LABELS = ("supported", "refuted", "not_enough_info")

FACTCHECK_INSTRUCTIONS = (
    "Read the claim and the evidence passages in `state`. Decide whether the "
    "evidence establishes the claim, contradicts it, or cannot decide it."
)

FACTCHECK_CRITERIA: dict[str, str] = {
    "supported": "The evidence passages together entail the claim: if they are true, the claim is true.",
    "refuted": "The evidence passages together contradict the claim: if they are true, the claim is false.",
    "not_enough_info": (
        "No evidence passages are given, or they are irrelevant or too weak to "
        "establish or contradict the claim."
    ),
}


def make_factcheck_choice(
    instructions: str = FACTCHECK_INSTRUCTIONS,
    criteria: dict[str, str | None] | None = None,
) -> ChoiceQuestion:
    """The canonical 3-way verdict question. Callers may vary the wording of
    `instructions`/`criteria` — that variation is training signal, not noise."""
    return ChoiceQuestion(
        instructions=instructions,
        criteria=dict(criteria) if criteria is not None else dict(FACTCHECK_CRITERIA),
    )


def make_factcheck_noul(instructions: str | None = None) -> NoulQuestion:
    """The two-label reduction: P(the claim is supported)."""
    return NoulQuestion(
        instructions=instructions
        or "Do the evidence passages in `state` establish that the claim is true?",
        criteria={
            "yes": "The evidence establishes the claim.",
            "no": "The evidence contradicts the claim, or does not establish it.",
        },
    )


# ---------------------------------------------------------------------------
# Data rows
# ---------------------------------------------------------------------------

SOURCES = (
    "synthetic",  # generated text with an oracle label, no model involved
    "jev_distill",  # labelled by live Jev
    "fever",
    "vitaminc",
    "scifact",
    "adversarial",
)

LABEL_SOURCES = ("ground_truth", "synthetic_oracle", "jev-1.13.0")


class DecisionRow(BaseModel):
    """One training/eval example: a state, some questions, and the target
    distribution over each question's labels."""

    row_id: str
    source: str
    split: Literal["train", "val", "test"]
    group_id: str  # rows sharing a group_id must stay in one split
    state: Any  # str, or a JSON object such as {"claim":..,"evidence":[..]}
    questions: Questions
    labels: dict[str, dict[str, float]]  # question_id -> {label: target prob}
    weight: float = 1.0
    label_source: str
    meta: dict[str, Any] = Field(default_factory=dict)

    def targets(self, question_id: str) -> dict[str, float]:
        """Normalized target distribution for one question."""
        return normalize(self.labels[question_id])


class PredictionRow(BaseModel):
    """One model answer. Every model (student, Jev, an NLI baseline) writes
    this same shape so the eval slice is model-agnostic."""

    row_id: str
    question_id: str
    probs: dict[str, float]
    confidence: float | None = None
    model: str
    latency_ms: float | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def normalize(probs: dict[str, float]) -> dict[str, float]:
    """Scale a distribution to sum to 1. All-zero input raises — that is a
    data bug, not something to paper over."""
    total = float(sum(probs.values()))
    if total <= 0:
        raise ValueError(f"cannot normalize a zero-mass distribution: {probs}")
    return {k: float(v) / total for k, v in probs.items()}


def argmax_label(probs: dict[str, float]) -> str:
    """Highest-probability label; ties break in dict insertion order."""
    return max(probs.items(), key=lambda kv: kv[1])[0]


def one_hot(labels: list[str], winner: str) -> dict[str, float]:
    if winner not in labels:
        raise ValueError(f"{winner!r} is not in the label space {labels}")
    return {label: (1.0 if label == winner else 0.0) for label in labels}


def read_rows(path: str | Path) -> list[DecisionRow]:
    with open(path) as f:
        return [DecisionRow.model_validate_json(line) for line in f if line.strip()]


def write_rows(path: str | Path, rows: list[DecisionRow]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(row.model_dump_json() + "\n")
    return len(rows)


def append_rows(path: str | Path, rows: list[DecisionRow]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for row in rows:
            f.write(row.model_dump_json() + "\n")
    return len(rows)


def write_predictions(path: str | Path, preds: list[PredictionRow]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for pred in preds:
            f.write(pred.model_dump_json() + "\n")
    return len(preds)


def read_predictions(path: str | Path) -> list[PredictionRow]:
    with open(path) as f:
        return [PredictionRow.model_validate_json(line) for line in f if line.strip()]


def to_option_scorer_rows(rows: list[DecisionRow]) -> list[dict]:
    """Adapter for the existing open option-scorers (jevlike / jevbetter).

    They take {"context", "options": [...], "label": int} — a fixed menu with no
    instructions and no per-label definitions. They therefore cannot represent a
    multi-question row, and they lose the label definitions entirely. This
    adapter exists only so those baselines can be run on comparable rows; the
    lossiness is the point of the comparison, not a bug.

    Raises if a row cannot be represented at all (more than one question).
    """
    out = []
    for row in rows:
        if len(row.questions) != 1:
            raise ValueError(
                f"row {row.row_id} has {len(row.questions)} questions; the "
                "option-scorer format holds exactly one"
            )
        question_id, question = next(iter(row.questions.items()))
        if not isinstance(question, ChoiceQuestion):
            raise ValueError(f"row {row.row_id} is not a choice question")
        labels = label_space(question)
        target = argmax_label(row.targets(question_id))
        out.append(
            {
                "context": (
                    row.state if isinstance(row.state, str) else json.dumps(row.state)
                ),
                "options": [
                    f"{label}: {question.criteria[label]}" if question.criteria[label] else label
                    for label in labels
                ],
                "label": labels.index(target),
                "row_id": row.row_id,
                "group_id": row.group_id,
                "split": row.split,
            }
        )
    return out
