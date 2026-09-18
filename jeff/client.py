"""A drop-in Jev replacement with TypeSafe's call and response shape.

The point of this module is that code written against TypeSafe can be pointed at
our local model by changing one line, because the *contract* is the same:

    client = SystemOneClient()
    result = client.system_one(state, {
        "verdict": Choice(instructions=..., criteria={...}),
        "urgent":  Noul(instructions=...),
        "quality": Score(instructions=..., criteria=["low", "mid", "high"]),
    })
    result.choices["verdict"].choice
    result.choices["verdict"].probabilities   # {label: p}, sums to 1
    result.choices["verdict"].confidence      # derived from the distribution
    result.nouls["urgent"].noul               # scalar in [0, 1], no confidence
    result.scores["quality"].score            # probability-weighted level index
    result.scores["quality"].probabilities

Differences from the hosted service, stated plainly:
  * one forward pass PER QUESTION (we share the prompt, not the compute). A
    hosted Jev evaluates every question over the state in one pass; an
    autoregressive model cannot do that, because the answer for each question
    is a different next token.
  * being text-conditioned is preserved: the label set and each label's
    definition arrive in the prompt at call time, so a differently-worded
    question or a different NUMBER of labels works without retraining.
  * `confidence` is derived from the distribution, as TypeSafe documents, and is
    their convenience statistic, not a fixed formula we are matching.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jeff import schema as S  # noqa: E402
from scripts.jeff_lm_eval import build_inputs, label_variants  # noqa: E402

DEFAULT_BASE = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_ADAPTER = str(ROOT / "artifacts/jeff/lora_4b")


# --------------------------------------------------------------------------
# Answer objects — same shape as the SDK's
# --------------------------------------------------------------------------


@dataclass
class ChoiceAnswer:
    choice: str
    probabilities: dict[str, float]
    confidence: float


@dataclass
class NoulAnswer:
    """A yes/no probability. TypeSafe's Noul answers carry NO confidence."""

    noul: float


@dataclass
class ScoreAnswer:
    score: float
    probabilities: dict[str, float]
    confidence: float


@dataclass
class Result:
    model: str
    choices: dict[str, ChoiceAnswer] = field(default_factory=dict)
    nouls: dict[str, NoulAnswer] = field(default_factory=dict)
    scores: dict[str, ScoreAnswer] = field(default_factory=dict)
    n_forward_passes: int = 0


# --------------------------------------------------------------------------
# Question constructors mirroring the SDK's ergonomics
# --------------------------------------------------------------------------


def Choice(instructions: str, criteria: dict[str, str | None]) -> S.ChoiceQuestion:
    return S.ChoiceQuestion(instructions=instructions, criteria=criteria)


def Noul(
    instructions: str, criteria: dict[str, str] | None = None
) -> S.NoulQuestion:
    return S.NoulQuestion(instructions=instructions, criteria=criteria)


def Score(instructions: str, criteria: list[str]) -> S.ScoreQuestion:
    return S.ScoreQuestion(instructions=instructions, criteria=criteria)


# --------------------------------------------------------------------------


class SystemOneClient:
    """Local stand-in for `TypeSafeClient`, same call shape."""

    def __init__(
        self,
        base_model: str = DEFAULT_BASE,
        adapter: str | None = DEFAULT_ADAPTER,
        device: str | None = None,
        dtype: Any = None,
        max_length: int = 2048,
    ) -> None:
        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        self.device = device
        self.dtype = dtype or (torch.float32 if device == "cpu" else torch.bfloat16)
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_pretrained(base_model, dtype=self.dtype)
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter)
        self.model = model.to(device).eval()
        self.model_id = f"{base_model}" + (f"+{Path(adapter).name}" if adapter else "")

    # -- one question -> a distribution over its declared labels ------------

    def _distribution(self, state: Any, question: S.Question) -> dict[str, float]:
        labels = S.label_space(question)
        variants = label_variants(self.tokenizer, labels)
        text = build_inputs(self.tokenizer, state, question)

        enc = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self.max_length
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**enc).logits[0, -1].float()
        sub = torch.tensor([logits[variants[label][0]] for label in labels])
        probs = torch.softmax(sub, dim=-1)
        return {label: float(p) for label, p in zip(labels, probs)}

    def system_one(self, state: Any, questions: dict[str, S.Question]) -> Result:
        result = Result(model=self.model_id)
        for qid, question in questions.items():
            probs = self._distribution(state, question)
            result.n_forward_passes += 1
            conf = max(probs.values())

            if isinstance(question, S.NoulQuestion):
                result.nouls[qid] = NoulAnswer(noul=probs["yes"])
            elif isinstance(question, S.ScoreQuestion):
                # probability-weighted level index, as TypeSafe documents
                score = sum(int(label) * p for label, p in probs.items())
                result.scores[qid] = ScoreAnswer(
                    score=score, probabilities=probs, confidence=conf
                )
            else:
                top = max(probs.items(), key=lambda kv: kv[1])[0]
                result.choices[qid] = ChoiceAnswer(
                    choice=top, probabilities=probs, confidence=conf
                )
        return result
