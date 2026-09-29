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
  * mostly one forward pass PER QUESTION (we share the prompt, not the
    compute). A hosted Jev evaluates every question over the state in one
    pass; an autoregressive model cannot do that, because the answer for
    each question is a different next token. The first-token readout (labels
    that are single, distinct first tokens, e.g. choice and yes/no) takes
    exactly one pass; the sequence readout (labels sharing a first token,
    e.g. the score levels "0".."3") takes one EXTRA pass per label.
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

from jev_clf import readout  # noqa: E402
from jev_clf import schema as S  # noqa: E402
from scripts.jev_clf_lm_eval import build_inputs  # noqa: E402

DEFAULT_BASE = "Qwen/Qwen3-4B-Instruct-2507"
HF_ADAPTER = "GestaltLabs/Jeff-1"
_LOCAL_MULTI = ROOT / "artifacts/jev_clf/lora_4b_multi"
DEFAULT_ADAPTER = str(_LOCAL_MULTI) if _LOCAL_MULTI.exists() else HF_ADAPTER


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
    readout_modes: dict[str, str] = field(default_factory=dict)  # qid -> resolved readout


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
        # Load to CPU first and let .to() move the real weights. Loading with
        # dtype=bf16 and no device_map can leave a meta tensor when memory is
        # tight, and .to() then raises "Cannot copy out of meta tensor".
        #
        # With several visible GPUs, shard the model across them instead of
        # filling one card: the 4B weights are ~8 GB in bf16, which is most of
        # a 15 GB A2, and two loads racing for one card is an OOM. An explicit
        # "cuda:0" opts back out of sharding.
        self.sharded = device == "cuda" and torch.cuda.device_count() > 1
        if self.sharded:
            model = AutoModelForCausalLM.from_pretrained(
                base_model, dtype=self.dtype, device_map="auto"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model, dtype=self.dtype, device_map="cpu"
            )
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter)
        if self.sharded:
            # accelerate dispatches the shards itself, and readout.distribution
            # has to place inputs on the device that holds the embedding layer.
            self.device = str(model.device)
            self.model = model.eval()
        else:
            self.model = model.to(device).eval()
        self.model_id = f"{base_model}" + (f"+{Path(adapter).name}" if adapter else "")

    # -- one question -> a distribution over its declared labels ------------

    def _distribution(self, state: Any, question: S.Question) -> dict[str, float]:
        labels = S.label_space(question)
        text = build_inputs(self.tokenizer, state, question)
        # readout.distribution resolves the readout per question: first_token
        # for labels that are single, distinct first tokens (the benchmarked
        # path, kept verbatim); sequence for label sets that share a first
        # token, where the first-token readout degenerates.
        return readout.distribution(
            self.model,
            self.tokenizer,
            text,
            labels,
            device=self.device,
            max_length=self.max_length,
        )

    def system_one(self, state: Any, questions: dict[str, S.Question]) -> Result:
        result = Result(model=self.model_id)
        for qid, question in questions.items():
            probs = self._distribution(state, question)
            result.n_forward_passes += 1
            result.readout_modes[qid] = readout.choose_mode(
                readout.label_token_variants(self.tokenizer, S.label_space(question))
            )
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
