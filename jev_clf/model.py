"""Text-conditioned option scorer for jev_clf.

OptionScorer is a single-pass scorer, not a per-option cross-encoder: the
state is encoded ONCE with a pretrained encoder (frozen by default), each
option's (label + definition + instructions) becomes one query vector, the
options attend to the encoded context tokens with multi-head attention, and
one scalar score per option is softmaxed ACROSS the option set.

Because labels and their definitions arrive as text in the request, the
parameter count is independent of the number of options — a request with 2
options and a request with 9 use the same weights.

The encoder is loaded with plain ``AutoModel``/``AutoTokenizer`` (verified
under transformers 5.17); the sentence-transformers package is NOT a
dependency. ``save``/``load`` persist only the head weights plus the encoder
name — frozen encoder weights are never copied into the checkpoint.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from .schema import (
    ChoiceQuestion,
    NoulQuestion,
    Questions,
    Question,
    ScoreQuestion,
    label_space,
    question_to_text,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Text rendering — the model only ever sees strings
# ---------------------------------------------------------------------------

def state_to_text(state: Any) -> str:
    """Render a row's ``state`` as the flat text the encoder consumes.

    Strings pass through. The common fact-check shape
    ``{"claim": ..., "evidence": [{"title":..,"text":..}]}`` gets a readable
    rendering; anything else falls back to sorted JSON.
    """
    if isinstance(state, str):
        return state
    if isinstance(state, dict) and "claim" in state:
        lines = [f"Claim: {state['claim']}", "Evidence:"]
        evidence = state.get("evidence") or []
        if not evidence:
            lines.append("(none)")
        for i, ev in enumerate(evidence, 1):
            if isinstance(ev, dict):
                title, text = ev.get("title"), ev.get("text", "")
                lines.append(f"{i}. {title}: {text}" if title else f"{i}. {text}")
            else:
                lines.append(f"{i}. {ev}")
        rest = {k: v for k, v in state.items() if k not in ("claim", "evidence")}
        if rest:
            lines.append(json.dumps(rest, sort_keys=True))
        return "\n".join(lines)
    return json.dumps(state, sort_keys=True)


def option_texts(question: Question) -> list[str]:
    """One query text per option, in ``label_space`` order.

    Per-option text = the question's instructions plus that label's own line
    from ``question_to_text`` (``"label: definition"`` or bare ``label``).

    The label/definition lines are matched by LABEL NAME, not by splitting the
    rendered text on newlines: instructions are prose and may contain wrapped
    newlines of their own, and a line-split assumed every line after the first
    was a label — which produced 6 option texts for 3 labels on every row and
    made the head's logit width disagree with the target.
    """
    instructions = question.instructions.strip()
    out = []
    for label in label_space(question):
        definition = None
        if isinstance(question, NoulQuestion):
            definition = (question.criteria or {}).get(label)
        elif isinstance(question, ChoiceQuestion):
            definition = question.criteria.get(label)
        elif isinstance(question, ScoreQuestion):
            idx = int(label)
            definition = (
                question.criteria[idx] if idx < len(question.criteria) else None
            )
        line = f"{label}: {definition}" if definition else label
        out.append(f"{instructions}\n{line}")
    return out


# ---------------------------------------------------------------------------
# The scorer
# ---------------------------------------------------------------------------

class OptionScorer(nn.Module):
    """Single-pass text-conditioned option scorer.

    Parameters
    ----------
    encoder_name:
        Any HF checkpoint loadable by ``AutoModel`` — verified:
        ``sentence-transformers/all-MiniLM-L6-v2`` (BERT, hidden 384) and
        ``Qwen/Qwen2.5-0.5B`` (decoder, hidden 896; last_hidden_state +
        attention-mask mean pooling works the same).
    head_width:
        Width of the attention/scorer head. Context tokens and option query
        vectors are both projected to this width.
    heads:
        Attention heads in the option->context multi-head attention.
    freeze_encoder:
        Freeze the encoder (default). The head is the only trained part.
    max_length:
        Tokenizer truncation length for both state and option texts.
    """

    def __init__(
        self,
        encoder_name: str = DEFAULT_ENCODER,
        head_width: int = 256,
        heads: int = 4,
        freeze_encoder: bool = True,
        max_length: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder_name = encoder_name
        self.head_width = head_width
        self.heads = heads
        self.freeze_encoder = freeze_encoder
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        if self.tokenizer.pad_token is None:  # decoder-only encoders (Qwen)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.encoder = AutoModel.from_pretrained(encoder_name)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            self.encoder.eval()

        d_enc = self.encoder.config.hidden_size
        self.ctx_proj = nn.Linear(d_enc, head_width)
        self.q_proj = nn.Linear(d_enc, head_width)
        self.attn = nn.MultiheadAttention(
            head_width, heads, dropout=dropout, batch_first=True
        )
        self.scorer = nn.Sequential(
            nn.Linear(head_width, head_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_width, 1),
        )

        # Calibration state, fitted on val by jev_clf.train. Temperature is
        # applied inside forward; isotonic parameters are stored in
        # `calibration` for consumers that want the comparison method.
        self.temperature = 1.0
        self.calibration: dict[str, Any] = {}

    # -- encoder -----------------------------------------------------------

    def train(self, mode: bool = True) -> "OptionScorer":
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()  # a frozen encoder never enters train mode
        return self

    @property
    def device(self) -> torch.device:
        return self.ctx_proj.weight.device

    def _encode(self, texts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize + encode. Returns (last_hidden_state [B,L,d], mask [B,L])."""
        batch = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}
        if self.freeze_encoder:
            with torch.no_grad():
                out = self.encoder(**batch)
        else:
            out = self.encoder(**batch)
        # encoders may load in their native dtype (Qwen2.5 is bf16); the head
        # is fp32 — cast at the boundary
        return (
            out.last_hidden_state.to(self.ctx_proj.weight.dtype),
            batch["attention_mask"].bool(),
        )

    @staticmethod
    def _mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)

    # -- scoring -----------------------------------------------------------

    @staticmethod
    def _normalize_questions(
        states: Sequence[Any], questions: Sequence[Questions] | Questions
    ) -> list[Questions]:
        """Accept one shared Questions dict or a per-state list."""
        if isinstance(questions, dict):
            return [questions] * len(states)
        if len(questions) != len(states):
            raise ValueError(
                f"got {len(states)} states but {len(questions)} question sets"
            )
        return list(questions)

    def forward_logits(
        self,
        states: Sequence[Any],
        questions: Sequence[Questions] | Questions,
    ) -> list[dict[str, Any]]:
        """Score every (row, question) pair.

        Returns one entry per pair, in row-major then question order:
        ``{"row_index", "question_id", "labels", "logits"}`` where ``logits``
        is a 1-D tensor aligned with ``labels`` (``label_space`` order).
        """
        qseq = self._normalize_questions(states, questions)
        state_hidden, state_mask = self._encode(
            [state_to_text(s) for s in states]
        )
        ctx = self.ctx_proj(state_hidden)  # [R, Ls, W]

        pairs: list[tuple[int, str, list[str], int]] = []
        opt_texts: list[str] = []
        for row_index, qs in enumerate(qseq):
            for qid, q in qs.items():
                texts = option_texts(q)
                pairs.append((row_index, qid, label_space(q), len(texts)))
                opt_texts.extend(texts)
        if not pairs:
            return []

        opt_hidden, opt_mask = self._encode(opt_texts)
        opt_vec = self.q_proj(self._mean_pool(opt_hidden, opt_mask))  # [total, W]

        n_pairs = len(pairs)
        n_max = max(p[3] for p in pairs)
        queries = opt_vec.new_zeros(n_pairs, n_max, self.head_width)
        ctx_rows: list[int] = []
        offset = 0
        for p, (row_index, _qid, _labels, n) in enumerate(pairs):
            queries[p, :n] = opt_vec[offset : offset + n]
            offset += n
            ctx_rows.append(row_index)
        ctx_idx = torch.tensor(ctx_rows, device=ctx.device)
        ctx_p = ctx[ctx_idx]  # [P, Ls, W] — rows repeat across their questions
        key_padding_mask = ~state_mask[ctx_idx]  # [P, Ls], True = pad

        attn_out, _ = self.attn(
            queries, ctx_p, ctx_p, key_padding_mask=key_padding_mask
        )
        scores = self.scorer(attn_out).squeeze(-1)  # [P, n_max]

        return [
            {
                "row_index": row_index,
                "question_id": qid,
                "labels": labels,
                "logits": scores[p, :n],
            }
            for p, (row_index, qid, labels, n) in enumerate(pairs)
        ]

    def forward(
        self,
        states: Sequence[Any],
        questions: Sequence[Questions] | Questions,
    ) -> list[dict[str, float]]:
        """One probability dict per (row, question) pair.

        Keys are in ``label_space`` order and values sum to 1. The fitted
        temperature (1.0 until calibration runs) divides the logits.
        """
        out = []
        for pair in self.forward_logits(states, questions):
            # detach: forward() returns plain floats — callers must never end
            # up holding the autograd graph (and float() on a grad tensor warns)
            probs = torch.softmax(
                pair["logits"].detach() / self.temperature, dim=-1
            )
            out.append(
                {label: float(p) for label, p in zip(pair["labels"], probs)}
            )
        return out

    # -- persistence -------------------------------------------------------

    def save(self, dir: str | Path) -> Path:
        """Persist head weights + encoder name + config + calibration.

        The frozen encoder's weights are NOT copied — ``load`` re-downloads /
        re-uses the named HF checkpoint.
        """
        dir = Path(dir)
        dir.mkdir(parents=True, exist_ok=True)
        config = {
            "encoder_name": self.encoder_name,
            "head_width": self.head_width,
            "heads": self.heads,
            "freeze_encoder": self.freeze_encoder,
            "max_length": self.max_length,
            "temperature": self.temperature,
        }
        (dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        head = {
            k: v for k, v in self.state_dict().items() if not k.startswith("encoder.")
        }
        torch.save(head, dir / "head.pt")
        if self.calibration:
            (dir / "calibration.json").write_text(
                json.dumps(self.calibration, indent=2) + "\n"
            )
        return dir

    @classmethod
    def load(cls, dir: str | Path, device: str | torch.device | None = None) -> "OptionScorer":
        dir = Path(dir)
        config = json.loads((dir / "config.json").read_text())
        model = cls(
            encoder_name=config["encoder_name"],
            head_width=config["head_width"],
            heads=config["heads"],
            freeze_encoder=config["freeze_encoder"],
            max_length=config["max_length"],
        )
        head = torch.load(dir / "head.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(head, strict=False)
        model.temperature = config.get("temperature", 1.0)
        cal_path = dir / "calibration.json"
        if cal_path.exists():
            model.calibration = json.loads(cal_path.read_text())
        if device is not None:
            model.to(device)
        model.eval()
        return model
