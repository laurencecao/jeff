"""A Jev-compatible HTTP API for our local model, with a docs page.

    uv run python -m scripts.jev_clf_server        # http://127.0.0.1:8078

Endpoints:
    POST /v1/systemone   the TypeSafe contract: state + typed questions
    GET  /v1/models      model catalogue
    GET  /               the demo page: docs, calibration plot, examples
    GET  /health         liveness

The response shapes mirror the hosted service so existing code can be pointed
here by changing the base URL:
    Choice -> {choice, probabilities, confidence}
    Noul   -> {noul}
    Score  -> {score, probabilities, confidence}
"""

from __future__ import annotations

from typing import Any

import json
import sys
import time
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jev_clf import schema as S  # noqa: E402
from jev_clf.client import SystemOneClient  # noqa: E402

MODEL_ID = "jeff-1"
PORT = 8079

app = FastAPI(title="jev_clf", description="An independent, decision-only fact-checking model.")
_client: SystemOneClient | None = None


def get_client() -> SystemOneClient:
    global _client
    if _client is None:
        _client = SystemOneClient()
    return _client


# --- request schema (mirrors the TypeSafe contract) -------------------------


class QChoice(BaseModel):
    type: str = Field("choice")
    instructions: str
    criteria: dict[str, str | None]


class QNoul(BaseModel):
    type: str = Field("noul")
    instructions: str
    criteria: dict[str, str] | None = None


class QScore(BaseModel):
    type: str = Field("score")
    instructions: str
    criteria: list[str]


class SystemOneRequest(BaseModel):
    state: Any = Field(..., description="Text, a JSON object, or a message list.")
    questions: dict[str, QChoice | QNoul | QScore]


# The union above is only resolvable once all three classes exist; without this
# Pydantic defers validation and the first request fails at runtime.
SystemOneRequest.model_rebuild()


# --- endpoints --------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    return {"ok": True, "model": MODEL_ID}


@app.get("/v1/models")
def models() -> dict:
    return {
        "models": [
            {"name": MODEL_ID, "type": "choice|noul|score",
             "context": 32768, "calibration": {"ece_test_n199": 0.063, "ece_n9730": 0.0805},
             "accuracy": {"test_n199": 0.794, "n9730": 0.8174},
             "note": "measured on human labels; Jev 1.13.0 = 0.799 / 0.8283 on the same rows"}
        ]
    }


@app.post("/v1/systemone")
def systemone(req: SystemOneRequest) -> JSONResponse:
    """The TypeSafe contract: state + typed questions -> typed answers."""
    t0 = time.perf_counter()
    questions: dict[str, S.Question] = {}
    for qid, q in req.questions.items():
        kind = (q.type or "choice").lower()
        if kind == "choice":
            if not q.criteria:
                raise HTTPException(422, f"{qid}: choice requires criteria")
            questions[qid] = S.ChoiceQuestion(instructions=q.instructions, criteria=q.criteria)
        elif kind == "noul":
            questions[qid] = S.NoulQuestion(instructions=q.instructions, criteria=q.criteria)
        elif kind == "score":
            if not q.criteria:
                raise HTTPException(422, f"{qid}: score requires criteria")
            questions[qid] = S.ScoreQuestion(instructions=q.instructions, criteria=q.criteria)
        else:
            raise HTTPException(422, f"{qid}: unknown type {q.type!r}")

    if not questions:
        raise HTTPException(422, "at least one question is required")

    client = get_client()
    state = req.state
    if isinstance(state, list):
        # message lists: flatten to the text the model was trained on
        parts = [m.get("content", "") for m in state if isinstance(m, dict)]
        state = "\n".join(str(p) for p in parts)

    try:
        with torch.no_grad():
            out = client.system_one(state, questions)
    except Exception as exc:
        raise HTTPException(500, f"inference failed: {exc}") from exc

    body: dict = {"model": MODEL_ID, "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
    for qid in questions:
        q = questions[qid]
        if isinstance(q, S.NoulQuestion):
            body[qid] = {"type": "noul", "noul": round(out.nouls[qid].noul, 6)}
        elif isinstance(q, S.ScoreQuestion):
            s = out.scores[qid]
            body[qid] = {"type": "score", "score": round(s.score, 4),
                         "probabilities": {k: round(v, 6) for k, v in s.probabilities.items()},
                         "confidence": round(s.confidence, 6),
                         "criteria": list(q.criteria)}
        else:
            c = out.choices[qid]
            body[qid] = {"type": "choice", "choice": c.choice,
                         "probabilities": {k: round(v, 6) for k, v in c.probabilities.items()},
                         "confidence": round(c.confidence, 6)}
    return JSONResponse(body)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    from scripts.jeff_demo_page import demo_page
    return demo_page()



def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=PORT)


if __name__ == "__main__":
    main()
