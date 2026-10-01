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
import os
import sys
import threading
import time
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jev_clf import schema as S  # noqa: E402
from jev_clf.client import HF_ADAPTER  # noqa: E402
from jev_clf.client import SystemOneClient  # noqa: E402

MODEL_ID = "jeff-1"
PORT = 8079
STATIC_DIR = ROOT / "results" / "static"

# Which adapter serves the DEMO.
#
# artifacts/jev_clf/lora_4b is the choice-accuracy champion, but it was trained
# on 9,119 Choice rows and ZERO Noul/Score rows, so those two primitives are
# untrained in it -- measured, its score output is dead-uniform 0.25 per level.
# lora_4b_multi adds 1,800 Score and 1,200 Noul rows. On the choice benchmark
# the two are a wash (val 0.7839 vs 0.7789, a 1-row difference, inside the
# 2-row noise floor), so the multi adapter costs nothing measurable and is the
# only one that answers Score at all. Override with JEVCLF_DEMO_{BASE,ADAPTER}.
DEFAULT_DEMO_BASE = "Qwen/Qwen3-4B-Instruct-2507"
# Use the local release adapter when it is actually present, otherwise the
# published one -- the same rule jev_clf/client.py already applies. Pointing
# PeftModel at the local path unconditionally makes it raise
# "Can't find 'adapter_config.json'" on any host that has not downloaded the
# artifact yet (the path is gitignored, so a fresh clone never has it).
_LOCAL_DEMO_ADAPTER = ROOT / "artifacts/jev_clf/lora_4b_multi"
DEFAULT_DEMO_ADAPTER = (
    str(_LOCAL_DEMO_ADAPTER) if _LOCAL_DEMO_ADAPTER.exists() else HF_ADAPTER
)

app = FastAPI(title="jev_clf", description="An independent, decision-only fact-checking model.")
_client: SystemOneClient | None = None
# Sync endpoints run in a threadpool, so two early requests can both see
# _client is None and build a model each -- ~8 GB apiece, which is an OOM on a
# 15 GB card. Build it once, under a lock.
_client_lock = threading.Lock()
# The fast tokenizer behind the model is not thread-safe: two overlapping
# requests raise "Already borrowed" from the Rust tokenizer, so the client
# serializes its own tokenizer calls (SystemOneClient._tok_lock).
#
# Forward passes are a separate question. On one GPU they contend for the same
# memory, which is why the default is still one request at a time. On CPU (or
# with the model sharded over several GPUs) set JEVCLF_SERVER_CONCURRENCY=N>1
# to let requests overlap; the tokenizer stays safe because the client holds
# its own lock.
_CONCURRENCY = max(1, int(os.environ.get("JEVCLF_SERVER_CONCURRENCY", "1")))
_infer_gate = threading.Semaphore(_CONCURRENCY)


def get_client() -> SystemOneClient:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = SystemOneClient(
                    base_model=os.environ.get("JEVCLF_DEMO_BASE", DEFAULT_DEMO_BASE),
                    adapter=os.environ.get("JEVCLF_DEMO_ADAPTER", DEFAULT_DEMO_ADAPTER),
                    # Batched readout only: prompts per forward pass. Lower it
                    # when the batch x context activations do not fit one GPU.
                    batch_size=int(os.environ.get("JEVCLF_SERVER_BATCH_SIZE", "8")),
                )
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
    batch: bool = Field(
        False,
        description=(
            "Score this request's questions with the batched readout: one "
            "forward for all first-token questions, one per label index for "
            "the sequence questions. Faster, but not bit-identical to the "
            "per-question path (see jev_clf/readout.py::distribution_batch)."
        ),
    )


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
             "context": 32768, "calibration": {"ece_test_n199": 0.063, "ece_n9730": 0.0807},
             "accuracy": {"test_n199": 0.794, "n9730": 0.8183},
             "note": "measured on human labels; Jev 1.13.0 = 0.799 / 0.8283 accuracy, ECE 0.0932 on the same rows under the same max-probability confidence definition"}
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
        with _infer_gate, torch.no_grad():
            out = client.system_one(state, questions, batch_questions=req.batch)
    except Exception as exc:
        raise HTTPException(500, f"inference failed: {exc}") from exc

    # Report the ACTUAL loaded model, not the static catalogue id: which
    # adapter is serving decides whether the Noul and Score primitives are
    # trained at all (lora_4b is Choice-only).
    body: dict = {
        "model": client.model_id,
        "device": client.device,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        "forward_passes": out.n_forward_passes,
        "batched": out.batched,
    }
    for qid in questions:
        q = questions[qid]
        # `readout` is informational: it tells a caller whether this question's
        # answer came from the first-token or the whole-sequence readout. Label
        # sets that share a first token (the score levels "0".."3") cannot use
        # the first-token readout at all.
        mode = out.readout_modes.get(qid)
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
        if mode:
            body[qid]["readout"] = mode
    return JSONResponse(body)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    from scripts.jeff_demo_page import demo_page
    return demo_page()


# --- static files ----------------------------------------------------------


if STATIC_DIR.is_dir():
    # A missing plot should never keep the server from booting.
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()
