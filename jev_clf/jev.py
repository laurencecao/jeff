"""Live Jev teacher for jeff, with an append-only disk cache.

`JevTeacher` wraps `TypeSafeClient.system_one`: it converts our schema-level
`Questions` into SDK question objects, calls the model, and returns the full
per-question probability distributions (soft targets — the calibration signal
we distill from).

Every call is cached in a JSONL file keyed by sha256(model, state, questions).
A cache hit never touches the network, so a labelling run is resumable: re-run
the same command and only the missing rows are paid for.

The API key comes from `TYPESAFE_API_KEY` in the environment (or the
`api_key` argument). The client is created lazily on the first cache miss, so
a fully-cached replay works with no key and no network.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from jev_clf.schema import (
    ChoiceQuestion,
    DecisionRow,
    NoulQuestion,
    Question,
    Questions,
    ScoreQuestion,
    normalize,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
JE_CACHE = REPO_ROOT / "data" / "factcheck" / "jev_cache.jsonl"

# Errors worth retrying at this layer on top of the SDK's own RetryPolicy:
# transport, timeout, rate limit, and 5xx. Auth/bad-request/validation are
# deterministic and propagate immediately.
_TRANSIENT_ERROR_NAMES = (
    "TypeSafeAPIConnectionError",
    "TypeSafeAPITimeoutError",
    "TypeSafeRateLimitError",
    "TypeSafeInternalServerError",
)


def _transient_errors() -> tuple[type[Exception], ...]:
    import typesafe_sdk

    return tuple(
        getattr(typesafe_sdk, name)
        for name in _TRANSIENT_ERROR_NAMES
        if hasattr(typesafe_sdk, name)
    )


def _canonical(obj: Any) -> str:
    """Stable JSON for hashing: sorted keys, tight separators."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _questions_payload(questions: Questions) -> dict[str, Any]:
    return {qid: q.model_dump(mode="json") for qid, q in sorted(questions.items())}


def cache_key(model: str, state: Any, questions: Questions) -> str:
    """The identity of a teacher call: same model + state + questions."""
    payload = {
        "model": model,
        "state": state,
        "questions": _questions_payload(questions),
    }
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()


def _to_sdk_question(question: Question):
    """Convert a schema question into the SDK primitive of the same shape."""
    from typesafe_sdk import Choice, Noul, Score

    if isinstance(question, NoulQuestion):
        criteria = None
        if question.criteria:
            # Our schema names the outcomes "yes"/"no"; the SDK's NoulCriteria
            # TypedDict names them "true"/"false".
            criteria = {
                "true": question.criteria.get("yes"),
                "false": question.criteria.get("no"),
            }
        return Noul(instructions=question.instructions, criteria=criteria)
    if isinstance(question, ChoiceQuestion):
        return Choice(
            instructions=question.instructions,
            criteria=dict(question.criteria),
        )
    if isinstance(question, ScoreQuestion):
        return Score(
            instructions=question.instructions,
            criteria=list(question.criteria),
        )
    raise TypeError(f"unsupported question type: {type(question).__name__}")


@dataclass
class TeacherAnswer:
    """What one teacher call bought us.

    `distributions[qid]` is the full probability distribution over that
    question's label space — kept whole, never argmaxed. `cached` marks an
    answer replayed from disk (latency_ms is then the original call's, and no
    request was spent).
    """

    distributions: dict[str, dict[str, float]]
    confidence: dict[str, float]
    usage: dict[str, int | None]
    latency_ms: float
    model: str
    cached: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class JevTeacher:
    """Resumable live-Jev labeller.

    `ask` is the unit of work; `ask_rows` maps it over `DecisionRow`s and
    writes the distributions into `labels` (soft targets) with
    `label_source=self.model`.
    """

    def __init__(
        self,
        model: str = "jev-1.13.0",
        cache_path: str | Path = JE_CACHE,
        api_key: str | None = None,
        max_retries: int = 4,
    ):
        self.model = model
        self.cache_path = Path(cache_path)
        self._api_key = api_key
        self._max_retries = max_retries
        self._client = None
        self._spent_requests = 0
        self._cache: dict[str, dict[str, Any]] = {}
        self._load_cache()

    # -- cache -------------------------------------------------------------

    def _load_cache(self) -> None:
        """Read the append-only cache; later lines win, malformed lines skip."""
        if not self.cache_path.exists():
            return
        with open(self.cache_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    key = record["key"]
                    answer = record["answer"]
                    if isinstance(key, str) and isinstance(answer, dict):
                        self._cache[key] = answer
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue

    def _store_cache(self, key: str, answer: TeacherAnswer) -> None:
        record = {
            "key": key,
            "answer": {
                "distributions": answer.distributions,
                "confidence": answer.confidence,
                "usage": answer.usage,
                "latency_ms": answer.latency_ms,
                "model": answer.model,
            },
        }
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
        self._cache[key] = record["answer"]

    # -- client ------------------------------------------------------------

    def _get_client(self):
        """Lazily build the SDK client so cache-only replays need no key."""
        if self._client is None:
            from typesafe_sdk import RetryPolicy, TypeSafeClient

            kwargs: dict[str, Any] = {
                "model": self.model,
                "retry": RetryPolicy(max_retries=self._max_retries),
            }
            if self._api_key is not None:
                kwargs["api_key"] = self._api_key
            self._client = TypeSafeClient(**kwargs)
        return self._client

    # -- ask ---------------------------------------------------------------

    def spent_requests(self) -> int:
        """Network calls made by this instance (cache hits do not count)."""
        return self._spent_requests

    def ask(self, state: Any, questions: Questions) -> TeacherAnswer:
        """Answer `questions` over `state`, from cache or the live API."""
        key = cache_key(self.model, state, questions)
        if key in self._cache:
            cached = self._cache[key]
            return TeacherAnswer(
                distributions=cached["distributions"],
                confidence=cached["confidence"],
                usage=cached["usage"],
                latency_ms=cached["latency_ms"],
                model=cached["model"],
                cached=True,
            )

        sdk_questions = {
            qid: _to_sdk_question(q) for qid, q in questions.items()
        }
        response = self._call(state, sdk_questions)

        distributions: dict[str, dict[str, float]] = {}
        confidence: dict[str, float] = {}
        extra: dict[str, Any] = {}
        for qid, question in questions.items():
            if isinstance(question, NoulQuestion):
                answer = response.nouls[qid]
                p_yes = float(answer.noul)
                distributions[qid] = {"yes": p_yes, "no": 1.0 - p_yes}
                # Noul reports no confidence; the modal probability is the
                # honest stand-in.
                confidence[qid] = max(p_yes, 1.0 - p_yes)
            elif isinstance(question, ChoiceQuestion):
                answer = response.choices[qid]
                distributions[qid] = normalize(dict(answer.probabilities))
                confidence[qid] = float(answer.confidence)
            elif isinstance(question, ScoreQuestion):
                answer = response.scores[qid]
                distributions[qid] = normalize(
                    {str(level): p for level, p in answer.probabilities.items()}
                )
                confidence[qid] = float(answer.confidence)
                extra[qid] = {"score": float(answer.score)}
            else:
                raise TypeError(
                    f"unsupported question type: {type(question).__name__}"
                )

        usage = {
            "input_tokens": getattr(response.usage, "input_tokens", None),
            "output_tokens": getattr(response.usage, "output_tokens", None),
        }
        answer = TeacherAnswer(
            distributions=distributions,
            confidence=confidence,
            usage=usage,
            latency_ms=self._last_latency_ms,
            model=getattr(response, "model", self.model),
            cached=False,
            extra=extra,
        )
        self._store_cache(key, answer)
        return answer

    def _call(self, state: Any, sdk_questions: dict[str, Any]):
        """One system_one call with bounded retries on transient errors."""
        client = self._get_client()
        transient = _transient_errors()
        attempt = 0
        started = time.perf_counter()
        while True:
            try:
                response = client.system_one(state=state, questions=sdk_questions)
                break
            except transient:
                attempt += 1
                if attempt > self._max_retries:
                    raise
                time.sleep(min(0.5 * (2 ** (attempt - 1)), 8.0))
        self._spent_requests += 1
        self._last_latency_ms = (time.perf_counter() - started) * 1000
        return response

    # -- rows ----------------------------------------------------------------

    def ask_rows(self, rows: Iterable[DecisionRow]) -> Iterator[DecisionRow]:
        """Yield copies of `rows` labelled by the teacher (soft targets).

        Input rows are never mutated; each yielded row carries the teacher's
        full distribution in `labels` and provenance in `meta["teacher"]`.
        """
        for row in rows:
            answer = self.ask(row.state, row.questions)
            meta = dict(row.meta)
            meta["teacher"] = {
                "model": answer.model,
                "confidence": answer.confidence,
                "usage": answer.usage,
                "latency_ms": answer.latency_ms,
                "cached": answer.cached,
            }
            yield row.model_copy(
                update={
                    "labels": answer.distributions,
                    "label_source": self.model,
                    "meta": meta,
                }
            )
