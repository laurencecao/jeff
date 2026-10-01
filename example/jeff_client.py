"""Jeff 批量提问封装：把数独候选问题打包成一次 system_one 调用。

传输方式（transport）
--------------------
* ``"http"``  —— 默认。POST 到本地 Jeff 服务的 ``/v1/systemone``
  （``uv run python -m scripts.jev_clf_server``，默认 ``127.0.0.1:8079``）。
  本进程不需要 torch，也不需要加载 4B 模型。
* ``"local"`` —— 在**本进程**里加载 Qwen3-4B-Instruct-2507 + Jeff LoRA
  （走 venv，能用 GPU 就自动用 GPU）。需要 torch/transformers/peft，
  只建议在 HTTP 服务不可用时用。

提问方式（mode）
----------------
* ``"choice"`` —— 每个格子一个问题，候选值是该问题的标签
  （标签用 one..nine 而不是 "1".."9"，原因见下）。一次前向即可拿到整格的
  概率分布，默认方式。
* ``"noul"``   —— 每个 ``(格子, 候选值)`` 一个问题，问题文本里写明
  “把 v 填进 R{row}C{col} 是否正确”，返回的 P(yes) **原样使用**。
  这些是彼此独立的问题，不是对候选集的一个分布，它们**不需要**和为 1
  （实测 sudoku 提示下同一格三个候选的 P(yes) 加起来只有 0.02–0.04；
  事实核查提示下同一个 Noul 却会给 0.9997）。排序只依赖大小关系，任何
  单调变换都不改变它；只有想做“本格内占比”式的阈值判断时才需要归一化，
  用 ``noul_normalize=True`` 显式打开。

为什么 Choice 的标签用英文数字词
--------------------------------
``jev_clf.readout`` 按“标签第一个 token 是否互不相同”自动选择读法：
``" 1"``、``" 2"``…… 都以同一个裸空格 token (220) 开头，首 token 读法会
退化成读同一个 logit 多次，只能走 sequence 读法（每个标签多跑一次前向）。
``" one"``、``" two"``…… 首 token 互不相同，可以走 first_token 读法：
**每个格子一次前向**。标签含义（哪个词=哪个数字）写在 criteria 定义里，
这也是 Jeff “标签及其定义在调用时给出”的文本条件化设计。

成本提醒
--------
本地客户端对**每个问题**各跑一次前向（见 ``jev_clf/client.py`` 顶部的
说明），所以“一次 system_one 调用”只是**一次 API 请求**，不是一次前向。
``mode="noul"`` 问一格最多 9 个问题 = 最多 9 次前向；``mode="choice"``
问一格 = 1 个问题 = 1 次前向。
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

DEFAULT_SERVER = os.environ.get("JEFF_SERVER", "http://127.0.0.1:8079")

# 数字 -> 首 token 互不相同的英文词（见模块 docstring）
DIGIT_WORDS: Dict[int, str] = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
}
WORD_DIGITS: Dict[str, int] = {w: d for d, w in DIGIT_WORDS.items()}


class JeffError(RuntimeError):
    """Jeff 调用失败（服务不可达、返回缺字段、推理报错等）。"""


# ---------------------------------------------------------------------------
# 轻量问题对象 —— 与 jev_clf.schema 的 primitive 同形，
# 但让 HTTP 传输不需要 import jev_clf / torch。
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChoiceQuestion:
    instructions: str
    criteria: Dict[str, str]

    kind = "choice"

    def payload(self) -> Dict[str, Any]:
        return {
            "type": "choice",
            "instructions": self.instructions,
            "criteria": dict(self.criteria),
        }


@dataclass(frozen=True)
class NoulQuestion:
    instructions: str
    criteria: Optional[Dict[str, str]] = None

    kind = "noul"

    def payload(self) -> Dict[str, Any]:
        body: Dict[str, Any] = {"type": "noul", "instructions": self.instructions}
        if self.criteria:
            body["criteria"] = dict(self.criteria)
        return body


Question = Any  # ChoiceQuestion | NoulQuestion


@dataclass
class Answers:
    """``/v1/systemone`` 的响应，三种 primitive 分开放。"""

    model: str = "unknown"
    choices: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    nouls: Dict[str, float] = field(default_factory=dict)
    scores: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    latency_ms: float = 0.0
    readout_modes: Dict[str, str] = field(default_factory=dict)
    batched: bool = False            # 服务端本请求是否用了批量读法
    forward_passes: int = 0          # 服务端真实前向次数（老服务端不返回）
    raw: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# 传输 1：HTTP（默认）
# ---------------------------------------------------------------------------


class HttpJeffClient:
    """POST ``{base_url}/v1/systemone``；与 SystemOneClient 调用形状一致。

    ``batch=True`` 会在请求里带 ``"batch": true``，让服务端用批量读法打分：
    一次请求里的所有 first_token 问题合并成一次前向（见
    jev_clf/readout.py::distribution_batch）。老版本服务端会忽略这个字段，
    行为退化成逐问题串行，不会报错。
    """

    def __init__(
        self,
        base_url: str = DEFAULT_SERVER,
        timeout: float = 900.0,
        batch: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.batch = batch

    def health(self, timeout: float = 2.0) -> Dict[str, Any]:
        url = f"{self.base_url}/health"
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - 探测用，调用方看布尔值
            raise JeffError(f"health check failed for {url}: {exc}") from exc

    def system_one(self, state: str, questions: Mapping[str, Question]) -> Answers:
        if not questions:
            return Answers(model=self.base_url)
        payload = {
            "state": state,
            "questions": {qid: q.payload() for qid, q in questions.items()},
        }
        if self.batch:
            payload["batch"] = True
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/v1/systemone",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8")[:500]
            except Exception:  # noqa: BLE001
                pass
            raise JeffError(f"HTTP {exc.code} from {self.base_url}: {detail}") from exc
        except Exception as exc:  # noqa: BLE001 - URLError / timeout / bad JSON
            raise JeffError(f"request to {self.base_url} failed: {exc}") from exc

        out = Answers(model=str(body.get("model", self.base_url)), raw=body)
        out.batched = bool(body.get("batched", False))
        out.latency_ms = float(body.get("latency_ms", (time.perf_counter() - t0) * 1000.0))
        out.forward_passes = int(body.get("forward_passes", 0) or 0)
        for qid in questions:
            ans = body.get(qid)
            if not isinstance(ans, dict):
                raise JeffError(f"response is missing question {qid!r}: {body!r}")
            kind = ans.get("type")
            if kind == "choice":
                out.choices[qid] = {
                    "choice": ans.get("choice"),
                    "probabilities": dict(ans.get("probabilities") or {}),
                    "confidence": ans.get("confidence"),
                }
            elif kind == "noul":
                if "noul" not in ans:
                    raise JeffError(f"noul answer for {qid!r} has no 'noul': {ans!r}")
                out.nouls[qid] = float(ans["noul"])
            elif kind == "score":
                out.scores[qid] = dict(ans)
            else:
                raise JeffError(f"unknown answer type {kind!r} for {qid!r}")
            if ans.get("readout"):
                out.readout_modes[qid] = str(ans["readout"])
        return out


# ---------------------------------------------------------------------------
# 传输 2：本进程加载模型（备用）
# ---------------------------------------------------------------------------


def ensure_jeff_importable() -> str:
    """把 jeff 仓库根目录放进 sys.path，返回该目录。

    ``example/`` 就在仓库里，所以仓库根 = ``parents[1]``；``JEFF_REPO`` 可以
    覆盖（例如把 example/ 拷到别处）。注意：**只有 transport="local" 需要
    jev_clf/torch**，HTTP 传输不需要这个仓库的任何依赖。
    """
    try:
        import jev_clf  # noqa: F401

        return str(Path(jev_clf.__file__).resolve().parents[1])
    except ImportError:
        pass

    here = Path(__file__).resolve()
    candidates: List[Path] = []
    env = os.environ.get("JEFF_REPO")
    if env:
        candidates.append(Path(env).expanduser())
    candidates += [here.parent.parent, here.parent]  # jeff 仓库根（example/ 的上一层）
    for path in candidates:
        if (path / "jev_clf" / "client.py").is_file():
            sys.path.insert(0, str(path))
            return str(path)
    raise JeffError(
        "cannot import jev_clf. Use the repo venv (uv run python example/main.py), "
        "set JEFF_REPO=/path/to/jeff, or use transport='http'."
    )


class LocalJeffClient:
    """在本进程里加载模型；只在 HTTP 服务不可用时用。"""

    def __init__(
        self,
        adapter: Optional[str] = "auto",
        base_model: Optional[str] = None,
        device: Optional[str] = None,
        dtype: Any = None,
        max_length: int = 2048,
    ) -> None:
        ensure_jeff_importable()
        from jev_clf.client import DEFAULT_ADAPTER  # noqa: E402
        from jev_clf.client import SystemOneClient  # noqa: E402

        if adapter == "auto":
            # 本地有 artifacts/jev_clf/lora_4b_multi 就用它，否则用发布版
            adapter = DEFAULT_ADAPTER
        kwargs: Dict[str, Any] = {"adapter": adapter, "max_length": max_length}
        if base_model:
            kwargs["base_model"] = base_model
        if device:
            kwargs["device"] = device
        if dtype is not None:
            kwargs["dtype"] = dtype
        self._client = SystemOneClient(**kwargs)

    def system_one(self, state: str, questions: Mapping[str, Question]) -> Answers:
        from jev_clf import schema as S  # noqa: E402

        typed: Dict[str, Any] = {}
        for qid, q in questions.items():
            if q.kind == "choice":
                typed[qid] = S.ChoiceQuestion(
                    instructions=q.instructions, criteria=dict(q.criteria)
                )
            elif q.kind == "noul":
                typed[qid] = S.NoulQuestion(
                    instructions=q.instructions,
                    criteria=dict(q.criteria) if q.criteria else None,
                )
            else:
                raise JeffError(f"unsupported question kind {q.kind!r}")
        res = self._client.system_one(state, typed)

        out = Answers(model=res.model)
        for qid, ans in res.choices.items():
            out.choices[qid] = {
                "choice": ans.choice,
                "probabilities": dict(ans.probabilities),
                "confidence": ans.confidence,
            }
        for qid, ans in res.nouls.items():
            out.nouls[qid] = float(ans.noul)
        for qid in res.readout_modes:
            out.readout_modes[qid] = res.readout_modes[qid]
        return out


def make_client(
    transport: str = "http",
    *,
    base_url: str = DEFAULT_SERVER,
    timeout: float = 900.0,
    batch: bool = False,
    adapter: Optional[str] = "auto",
    base_model: Optional[str] = None,
    device: Optional[str] = None,
) -> Any:
    """transport: "http" | "local" | "auto"（先探测 HTTP，失败则本地加载）。"""
    transport = transport.lower()
    if transport in ("http", "auto"):
        client = HttpJeffClient(base_url=base_url, timeout=timeout, batch=batch)
        try:
            client.health()
            return client
        except JeffError:
            if transport == "http":
                raise
            print(f"[jeff] {base_url} 不可用，回退到本进程加载模型", file=sys.stderr)
    if transport in ("local", "auto"):
        return LocalJeffClient(adapter=adapter, base_model=base_model, device=device)
    raise ValueError(f"unknown transport {transport!r}; expected http/local/auto")


# ---------------------------------------------------------------------------
# 棋盘渲染 / 问题构造
# ---------------------------------------------------------------------------


def cell_to_region(cell: int) -> Dict[str, int]:
    """cell: 0-80 -> 该格子所属行、列、宫（均为 1 起算）。"""
    r, c = divmod(cell, 9)
    box = (r // 3) * 3 + (c // 3)
    return {"row": r + 1, "col": c + 1, "box": box + 1}


def format_board(board: Sequence[int]) -> str:
    """把棋盘渲染成可读文本，作为 system_one 的 state。

    带行列号，问题文本里的 row/column 才有参照物。
    """
    lines = [
        "Sudoku grid ('.' = empty cell; rows R1-R9 top to bottom, "
        "columns C1-C9 left to right):"
    ]
    lines.append("       C1  C2  C3  C4  C5  C6  C7  C8  C9")
    for r in range(9):
        row = board[r * 9:(r + 1) * 9]
        cells = "   ".join(str(v) if v else "." for v in row)
        lines.append(f"  R{r + 1}: {cells}")
    return "\n".join(lines)


def parse_board(text: str) -> List[int]:
    """把 81 个数字/'.'/'0' 的文本（可有空白或换行）解析成棋盘。

    '.' 和 '0' 都表示空格（数独题目文本两种写法都很常见）。
    """
    board: List[int] = []
    for ch in text:
        if ch in ".0":
            board.append(0)
        elif ch.isdigit():
            v = int(ch)
            if not 1 <= v <= 9:
                raise ValueError(f"digit out of range: {ch}")
            board.append(v)
    if len(board) != 81:
        raise ValueError(f"expected 81 cells, got {len(board)}")
    return board


def render_board_chars(board: Sequence[int]) -> str:
    """紧凑的 81 字符表示（测试/对比用）。"""
    return "".join(str(v) if v else "." for v in board)


def build_question_key(cell: int, value: int) -> str:
    """问题 id，例如 R3C5_7。"""
    r, c = divmod(cell, 9)
    return f"R{r + 1}C{c + 1}_{value}"


def build_cell_key(cell: int) -> str:
    r, c = divmod(cell, 9)
    return f"R{r + 1}C{c + 1}"


def used_digits(board: Sequence[int], cell: int) -> Dict[str, List[int]]:
    """该格所在行/列/宫已经用掉的数字（写进问题文本，模型不必自己扫棋盘）。"""
    r, c = divmod(cell, 9)
    br, bc = (r // 3) * 3, (c // 3) * 3
    row = sorted({board[r * 9 + j] for j in range(9)} - {0})
    col = sorted({board[i * 9 + c] for i in range(9)} - {0})
    box = sorted(
        {board[(br + i) * 9 + (bc + j)] for i in range(3) for j in range(3)} - {0}
    )
    return {"row": row, "col": col, "box": box}


def _fmt_digits(values: Iterable[int]) -> str:
    vals = list(values)
    return ", ".join(str(v) for v in vals) if vals else "(none)"


def build_choice_question(
    board: Sequence[int], cell: int, values: Sequence[int]
) -> ChoiceQuestion:
    """整格一个问题：标签是候选值（英文数字词），返回在候选上的分布。"""
    region = cell_to_region(cell)
    used = used_digits(board, cell)
    instructions = (
        "Sudoku puzzle. The current grid is in the state.\n"
        f"The empty cell at row {region['row']}, column {region['col']} "
        f"(3x3 box {region['box']}) still has to be filled.\n"
        f"Digits already used in its row: {_fmt_digits(used['row'])}; "
        f"in its column: {_fmt_digits(used['col'])}; "
        f"in its box: {_fmt_digits(used['box'])}.\n"
        f"The digits that do not conflict with any of those are: {_fmt_digits(values)}.\n"
        "Which one of those is the correct value of that cell?"
    )
    criteria = {
        DIGIT_WORDS[int(v)]: (
            f"the digit {v}; it conflicts with no filled cell, and the rest of the "
            f"grid can still be completed with {v} at row {region['row']}, "
            f"column {region['col']}"
        )
        for v in values
    }
    return ChoiceQuestion(instructions=instructions, criteria=criteria)


def build_noul_question(board: Sequence[int], cell: int, value: int) -> NoulQuestion:
    """一个 (格子, 候选值) 一个问题，返回 P(yes)。"""
    region = cell_to_region(cell)
    instructions = (
        "Sudoku puzzle. The current grid is in the state.\n"
        f"Consider the empty cell at row {region['row']}, column {region['col']} "
        f"(3x3 box {region['box']}).\n"
        f"Is the digit {value} the correct value of that cell?\n"
        "It is already known to conflict with nothing in the row, column or box; "
        "answer yes only if it is the value that the rest of the grid requires."
    )
    criteria = {
        "yes": f"the digit {value} is the value of row {region['row']}, "
               f"column {region['col']}",
        "no": f"the digit {value} is not the value of that cell",
    }
    return NoulQuestion(instructions=instructions, criteria=criteria)


# ---------------------------------------------------------------------------
# 打分器
# ---------------------------------------------------------------------------


@dataclass
class ScorerStats:
    calls: int = 0          # system_one 调用次数
    questions: int = 0      # 问题数（本地客户端 ≈ 前向次数）
    cache_hits: int = 0
    errors: int = 0
    latency_ms: float = 0.0
    forward_passes: int = 0   # 服务端累计前向次数（批量服务端才有）
    batched_calls: int = 0    # 其中走批量读法的请求数
    readout_modes: Dict[str, int] = field(default_factory=dict)  # first_token / sequence 计数


class JeffSudokuScorer:
    """把 ``{cell: [候选值]}`` 打包成 system_one 调用，返回 ``{(cell, value): 概率}``。

    返回值的含义
    ------------
    * ``mode="choice"``：``P(该格的值 = v | 棋盘)``，同一格内和为 1（模型原始输出）。
    * ``mode="noul"``：每个候选**单独问一次**得到的 ``P(yes)``，原样返回。
      它不是一个分布，各候选之间不需要和为 1，也不该被当成“模型有多确信”
      的绝对量：那只是模型对它被问的那个问题的回答。求解器只用它排序
      （单调变换不改变排序），阈值剪枝在 ``noul_normalize=True`` 时才按
      “本格占比”理解。
    """

    def __init__(
        self,
        transport: str = "http",
        *,
        base_url: str = DEFAULT_SERVER,
        timeout: float = 900.0,
        adapter: Optional[str] = "auto",
        base_model: Optional[str] = None,
        device: Optional[str] = None,
        batch: bool = False,
        mode: str = "choice",
        noul_normalize: bool = False,
        cache: bool = True,
        max_cache_entries: int = 512,
        client: Any = None,
    ) -> None:
        if mode not in ("choice", "noul"):
            raise ValueError(f"mode must be 'choice' or 'noul', got {mode!r}")
        self.mode = mode
        # noul 的原始 P(yes) 直接返回；只有显式要求时才在格子内归一化
        self.noul_normalize = noul_normalize and mode == "noul"
        self.client = (
            client
            if client is not None
            else make_client(
                transport,
                base_url=base_url,
                timeout=timeout,
                adapter=adapter,
                base_model=base_model,
                device=device,
                batch=batch,
            )
        )
        self.stats = ScorerStats()
        self._cache: Optional[Dict[Any, Dict[int, float]]] = {} if cache else None
        self._max_cache_entries = max_cache_entries
        # 本地模型和 HTTP 服务都是单实例资源（服务端也用 _infer_lock 串行化），
        # 多线程分支共享一个 scorer 时要把调用串起来。
        self._lock = threading.Lock()

    # -- public API ---------------------------------------------------------

    def score_candidates(
        self,
        board: Sequence[int],
        candidates: Mapping[int, Sequence[int]],
    ) -> Dict[Tuple[int, int], float]:
        """board: 长度 81，0 表示空格；candidates: {cell: [合法候选值]}。

        返回 ``{(cell, value): 概率}``。单候选格不提问，直接给 1.0。
        评分失败时抛 ``JeffError``（调用方决定是否退化为均匀分布）。
        """
        clean: Dict[int, List[int]] = {}
        for cell, values in candidates.items():
            vals = sorted({int(v) for v in values})
            if vals:
                clean[int(cell)] = vals
        if not clean:
            return {}

        scores: Dict[Tuple[int, int], float] = {}
        pending: Dict[int, List[int]] = {}
        for cell, vals in clean.items():
            if len(vals) == 1:
                scores[(cell, vals[0])] = 1.0
            else:
                pending[cell] = vals
        if not pending:
            return scores

        state = format_board(board)
        todo: Dict[int, List[int]] = {}
        for cell, vals in pending.items():
            cached = self._cached(state, cell, vals)
            if cached is None:  # noqa: E501 - cache key includes the mode AND the convention
                todo[cell] = vals
            else:
                self.stats.cache_hits += 1
                for v, p in cached.items():
                    scores[(cell, v)] = p

        if todo:
            per_cell = self._ask(state, board, todo)
            for cell, vals in todo.items():
                probs = self._normalize(
                    vals, per_cell.get(cell, {}), force=self.noul_normalize
                )
                for v, p in probs.items():
                    scores[(cell, v)] = p
                self._store(state, cell, vals, probs)
        return scores

    def score_one_cell(
        self, board: Sequence[int], cell: int, values: Sequence[int]
    ) -> Dict[int, float]:
        """便捷入口：只问一个格子，返回 {value: 概率}。"""
        out = self.score_candidates(board, {cell: values})
        return {v: p for (c, v), p in out.items() if c == cell}

    def close(self) -> None:
        if hasattr(self.client, "close"):
            self.client.close()

    # -- internals ----------------------------------------------------------

    def _cached(
        self, state: str, cell: int, values: Sequence[int]
    ) -> Optional[Dict[int, float]]:
        if self._cache is None:
            return None
        return self._cache.get(
            (state, cell, tuple(values), self.mode, self.noul_normalize)
        )

    def _store(
        self, state: str, cell: int, values: Sequence[int], probs: Dict[int, float]
    ) -> None:
        if self._cache is None:
            return
        if len(self._cache) >= self._max_cache_entries:
            self._cache.clear()
        self._cache[(state, cell, tuple(values), self.mode, self.noul_normalize)] = dict(probs)

    def _ask(
        self,
        state: str,
        board: Sequence[int],
        pending: Mapping[int, List[int]],
    ) -> Dict[int, Dict[int, float]]:
        questions: Dict[str, Question] = {}
        plan: Dict[str, Tuple[int, Optional[int]]] = {}  # qid -> (cell, value 或 None)
        for cell, values in pending.items():
            if self.mode == "choice":
                qid = build_cell_key(cell)
                questions[qid] = build_choice_question(board, cell, values)
                plan[qid] = (cell, None)
            else:
                for v in values:
                    qid = build_question_key(cell, v)
                    questions[qid] = build_noul_question(board, cell, v)
                    plan[qid] = (cell, v)

        try:
            with self._lock:
                answers = self.client.system_one(state, questions)
        except JeffError:
            self.stats.errors += 1
            raise
        except Exception as exc:  # noqa: BLE001
            self.stats.errors += 1
            raise JeffError(f"scoring failed: {exc}") from exc

        self.stats.calls += 1
        self.stats.questions += len(questions)
        self.stats.latency_ms += float(getattr(answers, "latency_ms", 0.0) or 0.0)
        self.stats.forward_passes += int(getattr(answers, "forward_passes", 0) or 0)
        if getattr(answers, "batched", False):
            self.stats.batched_calls += 1
        for mode_name in (getattr(answers, "readout_modes", None) or {}).values():
            self.stats.readout_modes[str(mode_name)] = (
                self.stats.readout_modes.get(str(mode_name), 0) + 1
            )

        per_cell: Dict[int, Dict[int, float]] = {}
        if self.mode == "choice":
            for qid, (cell, _) in plan.items():
                ans = answers.choices.get(qid)
                if ans is None:
                    raise JeffError(f"missing choice answer for {qid!r}")
                probs = ans.get("probabilities") or {}
                per_cell[cell] = {
                    v: float(probs.get(DIGIT_WORDS[v], 0.0)) for v in pending[cell]
                }
        else:
            for qid, (cell, value) in plan.items():
                if qid not in answers.nouls:
                    raise JeffError(f"missing noul answer for {qid!r}")
                per_cell.setdefault(cell, {})[int(value)] = float(answers.nouls[qid])
        return per_cell

    @staticmethod
    def _normalize(
        values: Sequence[int], probs: Mapping[int, float], force: bool = False
    ) -> Dict[int, float]:
        """原样返回（默认）或归一化（``force=True``）；只有全 0 才兜底成均匀。

        归一化只是“把独立 yes/no 答案读成本格占比”的约定，不是对模型输出的
        修正：它不改变排序（除以同一个正数），只改变与固定阈值比较的含义。
        """
        vals = list(values)
        if not vals:
            return {}
        clean = {v: max(0.0, float(probs.get(v, 0.0))) for v in vals}
        total = sum(clean.values())
        if total <= 0.0:
            # 模型没给出可用数值（例如全 0），退化成均匀分布，由搜索逻辑兜底
            return {v: 1.0 / len(vals) for v in vals}
        if not force:
            return clean
        return {v: p / total for v, p in clean.items()}
