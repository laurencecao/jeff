"""数独求解器：程序化约束传播 + Jeff 打分排序候选（+ 可选并行分支）。

分工
----
* **约束传播**完全在本地做：裸单（候选数只剩一个）、隐性唯一（某数字在本行/
  列/宫里只剩一个可放的位置）、以及“某数字在本单位已无处可放 → 矛盾”。
* **Jeff** 只在无法继续传播时、对**一个**格子问一次（最小剩余值选格），
  拿到的概率只用来决定尝试顺序，必要时剪枝。
* **程序化合法性检查是最终裁判**：关掉剪枝时，Jeff 打分再离谱也不会让有解
  盘面求解失败，最多是搜索慢一点。

相对原始版本修掉的关键缺陷
--------------------------
1. ``peers()`` 的运算符优先级：原式 ``row | col | box - {cell}`` 实际是
   ``row | col | (box - {cell})``，格子自己仍在 peers 里（实测 21 个成员，
   应为 20）。于是“先落子、再调用 ``is_legal(board, cell, v)``”永远返回
   False，**每个分支都被跳过，solve() 恒返回 None**。现在 peers 由 27 个单位
   统一推导：``frozenset(含该格的 3 个单位) - {cell}``。
2. 原代码把候选格信息塞进共享的 ``state``，而每个问题的 instructions 完全
   一样 —— 模型无从知道在问哪一格、哪个值。现在问题文本里写明坐标与数字
   （见 jeff_client.build_choice_question / build_noul_question）。
3. 阈值会把候选**全部**剪掉：原实现里所有候选都低于 threshold 时直接
   return None。现在保证“至少尝试一个”，且当没有任何候选达到阈值时退回
   全部尝试。
4. 剪枝本身是有代价的（模型自信且错 → 正确分支被剪掉，实测 hard 盘面在
   真实 Jeff 上就是这样）。现在默认**两遍**：第一遍按阈值剪枝，若没解出来
   就关闭剪枝再跑一遍完整搜索；第二遍直接命中 scorer 缓存，通常不额外花
   Jeff 调用。想恢复原版“剪掉就不再看”的行为用 ``fallback_complete=False``。
5. Jeff 调用失败（服务掉线、返回缺字段）不再抛 KeyError 让求解崩溃，
   而是退化为均匀分布并计数。
6. 返回前校验解（81 格、行/列/宫 1-9 各一次），避免返回“没有 0 但冲突”的盘面。
"""

from __future__ import annotations

import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

ALL_DIGITS: Tuple[int, ...] = tuple(range(1, 10))

# ---------------------------------------------------------------------------
# 几何：27 个单位（9 行 + 9 列 + 9 宫），peers 由单位导出
# ---------------------------------------------------------------------------

_ROW_UNITS: Tuple[Tuple[int, ...], ...] = tuple(
    tuple(range(r * 9, r * 9 + 9)) for r in range(9)
)
_COL_UNITS: Tuple[Tuple[int, ...], ...] = tuple(
    tuple(c + 9 * i for i in range(9)) for c in range(9)
)
_BOX_UNITS: Tuple[Tuple[int, ...], ...] = tuple(
    tuple((br * 3 + i) * 9 + (bc * 3 + j) for i in range(3) for j in range(3))
    for br in range(3)
    for bc in range(3)
)
UNITS: Tuple[Tuple[int, ...], ...] = _ROW_UNITS + _COL_UNITS + _BOX_UNITS

CELL_UNITS: Tuple[Tuple[Tuple[int, ...], ...], ...] = tuple(
    tuple(unit for unit in UNITS if cell in unit) for cell in range(81)
)
PEERS: Tuple[FrozenSet[int], ...] = tuple(
    frozenset().union(*[set(unit) for unit in CELL_UNITS[cell]]) - {cell}  # type: ignore[arg-type]
    for cell in range(81)
)
assert all(len(p) == 20 and cell not in p for cell, p in enumerate(PEERS)), "peer geometry is wrong"


# ---------------------------------------------------------------------------
# 基础工具（保留原来的函数名）
# ---------------------------------------------------------------------------


def get_row(cell: int) -> List[int]:
    return list(_ROW_UNITS[cell // 9])


def get_col(cell: int) -> List[int]:
    return list(_COL_UNITS[cell % 9])


def get_box(cell: int) -> List[int]:
    r, c = divmod(cell, 9)
    return list(_BOX_UNITS[(r // 3) * 3 + (c // 3)])


def peers(cell: int) -> FrozenSet[int]:
    """同格所在行/列/宫的其他 20 个格子（**不含自己**）。

    原实现的 ``row | col | box - {cell}`` 因优先级问题把自己也算了进来，
    使得落子后的 is_legal() 恒为 False。
    """
    return PEERS[cell]


def is_solved(board: Sequence[int]) -> bool:
    return all(v != 0 for v in board)


def is_consistent(board: Sequence[int]) -> bool:
    """检查已填数字是否有冲突。"""
    for cell in range(81):
        v = board[cell]
        if v == 0:
            continue
        for p in PEERS[cell]:
            if board[p] == v:
                return False
    return True


def is_legal(board: Sequence[int], cell: int, value: int) -> bool:
    """把 value 放进 cell 是否违反行/列/宫约束（cell 已填别的值时也为 False）。"""
    if value not in (1, 2, 3, 4, 5, 6, 7, 8, 9):
        return False
    if board[cell] not in (0, value):
        return False
    return all(board[p] != value for p in PEERS[cell])


def validate_solution(board: Sequence[int]) -> bool:
    """完整解校验：81 格填满、无冲突、每个单位的 1-9 恰好各一次。"""
    if len(board) != 81 or not is_solved(board):
        return False
    if not is_consistent(board):
        return False
    for unit in UNITS:
        if sorted(board[c] for c in unit) != list(ALL_DIGITS):
            return False
    return True


def empty_cells(board: Sequence[int]) -> List[int]:
    return [cell for cell in range(81) if board[cell] == 0]


# ---------------------------------------------------------------------------
# 约束传播
# ---------------------------------------------------------------------------


def compute_candidates(board: Sequence[int]) -> Dict[int, List[int]]:
    """为每个空格计算合法候选值（程序化，不问 Jeff）。"""
    candidates: Dict[int, List[int]] = {}
    for cell in range(81):
        if board[cell] != 0:
            continue
        used = {board[p] for p in PEERS[cell] if board[p] != 0}
        candidates[cell] = [v for v in ALL_DIGITS if v not in used]
    return candidates


def fill_naked_singles(
    board: Sequence[int], candidates: Optional[Mapping[int, Sequence[int]]] = None
) -> List[int]:
    """把候选数 == 1 的格子填上，循环到没有新的裸单为止。

    这是个便捷函数：**不做矛盾检测**，遇到无解盘面可能返回内部不一致的结果；
    求解器内部用的是 :meth:`SudokuSolver._propagate`。
    """
    board = list(board)
    cand: Optional[Mapping[int, Sequence[int]]] = candidates
    while True:
        if cand is None:
            cand = compute_candidates(board)
        singles = {cell: vals[0] for cell, vals in cand.items() if len(vals) == 1}
        if not singles:
            return board
        for cell, v in singles.items():
            board[cell] = v
        cand = None


def find_hidden_singles(
    board: Sequence[int], candidates: Mapping[int, Sequence[int]]
) -> Dict[int, int]:
    """隐性唯一：某数字在某行/列/宫中只有一个格子可填，返回 ``{cell: value}``。

    同一格被两个数字同时独占属于矛盾，这种条目会被丢掉而不是随意覆盖
    （求解器内部的 ``_propagate`` 会把这种情况判为死分支）。
    """
    fills: Dict[int, int] = {}
    conflicted: set = set()
    for unit in UNITS:
        present = {board[c] for c in unit}
        for v in ALL_DIGITS:
            if v in present:
                continue
            spots = [c for c in unit if board[c] == 0 and v in candidates.get(c, ())]
            if len(spots) == 1:
                cell = spots[0]
                if cell in fills and fills[cell] != v:
                    conflicted.add(cell)
                else:
                    fills[cell] = v
    for cell in conflicted:
        fills.pop(cell, None)
    return fills


def dead_units(board: Sequence[int], candidates: Mapping[int, Sequence[int]]) -> List[Tuple[int, int]]:
    """返回 ``[(unit_index, digit), ...]``：该数字在该单位里已无处可放。"""
    dead: List[Tuple[int, int]] = []
    for i, unit in enumerate(UNITS):
        present = {board[c] for c in unit}
        for v in ALL_DIGITS:
            if v in present:
                continue
            if not any(board[c] == 0 and v in candidates.get(c, ()) for c in unit):
                dead.append((i, v))
    return dead


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------


class SearchLimitExceeded(RuntimeError):
    """超过 ``max_nodes`` 的节点预算。"""


@dataclass
class SolveStats:
    nodes: int = 0
    branches: int = 0
    propagation_fills: int = 0
    backtracks: int = 0
    jeff_calls: int = 0
    jeff_questions: int = 0
    scorer_errors: int = 0
    max_depth: int = 0
    elapsed_s: float = 0.0
    passes: int = 0
    prune_fallback: bool = False  # 第一遍剪枝无解，靠关闭剪枝的第二遍找回
    scorer_error_messages: List[str] = field(default_factory=list)

    def summary(self) -> str:
        extra = f" passes={self.passes}" + ("(剪枝后退回完整搜索)" if self.prune_fallback else "")
        return (
            f"nodes={self.nodes} branches={self.branches} "
            f"propagation_fills={self.propagation_fills} backtracks={self.backtracks} "
            f"jeff_calls={self.jeff_calls} jeff_questions={self.jeff_questions} "
            f"scorer_errors={self.scorer_errors} depth={self.max_depth}{extra} "
            f"elapsed={self.elapsed_s:.2f}s"
        )


# ---------------------------------------------------------------------------
# 核心求解
# ---------------------------------------------------------------------------


class SudokuSolver:
    """约束传播 + Jeff 排序/剪枝 + 回溯搜索。

    Args:
        scorer: 形如 ``JeffSudokuScorer`` 的对象（只要有 ``score_one_cell(board,
            cell, values) -> {value: p}`` 即可）。``None`` 表示不使用模型，
            退化成纯约束求解器。
        threshold: 剪枝阈值，在 **scorer 返回的原始尺度** 上比较。``None`` 表示
            不做任何剪枝（只排序）。注意 ``mode="noul"`` 默认给的是逐个独立问题
            的 P(yes)（实测常在 0.01–0.04 这种量级），此时阈值实际上不会剪枝——
            这正是“模型没对任何候选说 yes 就别剪”的合理行为。想让阈值表示
            “本格内占比”，用 ``JeffSudokuScorer(mode="noul", noul_normalize=True)``。
        prune: 是否按阈值剪枝。只有在**至少一个候选 ≥ threshold** 时才真正
            剪掉低分候选；若没有任何候选达到阈值，则全部按分数顺序尝试
            （保证不会因为模型整体不自信而把一个有解盘面判死）。
        fallback_complete: 剪枝那一遍如果没解出来，自动**关闭剪枝再跑一遍**
            （完整搜索）。第二遍对同一 ``(棋盘, 格子, 候选)`` 直接命中 scorer
            缓存，所以通常不会多花 Jeff 调用；这是“用模型加速但最终仍保证
            有解必得”的关键。设成 False 就恢复成原版“剪掉就不再看”的行为。
        parallel: 是否用线程并行探索同一分支点的候选。默认关闭：本地 Jeff
            服务端用 ``_infer_lock`` 串行推理，本模块的打分器也加了锁，
            真正的串行瓶颈在模型上，并行只对远程 scorer 有意义，并且会让
            结果不再确定。
        max_workers: 并行分支的最大线程数。
        max_nodes: 节点预算（含两遍搜索），0 表示不限；超过抛 :class:`SearchLimitExceeded`。
        log: 日志回调（默认丢弃）。传 ``print`` 即可看到分支过程。
    """

    def __init__(
        self,
        scorer: Optional[object] = None,
        *,
        threshold: Optional[float] = 0.6,
        prune: bool = True,
        fallback_complete: bool = True,
        parallel: bool = False,
        max_workers: Optional[int] = None,
        max_nodes: int = 0,
        log: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.scorer = scorer
        self.threshold = threshold
        self.prune = prune and threshold is not None
        self.fallback_complete = fallback_complete
        self.parallel = parallel
        self.max_workers = max_workers
        self.max_nodes = max_nodes
        self._log_fn = log
        self.stats = SolveStats()

    # -- public API ---------------------------------------------------------

    def solve(self, board: Sequence[int]) -> Optional[List[int]]:
        """返回解出的棋盘，或 None（无解）。

        Raises:
            ValueError: 输入不是 81 格。
            SearchLimitExceeded: 超过 ``max_nodes``。
        """
        board = list(board)
        if len(board) != 81:
            raise ValueError(f"board must have 81 cells, got {len(board)}")

        self.stats = SolveStats()
        start = time.perf_counter()
        result: Optional[List[int]] = None
        if is_consistent(board) and all(v in range(10) for v in board):
            self.stats.passes = 1
            result = self._search(board, depth=0)
            if result is None and self.prune and self.fallback_complete:
                # 剪枝可能因为“模型自信且错”而丢掉正确分支；关掉剪枝再来一遍。
                # 这一遍通常不产生新的 Jeff 调用（打分结果已在缓存里）。
                self.stats.passes = 2
                self.stats.prune_fallback = True
                self._log("剪枝后无解：关闭剪枝重新搜索（完整搜索，缓存命中打分）")
                saved, self.prune = self.prune, False
                try:
                    result = self._search(board, depth=0)
                finally:
                    self.prune = saved
        self.stats.elapsed_s = time.perf_counter() - start

        if result is not None and not validate_solution(result):
            raise AssertionError("internal error: solver returned an invalid grid")
        if self.scorer is not None:
            self.stats.jeff_calls = getattr(getattr(self.scorer, "stats", None), "calls", 0)
            self.stats.jeff_questions = getattr(
                getattr(self.scorer, "stats", None), "questions", 0
            )
        return result

    # -- 搜索 ---------------------------------------------------------------

    def _search(self, board: Sequence[int], depth: int) -> Optional[List[int]]:
        self.stats.nodes += 1
        self.stats.max_depth = max(self.stats.max_depth, depth)
        if self.max_nodes and self.stats.nodes > self.max_nodes:
            raise SearchLimitExceeded(f"node budget exhausted ({self.max_nodes})")

        board = self._propagate(list(board))
        if board is None:
            return None

        candidates = compute_candidates(board)
        if not candidates:
            return board  # 已解出

        cell = min(candidates, key=lambda c: (len(candidates[c]), c))
        values = candidates[cell]
        if len(values) == 1:  # 理论上 _propagate 已经处理，保险
            board[cell] = values[0]
            return self._search(board, depth)

        self.stats.branches += 1
        ordered = self._order(board, cell, values, depth)
        if self.parallel and len(ordered) > 1:
            return self._branch_parallel(board, cell, ordered, depth)

        indent = "  " * depth
        for v in ordered:
            if not is_legal(board, cell, v):  # 候选来自 compute_candidates，正常不会发生
                continue
            self._log(f"{indent}分支 R{cell // 9 + 1}C{cell % 9 + 1} = {v}")
            child = list(board)
            child[cell] = v
            result = self._search(child, depth + 1)
            if result is not None:
                return result
            self.stats.backtracks += 1
            self._log(f"{indent}死分支 R{cell // 9 + 1}C{cell % 9 + 1} = {v}")
        return None

    def _branch_parallel(
        self, board: Sequence[int], cell: int, ordered: Sequence[int], depth: int
    ) -> Optional[List[int]]:
        workers = self.max_workers or len(ordered)
        executor = ThreadPoolExecutor(
            max_workers=max(1, min(len(ordered), workers)), thread_name_prefix="sudoku"
        )
        try:
            futures = {}
            for v in ordered:
                child = list(board)
                child[cell] = v
                futures[executor.submit(self._search, child, depth + 1)] = v
            pending = set(futures)
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    result = future.result()
                    if result is not None:
                        self._log(f"{'  ' * depth}并行分支 R{cell // 9 + 1}C{cell % 9 + 1} 命中")
                        return result
            return None
        finally:
            # 命中的分支已经拿到解，剩下的分支不再等待（正在跑的那几个会自然结束）
            executor.shutdown(wait=False, cancel_futures=True)

    def _order(self, board: Sequence[int], cell: int, values: Sequence[int], depth: int) -> List[int]:
        """用 Jeff 给候选排序（必要时剪枝）。"""
        probs = self._score(board, cell, values, depth)
        ordered = sorted(values, key=lambda v: (-probs[v], v))

        indent = "  " * depth
        pretty = " ".join(f"{v}:{probs[v]:.2f}" for v in ordered)
        self._log(
            f"{indent}分支点 R{cell // 9 + 1}C{cell % 9 + 1} 候选 {sorted(values)} -> {pretty}"
        )

        if not self.prune or self.threshold is None:
            return ordered

        confident = [v for v in ordered if probs[v] >= self.threshold]
        if not confident:
            # 关键修复：没有候选达到阈值时**不能**把候选全剪掉，否则有解盘面会被判死
            self._log(f"{indent}没有任何候选 ≥ {self.threshold}，按分数顺序全部尝试")
            return ordered
        pruned = [v for v in ordered if probs[v] < self.threshold]
        if pruned:
            self._log(
                f"{indent}剪枝 (< {self.threshold}): "
                + ", ".join(f"{v}({probs[v]:.2f})" for v in pruned)
            )
        return confident

    def _score(
        self, board: Sequence[int], cell: int, values: Sequence[int], depth: int
    ) -> Dict[int, float]:
        uniform = {v: 1.0 / len(values) for v in values}
        if self.scorer is None or len(values) <= 1:
            return {v: 1.0 for v in values} if len(values) == 1 else uniform
        try:
            raw = self.scorer.score_one_cell(board, cell, values)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 - 模型/网络问题不应该让求解崩溃
            self.stats.scorer_errors += 1
            if len(self.stats.scorer_error_messages) < 3:
                self.stats.scorer_error_messages.append(str(exc))
            self._log(f"{'  ' * depth}Jeff 打分失败（改用均匀分布）：{exc}")
            return uniform
        probs = {v: float(raw.get(v, 0.0)) for v in values}
        if sum(probs.values()) <= 0.0:
            return uniform
        # 不在这里重新归一化：量纲由 scorer 决定（choice 是分布，noul 是逐个
        # 独立问题的原始 P(yes)，两者都不该被求解器偷偷改成别的含义）。
        # 阈值比较因此是在 scorer 给出的尺度上进行的，见 _order。
        return probs

    def _propagate(self, board: List[int]) -> Optional[List[int]]:
        """约束传播到不动点；返回 None 表示死分支。"""
        while True:
            candidates = compute_candidates(board)
            if not candidates:
                return board  # 解出
            if any(not vals for vals in candidates.values()):
                return None  # 有格子没有候选

            fills: Dict[int, int] = {}
            for cell, vals in candidates.items():
                if len(vals) == 1:
                    fills[cell] = vals[0]

            # 隐性唯一 + “数字无处可放 → 矛盾”
            for unit in UNITS:
                present = {board[c] for c in unit}
                for v in ALL_DIGITS:
                    if v in present:
                        continue
                    spots = [c for c in unit if board[c] == 0 and v in candidates[c]]
                    if not spots:
                        return None
                    if len(spots) == 1:
                        cell = spots[0]
                        if fills.get(cell, v) != v:
                            return None  # 同一格被两个数字独占
                        fills[cell] = v

            if not fills:
                return board

            for cell, v in fills.items():
                if board[cell] == v:
                    continue
                if not is_legal(board, cell, v):
                    return None
                board[cell] = v
                self.stats.propagation_fills += 1

    # -- 日志 ---------------------------------------------------------------

    def _log(self, message: str) -> None:
        if self._log_fn is not None:
            self._log_fn(message)
