"""jeff-sudoku 自检：几何/传播/搜索的回归测试 + 可选的 Jeff HTTP 联调。

    uv run python example/test_sudoku.py                 # 离线自检（不需要模型，秒级）
    uv run python example/test_sudoku.py --live          # 额外跑一次 8079 上的真实打分
    uv run python example/test_sudoku.py --live --full   # 用 Jeff（HTTP）完整解 hard 盘面

不依赖 pytest；每个 test_* 函数都可以被 pytest 直接收集。
"""

from __future__ import annotations

import argparse
import sys
import time
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence

try:  # 包内运行
    from .jeff_client import Answers, DEFAULT_SERVER, JeffError, JeffSudokuScorer, parse_board
    from .sudoku_solver import (
        PEERS,
        UNITS,
        SudokuSolver,
        compute_candidates,
        find_hidden_singles,
        is_consistent,
        is_legal,
        peers,
        validate_solution,
    )
except ImportError:  # 脚本运行
    from jeff_client import (  # type: ignore[no-redef]
        Answers,
        DEFAULT_SERVER,
        JeffError,
        JeffSudokuScorer,
        parse_board,
    )
    from sudoku_solver import (  # type: ignore[no-redef]
        PEERS,
        UNITS,
        SudokuSolver,
        compute_candidates,
        find_hidden_singles,
        is_consistent,
        is_legal,
        peers,
        validate_solution,
    )

MEDIUM = (
    "530070000"
    "600195000"
    "098000060"
    "800060003"
    "400803001"
    "700020006"
    "060000280"
    "000419005"
    "000080079"
)
# 上面这个盘面的唯一解（穷举核对过，见 test_medium_solution_is_unique）
MEDIUM_SOLUTION = (
    "534678912"
    "672195348"
    "198342567"
    "859761423"
    "426853791"
    "713924856"
    "961537284"
    "287419635"
    "345286179"
)

PUZZLES: Dict[str, str] = {
    "easy": (
        "003020600"
        "900305001"
        "001806400"
        "008102900"
        "700000008"
        "006708200"
        "002609500"
        "800203009"
        "005010300"
    ),
    "medium": MEDIUM,
    "hard": (
        "400000805"
        "030000000"
        "000700000"
        "020000060"
        "000080400"
        "000010000"
        "000603070"
        "500200000"
        "104000000"
    ),
    "evil": (
        "100007090"
        "030020008"
        "009600500"
        "005300900"
        "010080002"
        "600004000"
        "300000010"
        "040000007"
        "007000300"
    ),
}


# ---------------------------------------------------------------------------
# 独立参考实现（不复用被测代码的几何，避免同错同对）
# ---------------------------------------------------------------------------


def _ref_candidates(board: Sequence[int], cell: int) -> List[int]:
    r, c = divmod(cell, 9)
    used = set()
    for j in range(9):
        used.add(board[r * 9 + j])
        used.add(board[j * 9 + c])
    br, bc = (r // 3) * 3, (c // 3) * 3
    for i in range(3):
        for j in range(3):
            used.add(board[(br + i) * 9 + (bc + j)])
    return [v for v in range(1, 10) if v not in used]


def ref_solutions(board: Sequence[int], limit: int = 2) -> List[List[int]]:
    """独立穷举：返回最多 limit 个解。"""
    out: List[List[int]] = []

    def rec(b: List[int]) -> None:
        if len(out) >= limit:
            return
        best, best_cands = None, None
        for cell in range(81):
            if b[cell] == 0:
                cands = _ref_candidates(b, cell)
                if not cands:
                    return
                if best_cands is None or len(cands) < len(best_cands):
                    best, best_cands = cell, cands
                    if len(cands) == 1:
                        break
        if best is None:
            out.append(list(b))
            return
        for v in best_cands:
            nxt = list(b)
            nxt[best] = v
            rec(nxt)
            if len(out) >= limit:
                return

    rec(list(board))
    return out


# ---------------------------------------------------------------------------
# 打桩 scorer（不需要模型）
# ---------------------------------------------------------------------------


class StubScorer:
    """记录调用次数的最小 scorer 基类。"""

    def __init__(self) -> None:
        self.calls = 0
        self.questions = 0

    @property
    def stats(self) -> SimpleNamespace:
        return SimpleNamespace(calls=self.calls, questions=self.questions)

    def score_one_cell(self, board: Sequence[int], cell: int, values: Sequence[int]) -> Dict[int, float]:
        raise NotImplementedError


class UniformScorer(StubScorer):
    def score_one_cell(self, board, cell, values):
        self.calls += 1
        self.questions += len(values)
        return {v: 1.0 / len(values) for v in values}


class AdversarialScorer(StubScorer):
    """把真正的解压到最低分，专门考察“搜索不依赖模型”这一性质。"""

    def __init__(self, solution: Sequence[int], confident: bool = True) -> None:
        super().__init__()
        self.solution = list(solution)
        self.confident = confident

    def score_one_cell(self, board, cell, values):
        self.calls += 1
        self.questions += len(values)
        truth = self.solution[cell]
        if self.confident:
            raw = {v: (0.001 if v == truth else 1.0) for v in values}
        else:
            raw = {v: (0.5 if v == truth else 0.5) for v in values}
        total = sum(raw.values())
        return {v: p / total for v, p in raw.items()}


class FailingScorer(StubScorer):
    def score_one_cell(self, board, cell, values):
        self.calls += 1
        raise JeffError("simulated Jeff outage")


class GarbageScorer(StubScorer):
    """返回空字典 / 全 0，模拟服务返回缺字段。"""

    def score_one_cell(self, board, cell, values):
        self.calls += 1
        return {}


# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------

TESTS: List = []


def test(fn):
    TESTS.append(fn)
    return fn


@test
def test_geometry() -> None:
    assert len(UNITS) == 27
    for cell in range(81):
        p = peers(cell)
        assert cell not in p, "peers() 不能包含自己（原版的致命 bug）"
        assert len(p) == 20, (cell, len(p))
        r, c = divmod(cell, 9)
        assert set(peers(cell)) == set(PEERS[cell])
        assert peers(cell) == PEERS[cell]
        # 行/列/宫的成员数
        row = {r * 9 + j for j in range(9)}
        col = {j * 9 + c for j in range(9)}
        br, bc = (r // 3) * 3, (c // 3) * 3
        box = {(br + i) * 9 + (bc + j) for i in range(3) for j in range(3)}
        assert set(peers(cell)) == (row | col | box) - {cell}


@test
def test_is_legal_after_placing() -> None:
    """回归：落子之后 is_legal(board, cell, v) 必须为 True。"""
    board = [0] * 81
    board[0] = 5
    assert is_legal(board, 1, 7)
    assert not is_legal(board, 1, 5)
    # 原版把 cell 自己算进 peers：先落子再检查会永远 False
    board[1] = 7
    assert is_legal(board, 1, 7), "落子后的合法性检查必须通过"
    assert not is_legal(board, 1, 3)


@test
def test_medium_solution_is_unique() -> None:
    sols = ref_solutions(parse_board(MEDIUM))
    assert len(sols) == 1, f"medium 盘面应当唯一解，实际 {len(sols)} 个"
    assert "".join(map(str, sols[0])) == MEDIUM_SOLUTION


@test
def test_builtin_puzzles_are_unique() -> None:
    for name, text in PUZZLES.items():
        sols = ref_solutions(parse_board(text), limit=2)
        assert len(sols) == 1, f"{name}: 期望唯一解，实际 {len(sols)} 个"


@test
def test_solves_without_model() -> None:
    for name, text in PUZZLES.items():
        solver = SudokuSolver(None, log=None)
        board = parse_board(text)
        solution = solver.solve(board)
        assert solution is not None, f"{name}: 纯约束求解失败"
        assert validate_solution(solution), f"{name}: 解不合法"
        assert "".join(map(str, solution)) == "".join(map(str, ref_solutions(board)[0]))


@test
def test_medium_solution_string() -> None:
    solver = SudokuSolver(None)
    solution = solver.solve(parse_board(MEDIUM))
    assert "".join(map(str, solution)) == MEDIUM_SOLUTION


@test
def test_propagation_only_puzzle_needs_no_jeff() -> None:
    """只剩少数空格时应当被裸单/隐性唯一直接填完，0 次 Jeff 调用。"""
    full = parse_board(MEDIUM_SOLUTION)
    board = list(full)
    for cell in (0, 4, 8, 36, 40, 44, 72, 76, 80):  # 去掉几个角/中心
        board[cell] = 0
    scorer = UniformScorer()
    solver = SudokuSolver(scorer)
    solution = solver.solve(board)
    assert solution is not None and validate_solution(solution)
    assert solver.stats.branches == 0, solver.stats.summary()
    assert scorer.calls == 0, "纯传播能解出的盘面不应该问 Jeff"


@test
def test_uniform_scorer_still_solves() -> None:
    """所有候选都低于阈值时必须退回全尝试，不能把候选剪空。"""
    scorer = UniformScorer()
    solver = SudokuSolver(scorer, threshold=0.6, prune=True)
    solution = solver.solve(parse_board(PUZZLES["hard"]))
    assert solution is not None and validate_solution(solution), solver.stats.summary()
    assert scorer.calls >= 1, "hard 盘面应当需要问 Jeff"
    assert solver.stats.jeff_calls == scorer.calls


@test
def test_adversarial_scorer_without_prune() -> None:
    """关掉剪枝后，Jeff 排序再离谱也要解出来（只是慢）。"""
    reference = ref_solutions(parse_board(PUZZLES["hard"]))[0]
    scorer = AdversarialScorer(reference)
    solver = SudokuSolver(scorer, prune=False)
    got = solver.solve(parse_board(PUZZLES["hard"]))
    assert got is not None, "prune=False 时搜索必须完整"
    assert validate_solution(got)
    assert "".join(map(str, got)) == "".join(map(str, reference))
    assert solver.stats.branches >= 1


@test
def test_pruning_falls_back_to_complete_search() -> None:
    """模型自信但错时剪枝会丢解；默认的两遍搜索必须把它找回来。

    第二遍关掉剪枝，且打分结果全在缓存里，所以不额外产生 Jeff 调用。
    """
    reference = ref_solutions(parse_board(PUZZLES["hard"]))[0]
    scorer = AdversarialScorer(reference)
    solver = SudokuSolver(scorer, threshold=0.6, prune=True)
    got = solver.solve(parse_board(PUZZLES["hard"]))
    assert got is not None and validate_solution(got), solver.stats.summary()
    assert "".join(map(str, got)) == "".join(map(str, reference))
    assert solver.stats.prune_fallback is True
    assert solver.stats.passes == 2


@test
def test_prune_without_fallback_documents_the_loss() -> None:
    """关掉兜底就是原版语义：剪掉的候选不再尝试，允许因此找不到解。"""
    reference = ref_solutions(parse_board(PUZZLES["hard"]))[0]
    solver = SudokuSolver(
        AdversarialScorer(reference), threshold=0.6, prune=True, fallback_complete=False
    )
    got = solver.solve(parse_board(PUZZLES["hard"]))
    assert got is None or validate_solution(got)
    assert solver.stats.passes == 1


@test
def test_scorer_failure_is_survivable() -> None:
    # 1) 服务掉线/报错：必须被捕获、计数，并退化成均匀分布继续求解
    failing = SudokuSolver(FailingScorer(), prune=True)
    got = failing.solve(parse_board(PUZZLES["hard"]))
    assert got is not None and validate_solution(got), failing.stats.summary()
    assert failing.stats.scorer_errors >= 1

    # 2) 响应缺字段/全 0：走“没有可用数值 -> 均匀分布”分支，同样要继续求解
    garbage = SudokuSolver(GarbageScorer(), prune=True)
    got = garbage.solve(parse_board(PUZZLES["hard"]))
    assert got is not None and validate_solution(got), garbage.stats.summary()


@test
def test_invalid_input() -> None:
    solver = SudokuSolver(None)
    try:
        solver.solve([0] * 80)
    except ValueError:
        pass
    else:
        raise AssertionError("格子数不对时应当抛 ValueError")
    # 初始盘面冲突 -> None（不是异常）
    bad = [0] * 81
    bad[0] = 5
    bad[1] = 5
    assert solver.solve(bad) is None


@test
def test_hidden_single_helpers() -> None:
    board = parse_board(MEDIUM)
    cand = compute_candidates(board)
    hidden = find_hidden_singles(board, cand)
    for cell, v in hidden.items():
        assert board[cell] == 0 and v in cand[cell]
    assert is_consistent(board)


@test
def test_parallel_branching_matches_serial() -> None:
    board = parse_board(PUZZLES["hard"])
    serial = SudokuSolver(UniformScorer(), parallel=False).solve(board)
    parallel = SudokuSolver(UniformScorer(), parallel=True, max_workers=4).solve(board)
    assert serial is not None and parallel is not None
    assert "".join(map(str, serial)) == "".join(map(str, parallel))
    assert validate_solution(parallel)


def live_checks(server: str = DEFAULT_SERVER, full: bool = False) -> None:
    """对真实 Jeff 服务（默认 8079）做一次形状/联调检查。

    hard 盘面是“需要分支”的：传播后仍有空格可问，medium/easy 会被传播直接解掉。
    """
    board = parse_board(PUZZLES["hard"])
    propagated = SudokuSolver(None)._propagate(board)  # noqa: SLF001 - 自检脚本，允许碰内部
    assert propagated is not None
    cand = compute_candidates(propagated)
    assert cand, "传播后应当还有空格（否则没有可问的候选）"
    cell = min(cand, key=lambda c: (len(cand[c]), c))
    values = cand[cell]

    scorer = JeffSudokuScorer(transport="http", base_url=server, mode="choice")
    t0 = time.perf_counter()
    probs = scorer.score_one_cell(propagated, cell, values)
    dt = time.perf_counter() - t0
    print(f"  [live] 格子 {cell}（R{cell // 9 + 1}C{cell % 9 + 1}）候选 {values} -> "
          f"{ {v: round(p, 3) for v, p in probs.items()} }  {dt:.1f}s")
    assert set(probs) == set(values)
    assert all(0.0 <= p <= 1.0 for p in probs.values())
    assert abs(sum(probs.values()) - 1.0) < 1e-6, "choice 分布应归一"
    assert scorer.stats.readout_modes.get("first_token"), scorer.stats.readout_modes

    # noul：每个候选单独问，原始 P(yes) 原样返回（不要求和为 1）
    noul = JeffSudokuScorer(transport="http", base_url=server, mode="noul")
    probs2 = noul.score_one_cell(propagated, cell, values)
    print(f"  [live] noul 原始 P(yes) -> { {v: round(p, 4) for v, p in probs2.items()} }"
          f"  (和={sum(probs2.values()):.3f}，逐个独立问题，不必为 1)")
    assert set(probs2) == set(values)
    assert all(0.0 <= p <= 1.0 for p in probs2.values())
    noul_norm = JeffSudokuScorer(
        transport="http", base_url=server, mode="noul", noul_normalize=True
    )
    probs3 = noul_norm.score_one_cell(propagated, cell, values)
    print(f"  [live] noul 本格占比 -> { {v: round(p, 3) for v, p in probs3.items()} }")
    assert abs(sum(probs3.values()) - 1.0) < 1e-6
    assert sorted(probs3, key=lambda v: -probs3[v]) == sorted(
        probs2, key=lambda v: -probs2[v]
    ), "两种约定排序必须一致"

    if full:
        expected = "".join(map(str, ref_solutions(board)[0]))
        solver = SudokuSolver(
            JeffSudokuScorer(transport="http", base_url=server, mode="choice"),
            threshold=0.6,
        )
        t0 = time.perf_counter()
        got = solver.solve(board)
        dt = time.perf_counter() - t0
        assert got is not None, f"HTTP 模式下未能解出：{solver.stats.summary()}"
        assert "".join(map(str, got)) == expected, "解与参考实现不一致"
        print(f"  [live] 完整求解 {dt:.1f}s，{solver.stats.summary()}")


@test
def test_noul_raw_vs_normalized() -> None:
    """逐个 Noul 的原始 P(yes) 原样返回；只有显式要求才归一化。

    这是模型语义问题，不是"哪个更对"：逐个问的是彼此独立的问题，不构成分布，
    所以它们不需要和为 1。排序对单调变换不敏感，阈值比较才需要选一种约定。
    """
    class FixedNoulClient:
        def __init__(self, values):
            self.values = values
            self.calls = 0

        def system_one(self, state, questions):
            self.calls += 1
            nouls = {qid: self.values[int(qid.rsplit("_", 1)[1])] for qid in questions}
            return Answers(
                model="stub",
                nouls=nouls,
                readout_modes={qid: "first_token" for qid in questions},
                forward_passes=1,
            )

    board = parse_board(PUZZLES["hard"])
    propagated = SudokuSolver(None)._propagate(board)  # noqa: SLF001
    cell = min(compute_candidates(propagated), key=lambda c: len(compute_candidates(propagated)[c]))
    values = sorted(compute_candidates(propagated)[cell])
    raw_values = {v: 0.007 / (i + 1) for i, v in enumerate(values)}  # 全都很小、和 << 1

    raw = JeffSudokuScorer(client=FixedNoulClient(raw_values), mode="noul")
    got = raw.score_one_cell(propagated, cell, values)
    assert got == pytest_approx(raw_values), (got, raw_values)
    assert sum(got.values()) < 0.1, "原始值不应该被悄悄归一化"

    norm = JeffSudokuScorer(
        client=FixedNoulClient(raw_values), mode="noul", noul_normalize=True
    )
    got_norm = norm.score_one_cell(propagated, cell, values)
    assert abs(sum(got_norm.values()) - 1.0) < 1e-9
    assert sorted(got_norm, key=lambda v: -got_norm[v]) == sorted(
        got, key=lambda v: -got[v]
    ), "归一化是单调变换，排序必须不变"

    # 全 0（服务端缺字段）仍然兜底成均匀分布
    zero = JeffSudokuScorer(client=FixedNoulClient({v: 0.0 for v in values}), mode="noul")
    got_zero = zero.score_one_cell(propagated, cell, values)
    assert abs(sum(got_zero.values()) - 1.0) < 1e-9


def pytest_approx(d: dict) -> dict:
    return {k: round(v, 12) for k, v in d.items()}


@test
def test_parse_roundtrip() -> None:
    board = parse_board(MEDIUM)
    assert len(board) == 81 and board[0] == 5 and board[2] == 0
    assert parse_board(MEDIUM_SOLUTION) == list(map(int, MEDIUM_SOLUTION))
    assert is_consistent(parse_board(MEDIUM))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="jeff-sudoku 自检")
    parser.add_argument("--live", action="store_true", help="额外跑真实 Jeff 服务")
    parser.add_argument("--full", action="store_true", help="live 模式下完整解 medium")
    parser.add_argument("--server", default=DEFAULT_SERVER)
    args = parser.parse_args(argv)

    failures: List[str] = []
    for fn in TESTS:
        name = fn.__name__
        t0 = time.perf_counter()
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            print(f"FAIL  {name}  ({time.perf_counter() - t0:.2f}s) -> {exc}")
        else:
            print(f"ok    {name}  ({time.perf_counter() - t0:.2f}s)")

    if args.live:
        try:
            live_checks(args.server, full=args.full)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"live_checks: {type(exc).__name__}: {exc}")
            print(f"FAIL  live_checks -> {exc}")
        else:
            print("ok    live_checks")

    print()
    if failures:
        print(f"{len(failures)} 个失败:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"全部通过（{len(TESTS)} 个离线测试{' + live' if args.live else ''}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
