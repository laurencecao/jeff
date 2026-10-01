"""数独求解器入口（Jeff 仓库自带的例子）。

在仓库根目录下：

    uv run python example/main.py                # 默认 medium 盘面，走 HTTP API
    uv run python example/main.py hard --mode noul
    uv run python example/main.py --transport none   # 不调用 Jeff，纯约束求解（基线）
    uv run python example/main.py --health           # 只探测 Jeff 服务
    uv run python -m example.main hard               # 以包的方式跑也行
    ./example/run.sh hard

HTTP 模式只依赖标准库（系统 python3 也能跑）；Jeff 服务用：

    uv run python -m scripts.jev_clf_server      # http://127.0.0.1:8079
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

try:  # 包内运行：python -m example.main
    from .jeff_client import (
        DEFAULT_SERVER,
        HttpJeffClient,
        JeffError,
        JeffSudokuScorer,
        parse_board,
        render_board_chars,
    )
    from .sudoku_solver import (
        SearchLimitExceeded,
        SudokuSolver,
        is_consistent,
        validate_solution,
    )
except ImportError:  # 脚本运行：python example/main.py
    from jeff_client import (  # type: ignore[no-redef]
        DEFAULT_SERVER,
        HttpJeffClient,
        JeffError,
        JeffSudokuScorer,
        parse_board,
        render_board_chars,
    )
    from sudoku_solver import (  # type: ignore[no-redef]
        SearchLimitExceeded,
        SudokuSolver,
        is_consistent,
        validate_solution,
    )

# 0 表示空格。medium 就是原 main.py 里的示例盘面。
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
    "medium": (
        "530070000"
        "600195000"
        "098000060"
        "800060003"
        "400803001"
        "700020006"
        "060000280"
        "000419005"
        "000080079"
    ),
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


def print_board(board: Sequence[int], title: Optional[str] = None) -> None:
    if title:
        print(title)
    print("    +-------+-------+-------+")
    for r in range(9):
        row = board[r * 9:(r + 1) * 9]
        cells = " ".join(str(v) if v else "." for v in row)
        print(f" {r + 1}  | {cells[0:5]} | {cells[6:11]} | {cells[12:17]} |")
        if r % 3 == 2:
            print("    +-------+-------+-------+")
    print("      1 2 3   4 5 6   7 8 9")


def load_board(args: argparse.Namespace) -> List[int]:
    if args.puzzle_file:
        return parse_board(Path(args.puzzle_file).read_text(encoding="utf-8"))
    return parse_board(PUZZLES[args.puzzle])


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="数独求解器：约束传播 + Jeff 候选打分（HTTP API）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("puzzle", nargs="?", default="medium", choices=sorted(PUZZLES),
                        help="内置盘面")
    parser.add_argument("--puzzle-file", metavar="FILE",
                        help="从文件读盘面（81 个数字或 '.'）")
    parser.add_argument("--transport", choices=["http", "local", "auto", "none"], default="http",
                        help="http=调用 Jeff 服务；local=本进程加载模型；auto=先探测 http；none=不用模型")
    parser.add_argument("--server", default=DEFAULT_SERVER, help="Jeff 服务地址")
    parser.add_argument("--batch", action=argparse.BooleanOptionalAction, default=True,
                        help="请求服务端用批量读法：一次前向打多个问题（老服务端忽略此字段）")
    parser.add_argument("--noul-normalize", action=argparse.BooleanOptionalAction, default=False,
                        help="noul 模式下把本格各候选的 P(yes) 归一化成占比（默认原样使用）")
    parser.add_argument("--mode", choices=["choice", "noul"], default="choice",
                        help="choice=每格一个问题（一次前向）；noul=每个候选值一个问题")
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="剪枝阈值；只有存在达到它的候选时才会剪掉低分候选")
    parser.add_argument("--no-prune", action="store_true",
                        help="只用 Jeff 排序，不剪枝（保证搜索完整）")
    parser.add_argument("--no-fallback", action="store_true",
                        help="剪枝无解时不要自动关闭剪枝重搜（恢复原版行为，可能丢解）")
    parser.add_argument("--parallel", action="store_true",
                        help="并行探索分支（模型调用仍是串行的，收益有限且结果不确定）")
    parser.add_argument("--workers", type=int, default=None, help="并行线程数")
    parser.add_argument("--max-nodes", type=int, default=0, help="节点预算，0 表示不限")
    parser.add_argument("--quiet", action="store_true", help="不打印分支过程")
    parser.add_argument("--health", action="store_true", help="只探测 Jeff 服务后退出")
    args = parser.parse_args(argv)

    if args.health:
        try:
            print(f"{args.server}: {HttpJeffClient(base_url=args.server).health()}")
            return 0
        except JeffError as exc:
            print(f"Jeff 服务不可用：{exc}", file=sys.stderr)
            return 1

    board = load_board(args)
    if not is_consistent(board):
        print("输入盘面自相矛盾（行/列/宫有重复数字）", file=sys.stderr)
        return 2

    print_board(board, title="初始盘面:")

    scorer = None
    if args.transport != "none":
        try:
            scorer = JeffSudokuScorer(
                transport=args.transport, base_url=args.server, mode=args.mode,
                batch=args.batch, noul_normalize=args.noul_normalize,
            )
        except JeffError as exc:
            print(f"无法连接 Jeff（{exc}）；改用 --transport none 可纯约束求解", file=sys.stderr)
            return 1
        print(f"\nJeff: {args.server}  mode={args.mode}  batch={args.batch}  "
              f"noul_normalize={args.noul_normalize}  "
              f"threshold={args.threshold}  "
              f"prune={not args.no_prune}"
              + ("" if (args.no_prune or args.no_fallback) else "  (剪枝无解会自动关剪枝重搜)"))

    solver = SudokuSolver(
        scorer,
        threshold=args.threshold,
        prune=not args.no_prune,
        fallback_complete=not args.no_fallback,
        parallel=args.parallel,
        max_workers=args.workers,
        max_nodes=args.max_nodes,
        log=None if args.quiet else print,
    )

    print(f"\n求解中（空格 {sum(1 for v in board if v == 0)} 个）...")
    t0 = time.perf_counter()
    try:
        solution = solver.solve(board)
    except SearchLimitExceeded as exc:
        print(f"\n搜索中断：{exc}")
        print(solver.stats.summary())
        return 3
    wall = time.perf_counter() - t0

    if solution is None:
        print("\n无解（或 Jeff 的打分导致所有分支死亡）")
        print(solver.stats.summary())
        return 1

    print("\n解出:")
    print_board(solution)
    print(f"\n校验: validate_solution={validate_solution(solution)}")
    print(f"解（81 字符）: {render_board_chars(solution)}")
    print(f"\n{solver.stats.summary()}")
    if scorer is not None:
        s = scorer.stats
        modes = ",".join(f"{k}x{v}" for k, v in sorted(s.readout_modes.items())) or "n/a"
        passes = f" / 服务端真实前向 {s.forward_passes}" if s.forward_passes else ""
        print(f"Jeff: {s.calls} 次调用 / {s.questions} 个问题 / 缓存命中 {s.cache_hits} / "
              f"失败 {s.errors} / 服务端累计 {s.latency_ms:.0f} ms{passes} / readout={modes}")
    print(f"墙钟时间: {wall:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
