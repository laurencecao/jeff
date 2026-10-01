#!/usr/bin/env bash
# Jeff 仓库自带的数独例子。HTTP 模式只依赖标准库，但用仓库 venv 跑最省事
# （venv 里有 torch/jev_clf，可切 --transport local 在进程内直接加载模型）。
#
#   ./example/run.sh hard
#   ./example/run.sh hard --mode noul
#   ./example/run.sh hard --server http://127.0.0.1:8079
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"          # .../jeff/example
ROOT="$(cd "$HERE/.." && pwd)"                 # .../jeff

cd "$ROOT"
if command -v uv >/dev/null 2>&1 && [ -f "$ROOT/pyproject.toml" ]; then
  exec uv run python "$HERE/main.py" "$@"
fi
exec python3 "$HERE/main.py" "$@"
