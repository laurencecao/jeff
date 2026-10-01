# example: 用 Jeff 给数独分支打分

Jeff 仓库自带的可运行例子：`system_one` 的候选打分当搜索启发式。**约束传播在
本地做，Jeff 只决定尝试顺序**，最终合法性由程序判定。

```
example/main.py            命令行入口（内置 easy/medium/hard/evil 四个盘面）
example/sudoku_solver.py   求解器（传播 / MV 选格 / Jeff 排序 / 剪枝 / 回溯 / 可选并行）
example/jeff_client.py     Jeff 封装（HTTP 传输为主，进程内加载为备；choice / noul 两种问法）
example/test_sudoku.py     离线自检 + 可选真实服务联调
example/run.sh             启动脚本（自动用仓库 venv）
```

## 快速开始

```bash
# 1) 起 Jeff 服务（默认就用 GPU；见下面「GPU」一节）
uv run python -m scripts.jev_clf_server          # http://127.0.0.1:8079

# 2) 求解
uv run python example/main.py hard               # 需要分支的盘面
uv run python example/main.py evil --mode noul
uv run python example/main.py medium --quiet
uv run python example/main.py --transport none   # 不用模型，纯约束求解（基线）
uv run python example/main.py --health           # 只看服务是否可用
printf '530070000600195000098000060800060003400803001700020006060000280000419005000080079' \
  > /tmp/p.txt && uv run python example/main.py --puzzle-file /tmp/p.txt

./example/run.sh hard                            # 等价写法
```

HTTP 模式只用到标准库，`python3 example/main.py` 也能跑（不需要 venv）；
`--transport local` 才需要 venv 里的 torch/jev_clf。`JEFF_SERVER` 或 `--server`
可以改服务地址。

自检：

```bash
uv run python example/test_sudoku.py                 # 16 个离线测试，秒级，不需要模型
uv run python example/test_sudoku.py --live          # 额外跑一次真实打分（choice + noul）
uv run python example/test_sudoku.py --live --full   # 用 Jeff 完整解 hard 盘面
```

## GPU

**服务端默认就用 GPU，不用额外开关**：`jev_clf/client.py` 的 `SystemOneClient`
按 `cuda → mps → cpu` 选设备，多卡时把 4B 权重用 `device_map="auto"` 分片到所有
可见显卡上。实测（本机 4×A2 15GB）：

```bash
$ nvidia-smi --query-compute-apps=pid,used_memory --format=csv
pid, used_gpu_memory [MiB]
2423612, 3360 MiB      # ← 8079 服务进程，4 张卡各放一片（bf16 共 ~9.8 GB）
2423612, 2366 MiB
2423612, 2366 MiB
2423612, 1758 MiB
```

注意三点：

* `example/` 自己**不加载模型**（HTTP 传输只发 JSON）；GPU 只影响服务进程。
  用 `--transport local` 时才会在 example 进程里再加载一份（同样会用 GPU，
  但会和正在跑的服务抢显存）。
* 单卡显存不够时不要用 `--transport local`；服务端的分片不受影响。
* 想确认服务端真在用 GPU：`curl -s localhost:8079/v1/systemone ...` 的响应里有
  `device` 字段（本次改动新增），`nvidia-smi` 里也能看到该进程的显存。

批量读法（`batch=True`）在 GPU 上的显存随 `batch × 序列长度` 增长，服务端可用
`JEVCLF_SERVER_BATCH_SIZE`（默认 8）调小。

## 修掉了什么

| # | 原代码的问题 | 后果 | 现在 |
|---|---|---|---|
| 1 | `peers()` 里 `row | col | box - {cell}` 的优先级（等价于 `row | col | (box - {cell})`），格子自己留在 peers 里 | **每次 `is_legal(new_board, cell, v)`（先落子后检查）都返回 False，所有分支被跳过，`solve()` 恒为 None** | peers 由 27 个单位导出：`frozenset(含该格的 3 个单位) - {cell}`，并断言 20 个成员且不含自己 |
| 2 | 所有候选问题共用同一段 generic instructions，格子信息塞进**共享的 state** | 模型无法知道在问哪一格哪个值，打分等于噪声 | 问题文本里写明行/列/宫与数字；state 只放棋盘 |
| 3 | 所有候选 < threshold 时全部 `continue` | 有解盘面被直接判“无解” | 只有存在 ≥ threshold 的候选时才剪枝；一个都没有就退回全尝试 |
| 4 | `result.nouls[key]` 直接下标 | 服务抖动/缺字段 → KeyError 让求解崩掉 | 包 `JeffError`；失败退化为均匀分布并计数（`scorer_errors`） |
| 5 | 没有早停的矛盾检测 | 死分支探得更久 | 传播阶段检测“某数字在本单位已无处可放” |
| 6 | `is_solved` 只看“没有 0”；`find_hidden_singles` 冲突时静默覆盖 | 可能返回冲突盘面 | 返回前 `validate_solution()`（81 格 + 行/列/宫 1-9 各一次）；隐性唯一冲突判为死分支 |
| 7 | 无 CLI、无测试 | 改一行不知道有没有坏 | `main.py` 带参数/统计；`test_sudoku.py` 用独立穷举做对照 |

## 设计要点

1. **传播优先**：裸单 → 隐性唯一 → 单位内无处可放，循环到不动点。`easy`/`medium`
   这类盘面在传播阶段就解完了，**0 次 Jeff 调用**。
2. **MV 选格**：只在传播停住时选候选最少的格子（`min(len(candidates), cell)`，
   平局用格子编号，保证可复现），只问这一格。
3. **两种问法**：
   * `--mode choice`（默认）：一格一个 `Choice` 问题，标签 `one..nine`，
     一次前向拿到整格分布。用英文数字词是因为 `" 1".." 9"` 的首 token 都是
     空格 token 220，首 token 读法会退化，只能走 sequence 读法（每标签多一次
     前向）；`" one".." nine"` 首 token 互不相同，走 first_token。
   * `--mode noul`：每个 `(格, 值)` 一个 `Noul` 问题，**原样使用它的 P(yes)**。
     这和 choice 不是"哪个更对"，而是两种不同的提问：逐个问的是彼此独立的
     问题，模型只对它被问的那个问题作答，**没有理由构成一个分布**。实测同一格
     三个候选的原始 P(yes) 加起来只有 0.02–0.04（模型对每个候选都说"否"），
     而同一个 primitive 在事实核查提示下会给 0.9997；反过来两个候选都给 0.99
     同样正常。排序只依赖大小关系（任何单调变换都不改变它），所以求解器不受
     影响；只有"本格内占比"式的阈值判断需要归一化，用 `--noul-normalize` 显式
     打开（默认关）。
4. **成本**：没有批量时 Jeff 对**每个问题**各跑一次前向，`system_one` 只是
   “一次 API 请求”。`choice` = 每格 1 个问题；`noul` = 每格候选数（最多 9）个
   问题。跑完会打印调用数、问题数、服务端 `latency_ms` 和真实前向次数。
5. **阈值语义**：`--threshold 0.6` 在 **scorer 的原始尺度**上比较，只在
   “至少一个候选达标”时剪掉低分候选；全部低于阈值时不剪枝，只按分数排序
   （两遍搜索的兜底见下条）。所以 `noul` 默认的原始 P(yes)（常在 0.01–0.04）
   实际上等价于“模型没对任何候选说 yes，那就别剪”——这正是想要的行为；
   想要“占比 ≥ 0.6 才留”的语义就加 `--noul-normalize`。
6. **两遍搜索（默认）**：剪枝本身可能因为“模型自信且错”而丢解——实测
   `hard` 盘面上真实 Jeff 就会把正确分支剪掉，8 个分支全死、3.8 s 后返回
   无解。所以默认第一遍按阈值剪枝，**没解出来就关闭剪枝再跑一遍完整搜索**；
   第二遍对同样的 `(棋盘, 格子, 候选)` 直接命中 scorer 缓存，实测**不增加
   Jeff 调用**。想恢复“剪掉就不再看”的原版行为用 `--no-fallback`；
   想完全不用剪枝用 `--no-prune`。
7. **并行**：`--parallel` 用 `ThreadPoolExecutor` 并行探分支。说实话收益有限：
   服务端默认串行化推理（`JEVCLF_SERVER_CONCURRENCY` 可放开），本模块的打分器
   也有锁，真正的瓶颈是模型本身；结果也不再确定。默认关闭。
8. **模型只影响顺序**：关掉剪枝后，Jeff 打分再离谱也只是让搜索变慢，不会让
   有解盘面失败；`test_sudoku.py` 里用“把正确解压到最低分”的 adversarial
   scorer 验证了这一点。

## 实测（本机 4×A2，模型 bf16 分片在 4 张卡上；`--quiet`）

| 盘面 | 分支点 | Jeff 调用/问题 | 服务端耗时 | 墙钟 | 结果 |
|---|---:|---:|---:|---:|---|
| `easy` | 0 | 0 | 0 | 0.00s | 传播直接解出 |
| `medium` | 0 | 0 | 0 | 0.00s | 传播直接解出 |
| `hard` | 23 | 15 | 7.2s | 7.2s | 解出，与参考实现一致；`passes=2` |
| `evil` | 16 | 10 | 4.9s | 4.9s | 解出，与参考实现一致；`passes=2` |

两个需要分支的盘面都触发了第二遍（剪枝那遍把正确候选剪掉了），说明即使
Jeff 在某一格上给出 `0.996` 这样的高置信度，也不能把搜索完整性交给它；
缓存让第二遍没有多花一次 API 调用。

### 批量读法（`--batch`，默认开）

请求里带 `"batch": true`，服务端若支持批量读法（`jev_clf/readout.py::distribution_batch`）
就把一次请求里的多个问题合并成一次前向；**老服务端会忽略这个字段**，行为不变
（实测老服务端返回 200 且结果一致）。对 sudoku 的影响：

| 场景 | 请求/问题 | 服务端真实前向 | 墙钟 |
|---|---|---|---|
| `--mode choice`（默认，每请求 1 个问题） | 15 / 15 | 15 → 15 | 7.31s → 7.30s（无作用） |
| `--mode noul`（`hard`，每格多问题） | 6 / 12 | 12 → **6** | 4.89s → 4.61s |

一次请求里问题越多、长度越接近，批量越划算（服务端另测：8 个同长 Noul
1295 ms/8 前向 → 729 ms/1 前向）。批量与逐问题的概率**不是逐位相等**，见
`jev_clf/readout.py::distribution_batch` 的说明与 `scripts/test_readout_batch.py`。

## 关于 noul 的 P(yes)：不是分布，也不需要是

`--mode noul` 对同一格的每个候选各问一次（"8 是这个格子的值吗？"），得到的是
**若干独立问题的答案**，不是一个分布：

| 格子 | 候选 | 原始 P(yes) | 和 | 真值 |
|---|---|---|---|---|
| R7C2 | 8, 9 | 0.007 / 0.004 | 0.01 | 8 |
| R1C4 | 1, 3, 9 | 0.014 / 0.018 / 0.005 | 0.04 | 3 |
| R7C2（事实核查提示下的同一个 Noul primitive） | — | 0.9997 | — | — |

和远小于 1 不代表模型"错了"，两个候选都给 0.99 也一样合理：它回答的是被问的
那个问题。对本例唯一重要的是**大小关系**，而归一化（除以同一个正数）不改变
顺序——两种约定下排序逐格一致（`test_noul_raw_vs_normalized` 与 `--live` 都在
断言这点）。差别只在阈值剪枝：原始尺度下阈值几乎不触发（保守，配合两遍搜索
仍然完整），`--noul-normalize` 则按"本格占比"剪。

## 已知局限

* Jeff 1 是事实核查分类器，数独提示对它是分布外输入，它的概率不应当被当作
  “解题能力”的度量；这也是为什么不把正确性交给它。
* 盘面越难、候选越多，API 调用越多（`--mode noul` 一格最多 9 个问题）；
  实测 4×A2 分片下 `hard` 约 7 s、`evil` 约 5 s。
