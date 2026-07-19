# FaultScope 全项目代码审查报告

> 审查日期：2026-07-17
> 审查版本：`codex/fix_bug` / `22d780d`
> 项目版本：`0.2.4`
> 环境：Darwin 24.6.0 arm64、Python 3.14.6、Rust 1.96.0
> 结论性质：逐模块静态审查、现有测试、最小复现、临时回归测试与小型性能测量的综合结果

## 1. 执行摘要

本次审查覆盖 Python 公共 API、Rust core、collection 调度与计数、PyO3 绑定、Stim 导入、两个可选 native decoder backend、命令行工具、可视化、测试及 benchmark。纳入审查的主要 Python/Rust/C++ 源文件共 116 个，约 60,711 行。

当前常规测试基线良好，上一轮报告中的大多数高严重度问题已经修复；但本次仍确认 12 个尚未解决的问题：

| 级别 | 数量 | 含义 |
|---|---:|---|
| P1 | 3 | 合法调用可静默改变结果、丢失数据或触发 native panic，应优先修复 |
| P2 | 7 | collection 语义、输入校验、构建或安装流程存在明确错误 |
| P3 | 2 | 主要影响 Rust 公共边界的健壮性和跨 backend 一致性 |

最重要的三个结论是：

1. Stim 导入器会把部分普通、合法的扁平指令误判为重复块，并永久删除中间的测量和复位操作。
2. collection 的 resume identity 没有包含详细计数配置；已完成任务可被直接复用，但调用者新要求的计数仍为空，形成静默的统计数据缺失。
3. 两个公开 hotspot 估算入口都没有验证采样状态是否记录了事件 mask：DEM 路径会越界 panic，forward 路径则把缺失事件当成“从未发生”，静默返回错误敏感度。

此外，`max_errors=0` 在不同 collection 入口得到不同 shot 数；task-level options 无法显式恢复默认值或清空继承值；resume 会丢失当前 `task_id`；数值选项接受 `bool` 和浮点数并发生截断；macOS arm64 的 `cargo test --all-targets` 仍无法链接 PyO3 backend。

## 2. 审查范围与验证基线

### 2.1 代码范围

| 区域 | 文件数 | 行数 | 主要内容 |
|---|---:|---:|---|
| `faultscope/` | 35 | 6,795 | Python API、collection、decoder、Stim、可视化 |
| `crates/` | 49 | 31,417 | Rust core、collection、PyO3 绑定及 Rust 测试 |
| `backends/` | 7 | 5,815 | fusion-blossom、PyMatching native backend |
| `tests/`、`benchmarks/` | 25 | 16,684 | Python/Rust 集成测试与性能脚本 |

审查重点包括：

- 电路、DEM、decoder 与 packed mask 的语义一致性；
- repeat、measurement record、detector/observable ID 等边界条件；
- collection 停止条件、并发提交、随机种子、resume 和 CSV 持久化；
- Rust 公共 API、PyO3/FFI 边界、panic 与资源生命周期；
- 大 repeat、大 shot、大 detector 布局及 decoder fan-out 的复杂度。

### 2.2 已执行的基线检查

| 命令 | 结果 |
|---|---|
| `.venv/bin/python -m pytest -q` | `367 passed, 1 skipped, 297 subtests passed`，22.37 秒 |
| `cargo test --workspace --all-features` | 通过；core 与 collection 的单元/公开 API 测试全部通过 |
| `.venv/bin/python -m ruff check .` | 通过 |
| `.venv/bin/python -m mypy` | 通过，共检查 37 个 Python 源文件 |
| `cargo fmt --all -- --check` | 通过 |
| `cargo clippy --workspace --all-targets --all-features -- -D warnings` | 通过 |
| `cargo test --workspace --all-targets --all-features` | **失败**，见 FS-109 |

审查期间还临时加入并运行了两个 Rust 回归测试，用于确认缺失 hotspot event mask 时的 panic 和静默错误；验证结束后已删除临时文件，工作树未保留测试脚本。

## 3. 当前已确认的正确性与工程问题

### FS-101 [P1] Stim 的扁平重复恢复会删除合法指令

**位置**

- `faultscope/io/stim.py:88-92`
- `faultscope/io/stim.py:498-552`
- `faultscope/io/stim.py:570-601`

**现象**

导入器对所有不含显式 `REPEAT` 的输入无条件调用 `_recover_flattened_repeats()`。下面是合法的 Stim 文本：

```python
from faultscope.io.stim import parse_stim_circuit

text = """MR 0
TICK
MR 0
M 0
MR 0
TICK
MR 0
M 0
"""
result = parse_stim_circuit(text)
print(result.measurement_keys)
print([op.kind for op in result.circuit.operations])
```

输入共有 6 次测量，实际结果却是：

```text
('m0', 'm1', 'm2', 'm3')
['measure_reset', 'repeat', 'measure']
```

中间的第一条独立 `M 0` 及其后的 `MR 0` 被删除。

**根因**

候选块以 `TICK` 为起点，并会在独立 `M/MX/MY` 或 `OBSERVABLE_INCLUDE` 前截断。算法随后使用第一个块的起点和最后一个截断块的终点替换整个连续区间：

```python
return instructions[:first_start] + replacement + instructions[last_end:]
```

截断块之间不属于任何块的“间隙指令”没有被复制到 replacement，因此被静默丢弃。启发式算法也没有验证恢复后的展开指令流与原始输入等价。

**影响**

- 合法 Stim 的测量 key、detector lookback、observable 和最终采样语义均可能改变；
- 不报错，且输出仍是结构合法的 `Circuit`，很难在下游察觉；
- 任意符合启发式形状的手写扁平电路都可能受影响，而不仅是某个 exporter 的输出。

**建议修复**

1. 首选删除默认启发式恢复，或者把它改成显式 opt-in 的兼容模式。
2. 若必须自动恢复，只允许替换被证明连续且完全覆盖的候选 span，逐条保留块间 gap。
3. 增加核心不变量测试：恢复结构重新展开后，指令、参数、targets、测量数和 record 引用必须与原始扁平流完全一致。
4. 加入独立 `M`、`OBSERVABLE_INCLUDE`、多段 gap、不同 block 尾部和坐标 shift 的性质测试。

### FS-102 [P1] Resume identity 忽略详细计数配置，导致请求的统计静默缺失

**位置**

- `faultscope/collection/_collect.py:171-179`
- `faultscope/collection/_collect.py:437-458`
- `crates/faultscope-collection/src/scheduler.rs:163-179`
- `crates/faultscope-collection/src/scheduler.rs:924-945`

**现象**

`CollectionRunOptions` 的以下字段会改变要生成的 counter schema，或改变这些 counter 参与停止判断的方式：

- `count_detection_events`
- `count_observable_error_combos`
- `custom_error_count_key`

但 `_strong_id()` 的 payload 只包含 source、decoder、metadata 和 postselection mask，没有包含这些字段。最小复现分两次运行同一任务：

1. 第一次按默认选项收集 4 shots，并写入 resume CSV；
2. 第二次打开 `count_detection_events=True`，仍以 `max_shots=4` resume。

实际结果：两次 `strong_id` 相同；scheduler 判断任务已经完成，不采新样本；第二次返回的 `custom_counts` 仍是 `{}`，CSV 也只有原始一行。

若历史数据只完成了一部分，问题仍存在：旧 shots 没有详细计数，新 shots 才有，最终 custom count 的分母却覆盖全部 shots，得到系统性低估。

**根因**

- run options 被单独传给 Rust collection，但没有进入任务 identity；
- resume 校验只比较 decoder 和 metadata；
- scheduler 先用历史 shots/errors 判断完成状态，不检查历史 custom-count schema 能否满足本次请求。

**影响**

- 用户明确请求的 detector event 或 observable combo 数据可能为空或只覆盖部分 shots；
- CSV 看起来可正常 resume，但不同统计 schema 被错误合并；
- `custom_error_count_key` 还可能基于不完整历史计数改变停止时机。

**建议修复**

有两种可接受方案：

1. 把详细计数 schema 纳入版本化 strong identity；或
2. 在持久化统计中记录“哪些 counter 覆盖了哪些 shots”，resume 时验证兼容性，并为缺失 counter 补采独立样本。

不应只在结果为空时补一个零值，因为零既可能表示“真实为零”，也可能表示“从未计数”。修复时应升级 identity/CSV schema 版本，并测试所有 flag 的默认、部分 resume、完成后 resume 和 flag 切换组合。

### FS-103 [P1] Hotspot 估算未验证 event mask，分别产生 panic 和静默错误

**位置**

- `crates/faultscope-core/src/dem_sampling.rs:278-307`
- `crates/faultscope-core/src/dem_sampling.rs:360-391`
- `crates/faultscope-core/src/hotspot.rs:14-43`
- `crates/faultscope-core/src/hotspot.rs:78-107`
- `crates/faultscope-core/src/packed.rs:132-164`

**DEM 路径**

`DemHotspotEstimator::run_batch(..., return_edge_events=false)` 是合法公开调用，会返回空的 `edge_event_masks`。随后把该 batch 传给公开的 `estimate_from_loss()`，`compute_dem_estimate()` 直接执行：

```rust
let event_mask = &batch.edge_event_masks[edge_index];
```

临时回归测试稳定得到：

```text
index out of bounds: the len is 0 but the index is 0
```

**Forward sampler 路径**

`FaultScopeSimulator::run_batch(..., record_events=false)` 同样返回不含 per-location event mask 的状态。`estimate_from_loss()` 不报错，而是使用一个全零 mask 代替每个缺失事件：

```rust
let event_mask = state.event_masks.get(noise_id).unwrap_or(&zero_mask);
```

使用相同电路、seed 和 1024 shots 的临时对照测试中，记录事件与不记录事件得到的 `sensitivities` 不相等；后者是静默错误结果。

**附加不变量缺口**

这两个估算入口还没有验证：

- batch/state 是否由当前 estimator/simulator 产生；
- shots 是否大于零；
- loss mask、all mask 和 event mask 的 word 宽度是否一致；
- event mask 数量是否等于 edge/noise 数量。

**建议修复**

1. 两个 `estimate_from_loss` 都返回 `NpResult<...>`。
2. 明确要求采样时记录事件；缺失时返回可诊断错误，绝不能用零 mask 代替。
3. 给 compiled program/batch 增加不可伪造或至少可校验的 layout identity，并验证 shots 与所有 mask 宽度。
4. 增加 `record_events=false`、跨 estimator state、零 shots 和不同 word width 的回归测试。

### FS-104 [P2] `max_errors=0` 在普通 collection 与 hotspot collection 中语义不同

**位置**

- `crates/faultscope-collection/src/scheduler.rs:163-179`
- `crates/faultscope-collection/src/scheduler.rs:304-380`
- `crates/faultscope-collection/src/hotspot.rs:152-167`

**复现结果**

同一确定性任务使用：

```text
max_shots=10, min_shots=0, max_errors=0, batch_size=3
```

实际结果：

```text
collect(...)          -> shots=3
collect_hotspots(...) -> shots=0
```

单任务 Rust collection API 也会返回 0 shots，因此普通 task-set scheduler 是异常路径。

**根因**

普通 scheduler 只在存在历史 stats 时调用 `task_is_complete()`。新任务直接进入 `make_task_state()`；该函数只在 `remaining_shots == 0` 时标记完成，否则至少创建一个 batch。hotspot scheduler 则显式初始化：

```rust
complete: min_shots == 0 && max_errors == Some(0)
```

**建议修复**

把“是否完成”定义为一个共享纯函数，并在空 stats、resume stats、普通、adaptive 和 hotspot 所有入口调度前统一调用。增加 `min_shots` 为 0/正数、`max_errors` 为 0/1/None 的交叉测试。

### FS-105 [P2] Task-level option overlay 无法显式恢复默认值或清空继承值

**位置**

- `faultscope/collection/_types.py:28-67`
- `faultscope/collection/_collect.py:316-353`
- `docs/superpowers/specs/2026-07-10-collection-api-design.md:26-39`

设计文档规定 task-level `collection_options` 覆盖 Collector defaults，但当前合并逻辑用“值是否等于 dataclass 默认值”猜测调用者是否显式设置：

```python
if overlay.batch_size != CollectionOptions().batch_size:
    batch_size = overlay.batch_size
if overlay.max_errors is not None:
    max_errors = overlay.max_errors
```

复现：

```python
base = CollectionOptions(
    max_shots=100, batch_size=4, max_errors=3, start_batch_size=2
)
overlay = CollectionOptions(
    batch_size=10_000, max_errors=None, start_batch_size=None
)
```

合并后仍然是 `batch_size=4, max_errors=3, start_batch_size=2`。调用者无法恢复默认 batch size，也无法关闭继承的 error limit 或 adaptive 起始值。只有 `min_shots` 已使用单独 sentinel 记录“是否显式设置”。

**建议修复**

为所有可覆盖字段使用统一 `_UNSET` sentinel/explicit mask，或定义单独的 `CollectionOptionOverrides`。需要区分三个状态：未提供、显式 `None`、显式具体值；不要以默认值推断调用意图。

### FS-106 [P2] 从 CSV resume 后返回的 `task_id` 不是当前任务标签

**位置**

- `faultscope/collection/_types.py:15-24`
- `faultscope/collection/_types.py:228-247`
- `crates/faultscope-collection/src/api.rs:41-68`
- `crates/faultscope-collection/src/scheduler.rs:163-178`

CSV schema 没有 `task_id` 字段。读取时直接令：

```python
task_id = strong_id
```

若第一次运行的任务标签为 `first-label`，第二次以同一个 strong identity、当前标签 `renamed-label` resume，且历史 shots 已达到上限，返回的 `task_id` 是 64 字符 strong hash，而不是当前标签。

Rust 已提供 `DemLogicalCollectionStats::with_identity()`，但 scheduler 复用历史 stats 时没有调用。这个问题也会影响 decoder fan-out 的展示标签和 progress message。

**建议修复**

历史数值通过校验后，应把 stats 重新绑定到当前任务的 `task_id/strong_id/decoder/metadata`，再判断是否完成。CSV 可选保存原始 label 供审计，但返回给当前调用者的 label 应来自当前 task。

### FS-107 [P2] 不存在的 `custom_error_count_key` 会静默禁用 `max_errors`

**位置**

- `faultscope/collection/_types.py:90-103`
- `crates/faultscope-collection/src/api.rs:298-305`
- `crates/faultscope-collection/src/scheduler.rs:936-945`

停止计数实现为：

```rust
stats.custom_counts.get(key).copied().unwrap_or(0)
```

因此以下情况都不会报错，而是让 error count 永远看起来为 0：

- 拼错 counter 名称；
- 指定 `detection_events`，却没有打开 `count_detection_events`；
- 指定本次配置不可能生成的 observable-combo key。

任务最终运行到 `max_shots`，与调用者要求的 `max_errors` 停止条件不符。

**建议修复**

在启动任务前验证 key 可由当前 count schema 产生；产生第一批 stats 后若 key 仍不存在，应返回错误而不是按 0 处理。动态 combo key 需要明确支持的 pattern 或专门的 stop counter 配置，不能复用任意展示 key。

### FS-108 [P2] Collection 数值选项接受 `bool`/浮点数并发生静默截断

**位置**

- `faultscope/collection/_types.py:69-87`
- `faultscope/collection/_collect.py:356-397`

`min_shots` 已严格要求非 bool 整数，但其他计数字段只做大小比较。已确认：

| 输入 | 当前行为 |
|---|---|
| `max_shots=3.9` | 构造成功，`_native_task` 用 `int()` 静默变成 3 |
| `max_shots=True` | 构造成功，任务运行 1 shot |
| `max_errors=True` | 被当成 error limit 1 |
| `batch_size=True` | 被当成 batch size 1 |
| `batch_size=1.5` | Python 构造成功，直到 PyO3 边界才以不同异常失败 |

这种行为既隐藏配置错误，也让不同字段在不同层失败。

**建议修复**

所有 shot/error/batch/worker/seed 计数字段统一要求 `isinstance(value, int) and not isinstance(value, bool)`；秒数字段要求有限正数。Python 层应在任何编译或 native 调用之前给出一致的 `TypeError`/`ValueError`。

### FS-109 [P2] macOS arm64 下 `cargo test --workspace --all-targets` 无法链接 backend

**位置**

- `backends/faultscope-fusion-blossom/Cargo.toml:12-20`
- `backends/faultscope-pymatching/Cargo.toml:12-19`

**复现**

```text
cargo test --workspace --all-targets --all-features
```

常规 workspace tests 通过，但 all-targets 在链接 `faultscope-fusion-blossom-python` 的 PyO3 `cdylib` test target 时失败，出现大量未解析 Python C API 符号，例如 `_PyBytes_AsString`。链接器还提示部分对象的 macOS deployment target 不一致。

两个 backend manifest 都设置了 `crate-type = ["cdylib"]`、`test = false` 和 PyO3 `extension-module`；当前配置不足以让 Cargo 的 all-targets 测试矩阵避开需要 Python 链接语义的 target。

**影响**

- 标准的仓库级 Rust CI 命令不能在该平台使用；
- IDE、发行前检查或下游 packager 容易误判为代码测试失败；
- 当前 CI 若只运行常规 workspace tests，会持续遗漏这个构建矩阵问题。

**建议修复**

把可单测 backend 逻辑放入普通 `rlib` crate，PyO3 `cdylib` 保持薄绑定；或者显式从纯 Cargo test matrix 排除 extension target，并由 maturin/Python integration tests 覆盖。根据 Cargo 实际失败 target 补充 `doctest=false` 或 feature gating，并在 macOS arm64 CI 固化该命令。

### FS-110 [P2] Backend 安装计划 checkout 的 revision 不参与实际安装

**位置**

- `faultscope/backends/registry.py:69-97`
- `faultscope/backends/registry.py:241-281`
- `faultscope/backends/__main__.py:64-109`

`native_decoder_install_plan()` 对 PyMatching/fusion-blossom 执行：

1. clone 对应的上游 solver 仓库；
2. checkout 用户通过 `--rev` 指定的 revision；
3. 执行 `python -m pip install --upgrade faultscope-<backend>`。

最后的 pip 命令安装的是 package index 上的 FaultScope backend 包，既不引用 checkout 路径，也不把选择的 revision 传给构建。两个 catalog URL 还是上游 solver 仓库，而不是包含 FaultScope plugin package 的源码仓库。

**影响**

- `--rev` 对最终安装的 backend 二进制没有作用，却向用户显示已选择该 revision；
- 无谓消耗网络、磁盘和时间；
- 错误信息把 package 安装失败与无关 checkout revision 联系起来，增加诊断难度。

**建议修复**

若安装发布 wheel，应移除 checkout、`target-dir` 和 `--rev`；若要支持源码/指定 revision 安装，catalog 必须指向真实 plugin source，并执行 `pip install <checkout>`，同时把上游 solver revision作为明确的 backend build 参数或 lockfile 信息。

### FS-111 [P3] Graphlike weight 未在 core 边界校验，两个 backend 对它的语义不同

**位置**

- `crates/faultscope-core/src/dem_problem.rs:153-179`
- `crates/faultscope-core/src/dem_problem.rs:286-362`
- `backends/faultscope-pymatching/src/lib.rs:1148-1179`
- `backends/faultscope-fusion-blossom/src/lib.rs:936-946`

`GraphlikeDecodingProblem::new()` 验证 probability 和索引，但原样保存调用者提供的 `edge.weight`，因此 `NaN`/无穷值可以成功构造 core problem。PyMatching backend 会在后续缩放阶段拒绝非有限结果；fusion-blossom 的 builder 参数名为 `_weight`，完全忽略该字段并根据 probability 构建分组。

由 DEM 正常编译出的 weight 与 probability 一致，所以主要受影响的是 Rust 公共构造入口和 ABI 消费者。但同一个已成功构造的 problem 不能保证在两个官方 backend 上具有一致语义。

**建议修复**

在 core constructor 中至少拒绝非有限 weight，并明确规定 weight 是由 probability 派生还是允许调用者覆盖。若允许覆盖，两个 backend 都必须遵守；若不允许，删除 DTO 中的可写 weight 并统一在 core 计算。

### FS-112 [P3] 公开可变的 `SamplerProgram` 可绕过验证并触发 Rust panic

**位置**

- `crates/faultscope-core/src/packed.rs:11-25`
- `crates/faultscope-core/src/packed.rs:65-103`
- `crates/faultscope-core/src/packed.rs:111-145`
- `crates/faultscope-core/src/packed.rs:233-263`
- `crates/faultscope-core/src/packed.rs:287-318`

`SamplerProgram` 的所有字段、`FaultScopeSimulator.program` 和 `SamplerOperation` 都是公开可写的；`run_sampler_program()` 也是公开函数并返回 `NpResult`，但执行前不验证 program 内部索引。

Rust 调用者可以先通过安全 constructor 得到 simulator，再追加 `SamplerOperation::H(usize::MAX)`。运行时会直接索引 `state.x_frame[*qubit]` 并 panic，而不是返回 `Err`。同类问题还包括非法 noise ID、measurement ID、observable ID 和 capacities 不一致。

**建议修复**

将 compiled program 字段设为私有，只通过已验证 builder 创建；如果必须保留 DTO 式公共可写性，则 `run_sampler_program()` 每次先做完整 layout validation。编译后的不可变程序可以缓存 validation 标记，避免热路径重复检查。

## 4. 性能优化机会

以下项目不会直接改变正确结果，但在大 repeat、大任务集或详细计数场景中会明显增加时间、内存或 I/O。时间仅用于说明当前机器上的增长趋势，不应作为跨机器基准。

### PERF-101 Repeat 在导入和编译阶段仍按逻辑迭代数展开

**位置**

- `faultscope/io/stim.py:383-409`
- `crates/faultscope-core/src/program.rs:175-209`
- `crates/faultscope-core/src/program.rs:496-527`

Python importer 的 `_build_nodes()` 保留 compact repeat，但 `_scan_nodes()` 仍执行 `for _ in range(node.count)`。Rust `expand_operations()` 随后又为每次迭代递归展开 body，把每个逻辑 operation、key alias、location 和 loop range 写入 vector，之后才生成 compact loop kernel。

本次本机复测：

| 场景 | count | 时间 |
|---|---:|---:|
| Stim `REPEAT N { TICK }` 元数据扫描 | 10,000 | 0.0017 s |
| 同上 | 100,000 | 0.0162 s |
| 同上 | 1,000,000 | 0.1636 s |
| `REPEAT N { H 0 }` native 编译 | 10,000 | 0.0016 s |
| 同上 | 100,000 | 0.0160 s |
| 同上 | 500,000 | 0.0832 s |

时间呈线性增长；包含测量 key、detector、noise metadata 时还会产生同阶内存分配。

**建议**

为 repeat body 计算可组合的 measurement/coordinate/detector summary；Rust 侧保留 compact loop IR，先编译 body kernel，再用闭式 offset 或周期表示迭代。只在确实需要逐轮唯一 provenance 的边界上物化。

### PERF-102 任意详细计数选项都会关闭 optimized count plan

**位置**

- `crates/faultscope-collection/src/counting.rs:25-66`
- `crates/faultscope-collection/src/counting.rs:68-115`
- `crates/faultscope-collection/src/counting.rs:118-237`

只要启用 detector event、observable combo 或任意 postselection，`uses_detailed_path()` 就让 `prepare_dem_count_plan()` 返回 `Generic`。该路径重新构造 detector/observable `HashMap<Mask>`，并逐 shot、逐 observable 扫描；即使调用者只想增加 detector popcount，也会放弃 packed logical/decoder plan。

**建议**

- detection events 直接对已编译 packed detector 列做 popcount；
- postselection 用 packed OR/AND 生成 discard mask；
- 只物化本次选项真正需要的 detector/observable；
- observable combo 可按 word 聚合，或把高基数明细设为显式慢路径。

### PERF-103 Python PyMatching adapter 逐 detector 做 bit-layout 转置

**位置**

- `faultscope/decoders/pymatching.py:358-391`

`_masks_to_packed_shots()` 对每个 detector 把 Python bigint 转 bytes、调用 `np.unpackbits()` 生成 shots 长临时数组，再 OR 到 shot-major 输出。复杂度和中间内存均为 `O(shots * detectors)`，发生在 decoder 本身工作之前。

10,000 shots、预热 NumPy 后的本机结果：

| detector 数 | 输出大小 | 转换时间 |
|---:|---:|---:|
| 100 | 0.13 MB | 0.00095 s |
| 1,000 | 1.25 MB | 0.0163 s |
| 5,000 | 6.25 MB | 0.0777 s |

**建议**

让 sampler 与 decoder 直接交换同一种 shot-major packed ABI；否则至少在 Rust/NumPy 中一次性执行二维 bit transpose，避免每个 detector 创建 Python bytes 和临时 ndarray。

### PERF-104 Decoder fan-out 会重复编译同一个 circuit/DEM

**位置**

- `faultscope/collection/_collect.py:143-149`
- `faultscope/collection/_collect.py:356-413`
- `faultscope/collection/_collect.py:545-566`

`_expand_tasks_for_decoders()` 为 K 个 decoder 复制 task；随后 `_run_collect()` 对每个复制项调用 `_native_task()`，而 `_native_task()` 每次都会重新 `compile_native_dem_sampler_from_circuit(..., materialize_dem=True)`。

因此同一个 circuit 对 K 个 decoder 的 collection 初始化成本为 `O(K * compile(source))`，还会重复构造相同 DEM 和 source identity。

**建议**

先按 source/postselection 等 sampler 相关 identity 分组，每组只编译一次 sampler/DEM，再为不同 decoder 创建轻量 task view。确保 worker 是否可共享由 sampler 的线程安全契约决定，但 immutable compiled plan 应可共享。

### PERF-105 Strong identity 会重复构造大型嵌套 payload 和 JSON

**位置**

- `faultscope/collection/_identity.py:89-178`
- `faultscope/collection/_identity.py:181-240`
- `faultscope/collection/_collect.py:437-458`

当前 schema 已修复旧版不稳定 `repr()` 问题，但每个 task 仍会完整构造 Python 嵌套 dict/list、canonical JSON 字符串，再做 SHA-256。对于 circuit task，payload 同时包含原 circuit 和已经物化的 DEM；decoder fan-out 又会重复这一过程。

**建议**

在 immutable circuit、DEM 和 decoder 上缓存版本化 canonical digest；组合 task identity 时只哈希固定大小的子 digest、metadata 和 masks。大对象可使用流式编码，避免同时保留 payload 与完整 JSON 字符串。

### PERF-106 Streaming resume 每个 batch 都重新打开 CSV，且维护未使用的聚合副本

**位置**

- `faultscope/collection/_collect.py:151-190`
- `faultscope/collection/_types.py:325-343`

progress 模式下每个 committed delta 都调用 `write_stats_to_csv_file(..., append=True)`，重复执行 mkdir、exists/stat、open、DictWriter 创建和 close。同时 `existing_data = CollectionData(...)` 会累计每个 delta，但 collection 结束前从未读取该对象。

**建议**

在 collection 生命周期内持有 append writer，按明确的 durability 策略 flush/fsync，并在 finally 中关闭；删除未使用聚合，或让它承担真正的增量校验职责。

### PERF-107 Native backend 的动态错误字符串会无界增长

**位置**

- `backends/faultscope-fusion-blossom/src/lib.rs:352-452`
- `backends/faultscope-fusion-blossom/src/lib.rs:2559-2578`
- `backends/faultscope-pymatching/src/lib.rs:370-452`
- `backends/faultscope-pymatching/src/lib.rs:1859-1878`

为保证 FFI 返回的 `CString` 指针持续有效，factory 和 worker state 把每条动态错误 push 到 `Vec<CString>`，直到整个 state 被销毁。长生命周期进程若反复收到无效输入，内存会随错误次数无界增长。

**建议**

定义错误指针的有效期，改用每 worker “last error” slot、有界 ring buffer 或由调用方复制后释放的 owned status ABI，并增加错误缓存长度监控。

## 5. 上一轮问题的当前状态

仓库原报告针对 `d993ed4` / 0.2.1。当前 HEAD 已修复大多数旧问题，避免把历史结论误当成当前开放项，状态如下：

| 旧编号 | 当前状态 | 说明 |
|---|---|---|
| FS-01 | 已修复 | 无测量、超过 64 shots 的 frame-only observable 已覆盖多 word mask |
| FS-02 | 已修复 | decoder/source identity 已改为版本化 canonical payload，schema v2 |
| FS-03 | 已修复 | DEM 隐式 detector/observable ID 已统一 canonical layout |
| FS-04 | 已修复 | circuit/qubit 边界已在公共编译入口验证 |
| FS-05 | 已修复 | 相同 qubit 的 `CX/CZ` 已拒绝并加入回归测试 |
| FS-06 | 已修复 | noise model 有限性与 Python/native 配置语义已收紧 |
| FS-07 | 已修复 | Pauli/state 维度和 tableau 不变量已校验 |
| FS-08 | 已修复 | DEM edge 重复 ID 按 parity 统一规范化 |
| FS-09 | 已修复 | `RepetitionCodeDecoder` mapping 顺序已修正 |
| FS-10 | 已修复 | `TaskStats` 关系校验及分析输入验证已完善 |
| FS-11 | 原问题已修复，但出现新回归 | Stim 语法校验已收紧；扁平 repeat 恢复引入 FS-101 |
| FS-12 | 已修复 | packed correction padding 已由 ABI v4 的精确宽度语义约束 |
| FS-13 | 已修复 | repetition 可视化的 distance=1 和画布范围已修复 |
| FS-14 | 仍存在 | 对应本报告 FS-109 |

上一轮 PERF-01、02、03、05、06 的代码模式仍存在；旧 PERF-04 的不稳定 identity 已修复，但大型 canonical serialization 的成本仍存在，并在本报告 PERF-105 中更新描述。

## 6. 需要补齐的测试

1. **Stim 结构等价性质测试**：任何自动 repeat 恢复前后的完整指令流、测量数、lookback 与坐标效果必须等价。
2. **Resume schema 矩阵**：详细计数 flags、custom stop key、部分/完成历史数据和 CSV round-trip 的全组合。
3. **Collection 一致性测试**：普通、single-task、adaptive、hotspot 对相同停止配置必须给出同一提交语义。
4. **Option overlay 测试**：每个字段覆盖为默认值、`None`、零值和具体值；验证“未设置”与“显式清空”不同。
5. **Hotspot state 不变量测试**：缺 event mask、跨 program state、零 shots、不同 mask word 数必须返回错误而非 panic/假结果。
6. **Rust 公共 DTO fuzz/property test**：对 `SamplerProgram`、graphlike problem 和 packed views 的无效索引、长度、NaN/Inf 做边界测试。
7. **平台构建测试**：macOS arm64 固化 `cargo test --workspace --all-targets --all-features`，backend 另跑 maturin/Python smoke tests。
8. **性能回归 benchmark**：repeat count、decoder fan-out、详细 counter、bit transpose 和 streaming 小 batch 应有可比较基线。

## 7. 推荐修复顺序

### 第一阶段：阻止静默数据错误和 panic

1. 修复 FS-101，保证 Stim 导入绝不删除未覆盖指令。
2. 为 collection counter schema 建立版本化 resume 兼容规则，一并修复 FS-102、FS-107。
3. 收紧 hotspot state 契约并让两个 estimate 入口返回 `NpResult`，修复 FS-103。

### 第二阶段：统一 collection 语义

4. 复用统一完成谓词修复 FS-104。
5. 引入全字段 explicit override 模型修复 FS-105。
6. resume 后重绑当前 identity 修复 FS-106。
7. 统一 Python 数值边界修复 FS-108。

### 第三阶段：工程和公共 API

8. 修复 backend install plan（FS-110），避免无效 checkout 和虚假的 revision pin。
9. 拆分/隔离 PyO3 cdylib 测试 target（FS-109）。
10. 收紧 graphlike weight 和 compiled sampler program 的不变量（FS-111、FS-112）。

### 第四阶段：性能

11. 优先消除 decoder fan-out 的重复编译，并为详细计数保留 packed plan。
12. 再处理 repeat compact compilation、PyMatching transpose、identity cache、CSV writer 和 backend error cache。

## 8. 总体评价与限制

FaultScope 当前的常规测试、类型检查和 lint 覆盖已经较好，上一轮多数核心语义问题也确实得到修复。当前最值得警惕的风险不再是基础 stabilizer/DEM 算法，而是位于“输入结构恢复、任务身份、持久化统计、可选状态”这些跨层契约处：单层测试都可能通过，但组合后产生静默错误。

本审查没有运行长时间 release-mode 基准、Miri、sanitizer、全量 fuzz，也没有对外部网络上的所有 backend 发布 wheel 做安装验证；性能数字是本地小型趋势测量。除这些限制外，报告中的 FS-101 至 FS-110 均有直接复现或确定的执行路径证据；FS-111、FS-112 是限定在 Rust 公共 API 的明确不变量缺口。
