# FaultScope 全项目代码审查报告

> 审查日期：2026-07-16
>
> 审查版本：`dev` / `d993ed4`
>
> 项目版本：`0.2.1`
>
> 结论性质：静态审查、现有测试、差分验证和最小复现的综合结果

## 1. 执行摘要

本次审查覆盖 Python 公共 API、Rust 核心、collection 调度与计数、PyO3 边界、Stim 导入、两种可选 decoder backend、可视化、测试和 benchmark。共检查 113 个 Python/Rust/C++ 源文件，约 55,065 行。

现有测试基线整体良好：Python、Rust workspace、Ruff、mypy 和 Clippy 的常规检查均通过。但仍确认了 14 个问题，其中两个 P0 会在合法使用方式下静默产生错误结果或错误合并统计：

1. 无测量、仅使用最终 Pauli frame 定义逻辑可观测量的电路，在 `shots > 64` 时只保留前 64 个 shot。
2. collection 的 `strong_id` 没有包含已构造 decoder 对象的真实配置，不同行为的 decoder 会被合并为同一任务，污染 resume 数据并改变任务随机流。

此外，DEM 的隐式 ID 在快速路径和详细计数路径之间语义不一致；多个公共入口可因越界 qubit 触发 Rust panic；同一 qubit 上的 `CX/CZ` 会破坏 tableau；若干噪声模型在 Python 与 native sampler 中行为不同。

### 严重级别汇总

| 级别 | 数量 | 含义 |
|---|---:|---|
| P0 | 2 | 合法输入下静默错误、统计污染或持久化身份冲突，应立即修复 |
| P1 | 5 | 明显逻辑错误、跨路径语义不一致或可触发 native panic |
| P2 | 5 | 边界输入、兼容性、API 不变量或次级计数错误 |
| P3 | 2 | 可视化与测试基础设施问题 |

## 2. 审查范围与方法

### 2.1 代码范围

| 区域 | 文件数 | 行数 | 主要内容 |
|---|---:|---:|---|
| `faultscope/` | 34 | 6,341 | Python API、collection、decoder、Stim、可视化 |
| `crates/` | 48 | 28,137 | Rust core、collection、PyO3 绑定及 Rust 集成测试 |
| `backends/` | 7 | 5,436 | fusion-blossom、PyMatching native backend |
| `tests/` | 16 | 11,728 | Python 单元、集成、发布契约测试 |
| `benchmarks/` | 8 | 3,423 | sampler、DEM、decoder、collection benchmark |

### 2.2 已执行的基线检查

| 命令 | 结果 |
|---|---|
| `.venv/bin/python -m pytest -q` | `284 passed, 25 skipped, 64 subtests passed` |
| `.venv/bin/python -m unittest discover -s tests -q` | `278 tests OK, 25 skipped` |
| Ruff 全项目检查 | 通过 |
| mypy（36 个 Python 源文件） | 通过 |
| `cargo test --workspace --all-features` | 通过 |
| `cargo test -p faultscope-core -p faultscope-collection` | core 与 collection 的单元/公开 API 测试全部通过 |
| Clippy，warnings 作为错误 | 通过 |

审查额外使用了以下方法：

- 对 packed mask 的 63/64/65/129 shot 边界进行差分验证；
- 比较同一 DEM 在默认计数、详细计数、decoder 和矩阵编译路径的结果；
- 比较 Python noise API 与 native sampler 对同一配置的行为；
- 使用本地安装的 Stim 对导入器的语法接受范围做交叉验证；
- 对 repeat 编译、详细 collection 路径和 PyMatching 数据转换做小型性能测量；
- 对无效 qubit、重复 target、长度不一致的 Pauli 向量和 padding bit 做不变量检查。

## 3. 已确认的正确性问题

### FS-01 [P0] 无测量电路在 `shots > 64` 时截断逻辑可观测量

**位置**

- `crates/faultscope-core/src/packed.rs:424-459`
- `crates/faultscope-core/src/mask.rs:22-25`

**现象**

下面的合法电路没有测量，只通过最终 Pauli frame 定义逻辑可观测量。噪声率为 1，因此每个 shot 都应发生逻辑翻转。

```python
from faultscope import (
    BernoulliPauliNoise, Circuit, LogicalObservable,
    NoiseLocation, Operation, compile_native_sampler,
)

loc = NoiseLocation("x", BernoulliPauliNoise("X"), 1.0, (0,))
sampler = compile_native_sampler(
    Circuit(1, [Operation.noise(loc)]),
    observables=(LogicalObservable(0, pauli_qubits=(0,), pauli="Z"),),
)

for shots in (63, 64, 65, 129):
    batch = sampler.sample(shots=shots, seed=1)
    print(shots, batch.observables[0].bit_count(), sampler.estimate(shots=shots, seed=1).mean_loss)
```

实际输出：

```text
63  63  1.0
64  64  1.0
65  64  0.9846153846153847
129 64  0.49612403100775193
```

**根因**

`indexed_measurement_parity` 通过第一个测量 mask 推断 word 数；当电路没有任何测量时，使用 `unwrap_or(1)`，固定创建一个 64-bit word。随后 `xor_frame_measurement_flip_into` 把多 word frame XOR 到该 mask，但 `Mask::xor_assign` 使用 `zip`，静默忽略剩余 word。

**影响**

- `sample()` 的逻辑 observable mask 错误；
- `estimate()` 的 `mean_loss` 随 shot 数增大而被系统性低估；
- 问题不会报错，且 64 shot 以内完全正常，因此很容易逃过常规测试；
- 所有依赖最终 frame、且没有记录测量的 observable 都受影响。

**建议修复**

1. parity mask 的宽度必须由当前 batch 的 `word_count(shots)` 或 `state.all_mask.words.len()` 显式传入，不能从测量集合推断。
2. `Mask::{xor_assign,or_assign,and_assign}` 至少增加长度相等断言；更稳妥的公共 API 应返回 `NpResult`。
3. 增加无测量、空测量 parity、frame-only observable 的 `0/1/63/64/65/129/513` shot 回归测试。

### FS-02 [P0] `strong_id` 忽略 decoder 对象配置，导致统计错误合并

**位置**

- `faultscope/collection/_collect.py:420-454`
- `faultscope/collection/_types.py:220-242`
- `docs/user_guide.md:625-628`

**现象**

`_strong_id` 只记录 `_decoder_name()` 和显式 `task.decoder_options`。对于已经构造好的 decoder 对象，其内部结构、参数、子 decoder 和 backend 配置均不参与哈希。

使用同一个确定性 DEM，以下两个 decoder 都名为 `composite`，但一个不纠错、另一个能够纠错：

```text
decoder A strong_id = 2cb06a3e...  shots=4 errors=4
decoder B strong_id = 2cb06a3e...  shots=4 errors=0
```

将两行放入 `CollectionData` 后得到一行：

```text
shots=8 errors=4 strong_id=2cb06a3e...
```

这不是展示层问题，而是两个不同实验被当成同一个实验累加。

同类问题还包括：

- 不同 `distance` 或 `measurement_keys` 的 `RepetitionCodeDecoder` 只会记录类名；
- 不同 matching graph 的 `PyMatchingDecoder` 只会记录类名；
- `NativeCompositeDecoder` 的子 decoder 不进入 identity；
- `repr(dem)` / `repr(circuit)` 内的 dict 保留插入顺序。语义相同但 tags 插入顺序不同的两个 DEM 会得到不同 `strong_id`，与“stable problem identity”的文档承诺相反。

**影响**

- CSV resume 可能把错误 decoder 的历史统计当作当前任务数据；
- `CollectionData` 静默合并不同行为；
- Rust scheduler 使用 strong id 派生 task-local RNG，碰撞任务会得到相同随机流；
- 语义相同但 map 顺序不同的任务又会失去 resume 命中，形成“该合并的不合并，不该合并的合并”。

**建议修复**

1. 为 decoder 定义稳定的 fingerprint 协议，例如 `decoder.strong_id_payload()`，包含 backend、版本、参数、ID 布局和所有子 decoder。
2. 对无法提供稳定 fingerprint 的任意对象，要求调用者显式提供 identity，或拒绝启用 resume。
3. DEM/circuit 使用规范化的结构化序列化，所有 mapping 按 key 排序，不依赖 `repr()`。
4. 在 payload 中加入 identity schema 版本；修复会改变历史 strong id，应提供清晰的 resume 迁移策略。
5. 回归测试必须覆盖“同名不同行为 decoder 不碰撞”和“map 插入顺序不影响 identity”。

### FS-03 [P1] DEM 隐式 detector/observable ID 在计数路径间语义不一致

**位置**

- `crates/faultscope-core/src/dem_problem.rs:188-231`
- `crates/faultscope-core/src/dem_sampling.rs:145-169`
- `crates/faultscope-core/src/dem_sampling.rs:536-568`
- `crates/faultscope-core/src/dem_sampling.rs:722-780`
- `crates/faultscope-collection/src/counting.rs:49-65`
- `crates/faultscope-collection/src/counting.rs:111-180`

**现象**

项目明确支持 edge 引用未显式声明的 ID：`compile_indexed()` 会保持已声明顺序，并把 edge 中首次出现的 ID 追加到列表；现有测试也验证了这一行为。

最小 DEM：没有 detector/observable 声明，只有一条概率 1 的 edge，翻转 `D7 L9`。

```text
compile_indexed IDs: detectors=(7,), observables=(9,)
run_batch(4):         D7=0b1111, L9=0b1111
estimate_default:     mean_loss=1.0
normal collect:       errors=4
detection counting:   errors=0, detection_events=4, detectors_checked=0
observable combos:    errors=0
```

仅打开一个“报告更多统计”的选项，就把逻辑错误数从 4 改成了 0。

**根因**

- `DemHotspotEstimator::new` 的公开 `detector_ids` / `observable_ids` 只来自显式声明；
- generic `run_batch` 遇到 edge ID 时通过 `HashMap::entry` 动态创建 mask；
- 默认快速 logical-count plan 又会把 edge observable ID 追加进去，因此默认结果正确；
- 任意 postselection、observable combo 或 detection-event 选项都会切到 `PreparedDemCountPlan::Generic`；详细计数只遍历 estimator 的显式 observable ID，因此忽略 `L9`，且 `detectors_checked` 也按零个显式 detector 计算。

**影响**

- 报告选项改变主结果，违反 collection 选项应只增加统计的直觉和接口契约；
- postselection mask 无法引用隐式 ID；
- decoder 和 packed sampling plan 可能拒绝 generic sampler 已接受的 ID；
- 同一个 DEM 在 indexed problem、default estimate、detailed collection 和 decoder 中具有不同布局。

**建议修复**

在 estimator 构造时生成唯一的 canonical ID 列表：`显式声明顺序 + edge 中未见 ID 的首次出现顺序`，并让 generic sampler、packed plan、logical plan、详细计数、postselection 和 decoder 全部使用该列表。若决定禁止隐式 ID，则必须统一拒绝，并修改当前明确允许该行为的测试与文档；不应保留当前混合语义。

### FS-04 [P1] 缺少统一 qubit 范围校验，Python 可直接触发 Rust panic

**位置**

- `crates/faultscope-python/src/core_api.rs:802-850`
- `crates/faultscope-core/src/compile.rs:263-339`
- `crates/faultscope-core/src/stabilizer.rs:531-635`
- `crates/faultscope-core/src/packed.rs:649-679`
- `faultscope/runtime/native.py:41-54`

**现象**

对 `n_qubits=1` 的电路使用 `H(1)`、`CX(0,1)`、`CZ(0,1)` 或 `SWAP(0,1)`，`compile_native_sampler` 会产生 `pyo3_runtime.PanicException: index out of bounds`。越界 noise target 可先成功编译，然后在 `sample()` 中 panic。

公共 `StabilizerState.zero(1).apply_h(1)` 和 `apply_cx(0,1)` 也会 panic。

**根因**

`Circuit` 构造和编译入口没有递归验证每个 operation、noise location、Pauli target 和 observable target 是否小于 `n_qubits`。symbolic tableau 和 packed runtime 随后直接索引 vector。PyO3 panic 使用 `PanicException`，它不属于普通 `Exception`，所以 `faultscope/runtime/native.py` 的 `except Exception` 也无法把它转换为 `UnsupportedNativeCircuitError`。

**影响**

- 无效用户输入跨越 Rust 安全边界并表现为 panic，而不是稳定的 `ValueError`；
- 在批处理、服务或 notebook 中可能绕过正常异常处理；
- 某些错误在 compile 阶段发生，某些直到 sample 阶段才发生，行为不可预测。

**建议修复**

增加一个共享、递归的 `Circuit::validate()`：检查所有 gate、repeat body、measurement/reset、noise、Pauli product 和 observable target；编译 sampler、生成 DEM 和 PyO3 公共构造/执行入口均调用它。公共 state/frame 方法也应在索引前返回 `PyValueError`。

### FS-05 [P1] 同一 qubit 上的 `CX/CZ` 被接受并破坏 stabilizer tableau

**位置**

- `crates/faultscope-core/src/stabilizer.rs:574-604`
- `crates/faultscope-core/src/packed.rs:470-481`
- `crates/faultscope-core/src/packed.rs:818-830`

**现象**

`StabilizerState.zero(1)` 的生成元原本是 `Z`。执行 `apply_cx(0,0)` 或 `apply_cz(0,0)` 后，生成元被改成 identity：

```text
before: x=[[0]], z=[[1]], sign=[0]
after:  x=[[0]], z=[[0]], sign=[0]
```

这是无效的 stabilizer tableau。电路编译同样接受 `CX(0,0)` / `CZ(0,0)`。更危险的是 symbolic 与 packed 路径不一致：symbolic `CZ(q,q)` 通过 `H-CX-H` 破坏 tableau，而 packed runtime 把它当作 no-op。当前 packed 单元测试甚至明确断言 `CX(0,0)` 把 frame 清零，固化了错误行为。

**根因**

`apply_cx` 在 `control == target` 时主动把两列设为零；`apply_cz` 的同 target 特殊分支继续调用该逻辑。没有任何公共入口拒绝重复 target。

**影响**

- invalid gate 没有被拒绝，而是生成看似正常但物理上无效的确定性结果；
- symbolic ideal、packed frame 和 DEM propagation 可能互相不一致；
- duplicate qubit 的两 qubit noise 和 sparse Pauli product 也缺少统一策略，可能忽略相位或把两个独立 target 折叠到一根 wire。

**建议修复**

编译前拒绝 `CX/CZ` 的相同 target。`SWAP(q,q)` 可以明确规定为 no-op 或统一拒绝，但必须一致。Pauli product、measurement product 和 two-qubit noise 也应要求 target 唯一，除非实现了包含相位的规范化规则。

### FS-06 [P1] 噪声模型接受非法配置，且 Python 与 native sampler 语义不同

**位置**

- `crates/faultscope-python/src/noise_api.rs:15-43`
- `crates/faultscope-python/src/noise_api.rs:68-147`
- `crates/faultscope-python/src/noise_api.rs:206-273`
- `crates/faultscope-python/src/noise_api.rs:348-395`
- `crates/faultscope-python/src/spec.rs:517-535`
- `crates/faultscope-core/src/sampling.rs:86-131`

**已确认的子问题**

| 配置 | Python 行为 | native 行为 |
|---|---|---|
| `PauliChannel({"": 1})` | 接受，采样空字符串 | 接受，成为无操作事件 |
| `PauliChannel({"X": inf, "Z": 1})` | rate=1 时选择 `X` | 相同模型选择 `Z` |
| `PauliChannel` 含 NaN | 接受，`total_weight=nan` | 接受，选择逻辑依赖 NaN 比较 |
| `TwoQubitDepolarizing([])` | 构造成功，Python `sample()` 因空 `randrange` 报错 | native 完全忽略空 events，按标准 15 事件采样 |
| `TwoQubitDepolarizing(["XX"])` | Python 始终采样 `XX` | native 忽略配置，仍按标准 15 事件采样 |
| `BernoulliPauliNoise("A")` | 构造成功 | rate=0 时静默成功，rate=1 时才报非法 Pauli |

**根因**

- `validate_pauli_channel` 只检查 `< 0` 和 `total <= 0`，没有检查 `is_finite()`；空字符串绕过 identity 检查；
- `TwoQubitDepolarizing` 保存任意 `_events`，但 native cache 只记录枚举类型 `NoiseModel::TwoQubitDepolarizing`，丢失配置；
- `BernoulliPauliNoise` 构造时完全不验证 Pauli；packed runtime 在 event mask 全零时提前返回，因此错误是否出现取决于 rate 和随机结果。

**影响**

同一个公开模型通过 Python reference path 与 native path 可生成不同噪声分布。这会破坏差分测试、可复现性和用户对模型配置的信任。

**建议修复**

1. 所有权重及其总和必须 finite，事件必须非空、非 identity、长度一致且字符合法。
2. `BernoulliPauliNoise` 在构造时验证非空合法 Pauli，不允许由采样结果决定配置是否合法。
3. `TwoQubitDepolarizing` 若不支持自定义 events，就移除/拒绝 `_events`；若支持，则 native `NoiseModel` 必须完整保存并使用配置。
4. 加入 Python/native 确定性对照测试，尤其覆盖 rate 0/1 和自定义事件。

### FS-07 [P2] Pauli 与 stabilizer API 静默截断维度或破坏 tableau 行宽

**位置**

- `faultscope/core/pauli.py:50-97`
- `faultscope/core/pauli.py:118-136`
- `crates/faultscope-core/src/pauli.rs:5-20`
- `crates/faultscope-core/src/pauli.rs:43-97`
- `crates/faultscope-python/src/stabilizer_api.rs:194-259`

**现象**

```text
multiply_pauli_rows(1-qubit row, 2-qubit row) -> 返回 1-qubit 结果，不报错
sparse_pauli_to_xz(2, [-1], "X")             -> 把 X 写到最后一个 qubit
StabilizerState.zero(2).measure_pauli([1], [0], rng)
                                                   -> 接受，并把 tableau 第一行长度从 2 改成 1
```

Python 与 Rust 的 symplectic product 和 row multiplication 均使用 `zip`，长度不同时按最短输入截断。PyO3 的 `apply_pauli_string`、`measure_pauli`、`is_deterministic_pauli` 没有校验向量长度等于 `n_qubits`，也没有校验值是二进制。Python sparse helper 又允许负索引。

**影响**

- 错误输入可能立即产生错误答案，也可能先破坏 tableau，再在后续操作中 panic；
- 同一类错误在 Python helper、Rust core 和 PyO3 state API 中表现不同；
- 公共 `Mask` 和若干 word helper 也依赖相同长度不变量，但使用 `zip` 静默截断。

**建议修复**

建立共享维度校验：所有 x/z 向量长度必须彼此相等且等于 `n_qubits`，元素必须是 0/1，所有 sparse index 必须满足 `0 <= q < n_qubits`。在任何 tableau mutation 前完成校验。

### FS-08 [P1] DEM edge 内的重复 ID 在 sampler 与 decoder 编译中含义不同

**位置**

- `crates/faultscope-core/src/model.rs:247-267`
- `crates/faultscope-core/src/dem_problem.rs:60-97`
- `crates/faultscope-core/src/dem_sampling.rs:746-760`
- `faultscope/decoders/pymatching.py:64-86`

**现象**

对概率 1 的 edge，令 `detectors=(1,1)`、`observables=(9,9)`：

```text
native DEM sampler: D1=0, L9=0, mean_loss=0
binary problem H:   ((0,0), (0,0))
binary problem F:   ((0,0), (0,0))
PyMatching build:   ValueError: matrix.data contains element 2
```

sampler 按 GF(2) 连续 XOR，因此重复两次相互抵消。`CompiledDemLogicalCountPlan` 的现有测试也明确验证重复 observable ID 的奇偶抵消，说明 duplicate 并非简单的未定义输入。可是 binary matrix 和 Python PyMatching path 逐项添加 sparse entry；SciPy 合并 duplicate 后得到数值 2，PyMatching 拒绝该矩阵。`is_graphlike()` 同样按原始列表长度判断，而不是按 parity 规范化后的 detector 数量判断。

**影响**

- 同一 DEM 在采样、默认 logical count、graphlike 判断、binary problem 和 decoder backend 中不等价；
- 用户可能先成功采样，直到创建 decoder 才收到不相关的 sparse matrix 错误；
- 三次重复等其他情况也会被错误分类为 hyperedge。

**建议修复**

在 DEM canonicalization 时按 GF(2) 对每条 edge 的 detector/observable ID 做 parity 去重，并让所有下游表示消费同一个规范化结果。若决定禁止 duplicate，则应在 `DetectorErrorEdge::validate` 统一拒绝，并修改当前依赖奇偶抵消的测试；不能只在部分路径保留 duplicate。

### FS-09 [P2] `RepetitionCodeDecoder` 对 mapping syndrome 使用错误顺序

**位置**

- `faultscope/decoders/classical.py:58-114`

**现象**

decoder 保存了显式 `measurement_keys`，但单 shot `decode()` 对 mapping 直接执行 `sorted(detector_record)`，完全不使用这些 keys。

```python
decoder = RepetitionCodeDecoder(5, measurement_keys=("b", "a", "c", "d"))
decoder.decode([1, 0, 0, 0], {}, None)
# [1, 0, 0, 0, 0]
decoder.decode({"b": 1, "a": 0, "c": 0, "d": 0}, {}, None)
# [1, 1, 0, 0, 0]
```

语义相同的 sequence 和 mapping 因字典 key 的字典序不同，得到不同 correction。`m10` / `m2` 之类 key 也会出现自然顺序与字典序不一致。构造器还接受 `distance=0`、负数和没有意义的配置。

**建议修复**

如果提供了 `measurement_keys`，mapping 必须严格按该顺序读取，并检查缺失/多余 key。若没有 keys，应只接受有明确规范顺序的 detector ID mapping，或要求 sequence。构造时验证 distance，解码时验证每个 syndrome 值为 bit。

### FS-10 [P2] `TaskStats` 缺少关系校验，`error_rate_points` 对 custom count 的统计模型错误

**位置**

- `faultscope/collection/_types.py:129-218`
- `faultscope/collection/analysis.py:14-40`
- `faultscope/collection/threshold.py:473-485`（可作为正确的严格行为对照）

**现象**

- `TaskStats(shots=2, errors=3)` 可构造，`accepted_error_rate_stderr` 对负数开方并报错；
- `TaskStats(shots=2, discards=3)` 可构造，`accepted_shots=-1`；
- `error_rate_points(..., count_key="typo")` 把缺失 key 静默当作 0；
- `detection_events=5, shots=2` 是合法计数，但 helper 把它当成二项错误率 2.5，计算 `sqrt(p*(1-p)/n)` 时崩溃。

Threshold 模块已经采用更合理的严格规则：缺失 custom count 报错，并在把 count 当作二项错误数时验证 `0 <= count <= accepted_shots`。两个公共分析入口因此行为不一致。

**影响**

- CSV 或直接构造的坏统计可传播到合并、plot 和 threshold 之外的分析；
- `detection_events`、`detectors_checked` 等本来允许大于 shots 的计数无法安全绘图；
- 拼错 count key 会产生看似有效的零错误曲线。

**建议修复**

1. `TaskStats.__post_init__` 和 CSV 读取同时验证 `0 <= discards <= shots`、`0 <= errors <= shots-discards`、seconds finite 且非负。
2. `error_rate_points` 对缺失 key 严格报错。
3. 区分“二项错误数”和“任意事件计数”：后者不能使用 binomial stderr；应要求显式 denominator 或返回 `stderr=None/NaN`。

### FS-11 [P2] Stim 子集解析器接受非法 Stim，并拒绝一部分合法 Stim

**位置**

- `faultscope/io/stim.py:317-369`
- `faultscope/io/stim.py:694-703`
- `faultscope/io/stim.py:733-740`

**本地 Stim 交叉验证结果**

| 输入 | FaultScope | Stim |
|---|---|---|
| `PAULI_CHANNEL_1(-0.1,0.2,0) 0` | 接受，变成 rate=0.1、仅 Y 权重 0.2 | 拒绝负概率 |
| `X_ERROR(0.1,) 0` | 接受为一个参数 | 拒绝，尾部空项是额外参数 |
| `PAULI_CHANNEL_1(0.1,,0.2) 0` | 因只剩两个参数而拒绝 | 接受，中间空项解释为 0 |
| `MPP X0**Y1` | 接受为 `X0*Y1` | 拒绝非法 combiner |
| `MPP X0*` | 接受为 `X0` | 拒绝尾部 combiner |

**根因**

- Pauli channel 先过滤 `probability > 0`，但只校验总 rate，不校验每个分量；
- `_parse_args` 删除空字段，丢失 Stim 语法中的参数位置；
- `_split_mpp_products` 删除空 factor，把重复或尾随 `*` 自动修复为另一条有效指令。

**影响**

导入器可能悄悄改变用户给出的噪声模型或 Pauli product，而不是报告源文件错误；同时与 Stim 的 round-trip / differential tests 不完全兼容。

**建议修复**

使用保留 token 位置的参数解析；逐分量验证概率 finite、位于 `[0,1]` 且总和不超过 1；MPP 明确验证 combiner 必须恰好位于两个合法 factor 之间。把上述五个例子加入测试，并在安装 Stim 时运行 differential parser 测试。

### FS-12 [P2] packed correction 的 padding bit 会被当作逻辑错误

**位置**

- `crates/faultscope-core/src/decoder.rs:310-387`
- `crates/faultscope-core/src/decoder.rs:390-454`

**现象**

`PackedObservableShotBatch::new` 只检查 byte 长度和 ID，不验证最后一个 byte 的未使用 bit 必须为零。identical-layout 快速路径按整个 byte 做 XOR：

```rust
let correction = PackedObservableShotBatch::new(vec![10], vec![0x80], 1)?;
let failures = packed_residual_failure_count(&[10], &[0x00], 1, &correction, 1)?;
// 当前 failures == 1；bit 7 是 padding，不对应任何 observable，期望为 0。
```

若 observable 布局不同，慢路径只逐个读取合法 observable bit，会忽略相同 padding。因此快速和慢速路径对同一逻辑数据不等价。

**影响**

内置 backend 当前通常会先清零输出 buffer，因此不容易触发；但第三方 native decoder plugin 或错误 backend 输出可造成虚假的 logical failure，而 shape validation 无法发现。

**建议修复**

在构造/校验时拒绝非零 padding，或在 residual 计算中对最后一个 byte 应用有效位 mask。增加 1、7、8、9 个 observable 的 padding 回归测试，并验证 reordered-layout 与 fast path 一致。

### FS-13 [P3] repetition 可视化对合法 `distance=1` 崩溃，较大参数被固定画布裁切

**位置**

- `faultscope/experiments/repetition.py:43-49`
- `faultscope/viz/drawing.py:38-56`
- `faultscope/viz/repetition.py:39-89`
- `faultscope/viz/repetition.py:119-180`

experiment builder 明确接受任意正奇数 distance，因此 `distance=1` 合法；此时 measurement heatmap 每行有零列，`_draw_heatmap` 的 `max(max(row) for row in matrix)` 抛出 `ValueError: max() iterable argument is empty`。

两个 repetition renderer 还固定使用 `1500x860` 和 `2100x1160` 画布。heatmap 在较大 distance 时会与 top-hotspot panel 重叠或越界；gate-structure 图在较多 rounds / lanes 时被裁切。surface-code renderer 已经根据 distance 动态计算画布，可复用相同做法。

**建议修复**

对零列矩阵绘制明确的“no checks”占位；根据 distance、rounds、cell size 和 side panel 动态计算画布及 panel 起点；增加 `d=1/3/9`、`rounds=1/8` 的像素边界或 snapshot 测试。

### FS-14 [P3] `cargo test --workspace --all-targets` 在 macOS arm64 无法链接

**位置**

- `backends/faultscope-fusion-blossom/Cargo.toml`
- `backends/faultscope-pymatching/Cargo.toml`
- `crates/faultscope-python/Cargo.toml`

**现象**

普通 `cargo test --workspace` 通过，但 `cargo test --workspace --all-targets` 会尝试为 PyO3 `cdylib` 构建 lib test harness，并因 `_Py*` 符号未链接而失败：

```text
could not compile faultscope-fusion-blossom-python (lib test)
Undefined symbols for architecture arm64: _PyBytes_AsString, _PyErr_*, ...
```

这些 crate 已设置 `test = false`，但 `--all-targets` 会显式请求对应 target。该命令不是当前 README/CI 的标准命令，所以不影响正常测试结论，但它构成工具链盲区，也使常见的“全 target”审查命令不可用。

**建议修复**

提供一个仓库级统一测试脚本，明确把纯 Rust crate 的 all-target 测试与 PyO3 cdylib 的 Python/maturin 测试分开；或者把可单测逻辑拆到 `rlib` crate，cdylib 仅保留薄绑定层。

## 4. 性能优化机会

以下不是凭接口猜测的微优化，而是代码热路径和小型本地测量共同指出的改进方向。时间数据用于展示趋势，不应当作跨机器 benchmark 基线。

### PERF-01 Repeat 在优化前完全展开，编译时间和内存按逻辑操作数增长

**位置**

- `crates/faultscope-core/src/program.rs:86-120`
- `crates/faultscope-core/src/program.rs:394-439`

`expand_operations` 会为 repeat 的每次迭代调用 `expand_sequence`，把完整逻辑操作写入 `state.output`，之后才进行 loop kernel / affine 优化。即使输入只存两个 operation node，编译成本仍按 repeat count 线性增长。

对 `REPEAT N { H 0 }` 的本地结果：

| N | 编译时间 | `stored_operation_count` | `operation_count` |
|---:|---:|---:|---:|
| 1,000 | 0.0003 s | 2 | 1,000 |
| 10,000 | 0.0016 s | 2 | 10,000 |
| 100,000 | 0.0163 s | 2 | 100,000 |
| 500,000 | 0.0862 s | 2 | 500,000 |
| 5,000,000 | 1.19 s（进程总时间） | 2 | 5,000,000 |

**建议**

保留 compact repeat IR，先对 body 编译 affine/periodic kernel，再处理迭代次数。只有 measurement key、record lookback、坐标或逐轮 location identity 真正要求物化时才展开相应元数据。对纯 Clifford 周期，编译复杂度可从 `O(body * count)` 降到接近 `O(body + period)`。

### PERF-02 任意详细计数选项都会关闭 optimized count plan

**位置**

- `crates/faultscope-collection/src/counting.rs:33-65`
- `crates/faultscope-collection/src/counting.rs:111-223`

只要启用 detection count、observable combo 或任意 postselection，`prepare_dem_count_plan` 就直接返回 `Generic`。该路径构建 detector/observable `HashMap<Mask>`，然后逐 shot、逐 observable 扫描。

100 detector edges、200,000 shots 的本地测量：

```text
default logical count:             0.0121 s
count_detection_events=True:       0.0293 s
```

两者 errors 完全相同，后者仅增加 detection 计数，却约慢 2.4 倍。

**建议**

为详细选项编译稳定列布局：detection events 直接对 packed detector masks 做 popcount；postselection 用 packed OR/AND；observable combo 可按 word 聚合常见布局。至少不应因一个计数 flag 放弃所有 compiled plan 优化。

### PERF-03 Python PyMatching adapter 做逐 detector/observable 的位布局转置

**位置**

- `faultscope/decoders/pymatching.py:275-330`

`_masks_to_packed_shots` 对每个 detector：把大整数转 bytes、`unpackbits` 成 shots 长数组，再 OR 到 shot-major 输出列。反向转换也逐 observable `packbits` 并创建 Python 大整数。

10,000 shots 的本地转换时间：

| detectors | 输出大小 | 转换时间 |
|---:|---:|---:|
| 100 | 0.13 MB | 0.0014 s |
| 1,000 | 1.25 MB | 0.0162 s |
| 5,000 | 6.25 MB | 0.0751 s |

该成本严格随 `shots * detectors` 增长，并发生在 decoder 真正工作之前。

**建议**

优先让 sampler/native decoder 直接交换相同的 shot-major packed ABI，避免 Python big-int 中转；否则至少把所有 detector masks 一次性构造成二维 byte array，再在 NumPy/Rust 中做批量 bit transpose。

### PERF-04 Strong-id 对大型 source 重复执行 `repr()` 和哈希

**位置**

- `faultscope/collection/_collect.py:431-454`

每个 task/decoder fanout 都重新物化完整 circuit/DEM repr。大型 DEM、多个 decoder 或重复 collection 会重复分配字符串和执行 `O(source size)` 哈希，同时 repr 又不是可靠的 canonical identity。

**建议**

在不可变 native circuit/DEM 上缓存版本化 canonical fingerprint；decoder fanout 只组合 source fingerprint、decoder fingerprint、metadata 和 masks。

### PERF-05 Streaming resume 每个 batch delta 都重新打开 CSV

**位置**

- `faultscope/collection/_collect.py:156-163`
- `faultscope/collection/_types.py:296-330`

`iter_progress` 配合 `save_resume_filepath` 时，每个 native batch callback 都调用一次 `write_stats_to_csv_file(..., append=True)`，重复 open、header 检查、writer 创建和 close。小 batch 或大量 task 时，I/O 元数据成本会变得明显。

**建议**

在 collection 生命周期内持有一个 append writer，按 batch `flush`，并把 durability 策略做成明确选项；任务结束时关闭。仍可保持每个已提交 delta 都可恢复。

### PERF-06 Native backend 动态错误字符串无上限保留

**位置**

- `backends/faultscope-fusion-blossom/src/lib.rs:349-449`
- `backends/faultscope-pymatching/src/lib.rs:367-449`
- 两个 backend 的 README 已明确记录该行为

backend 为保证 callback 指针生命周期，会把每条动态错误保存到 state-owned `Vec<CString>`，直到 state/worker 被销毁。长生命周期服务若持续收到无效 batch，内存会随错误次数无界增长。

**建议**

使用有界 ring buffer、每 worker 的“最后一条错误”槽位或带 generation 的引用计数存储；ABI 文档明确错误消息指针的有效期。既然 README 已承认该权衡，建议同时提供监控计数或上限。

## 5. 测试缺口

现有测试数量多、常规路径覆盖较好，问题主要集中在“两个各自正确的路径交界处”。建议优先补齐以下测试矩阵：

1. **packed 边界**：所有 mask-producing API 在 `0/1/63/64/65/129/513` shots 上测试；必须包含无测量电路。
2. **路径一致性**：同一 DEM 分别走 generic batch、logical fast path、detailed count、postselection、binary problem 和所有 decoder，逻辑结果必须一致。
3. **ID canonicalization**：显式/隐式 ID、重复 ID、不同声明顺序、重复 parity cancellation。
4. **输入不变量**：每个 operation 的越界 qubit、相同 two-qubit target、Pauli 长度、duplicate sparse target、非法 noise event。
5. **Python/native noise 对照**：rate 0、1、中间值、自定义 channel、NaN/Inf 拒绝行为。
6. **Strong-id 属性测试**：语义等价对象的 fingerprint 相同；任一影响行为的 decoder 参数变化都改变 fingerprint。
7. **Stim differential**：对支持的子集使用 Stim parser 作为 oracle，比较接受/拒绝和解析后的参数。
8. **第三方 decoder ABI**：非零 padding、reordered observable IDs、缺失/额外 IDs、错误消息压力测试。
9. **可视化尺寸**：合法最小值、较大 distance/rounds、无 check 的布局。

## 6. 推荐修复顺序

### 第一阶段：阻止静默数据错误

1. 修复 FS-01，并给 `Mask` 增加宽度不变量。
2. 修复 FS-02，设计版本化 decoder/source fingerprint；在此之前不应信任跨 decoder 的 resume 合并。
3. 统一 DEM canonical ID 与 parity 规则，一次解决 FS-03 和 FS-08。

### 第二阶段：建立输入边界

4. 实现共享 `Circuit::validate()`，覆盖 FS-04、FS-05 的 qubit/target 校验。
5. 建立 Pauli/state 向量不变量，修复 FS-07。
6. 收紧 noise model 构造，并保证 Python/native 配置同构，修复 FS-06。

### 第三阶段：公共工具与兼容性

7. 修复 Repetition decoder mapping 顺序和 `TaskStats`/analysis 校验。
8. 收紧 Stim tokenizer/MPP parser，并加入 Stim differential tests。
9. 校验 packed padding，补齐第三方 decoder ABI 测试。

### 第四阶段：性能和工程质量

10. 优先优化 detailed count plan 和 repeat compact IR；二者收益最大且代码根因明确。
11. 消除 Python PyMatching bit transpose、重复 strong-id repr 和 streaming CSV open/close。
12. 动态画布、backend 错误存储上限以及统一的全仓测试入口。

## 7. 总体评价

项目的主架构方向是合理的：typed Rust core、packed shot 表示、compiled sampling plan、Python orchestration 和独立 decoder ABI 都有清晰边界，常规测试也相当充足。当前最突出的问题并非主算法完全错误，而是多个内部表示对同一概念使用了不同的隐含规则：mask 宽度从测量推断、DEM ID 有时显式有时动态、duplicate 有时按 GF(2) 有时按 sparse 数值、decoder identity 有时只是名字。

因此最有效的修复策略不是逐处打补丁，而是把以下四个不变量提升为共享、可测试的核心协议：

1. 每个 batch 的 mask 宽度只由 shots 决定；
2. 每个 circuit/DEM 在编译前有唯一 canonical target/ID 布局；
3. 所有影响行为的配置都有稳定、版本化 fingerprint；
4. 所有 Python/native 边界在进入索引和 mutation 前完成完整校验。

做到这四点后，本报告中的大多数高优先级问题会同时消失，后续性能优化也能建立在更稳定的内部表示上。
