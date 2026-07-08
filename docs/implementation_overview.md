# FaultScope 实现概览

本页是 FaultScope 当前 runtime 的实现概览。更完整的理论原理、公式推导和 packed/DEM 计算细节见
[FaultScope 理论原理与公式细节](theory.md)。

本页描述 FaultScope 当前实现背后的数学模型和运行时边界。产品执行路径是 Rust
`faultscope-core` 中的 bit-packed batch runtime，通过 Python API 暴露为
`FaultScopeSimulator`、`DetectorErrorModelGenerator` 和
`DemFaultScopeSimulator`、`DemHotspotEstimator`。本文中的 trajectory 是概念模型；实际 Python 回调接收的是
batch mask 对象，而不是逐 shot trajectory 对象。本文的 DEM 术语使用 detector error model
formalism：detector matrix \(D\) 表示 measurement parity constraints，measurement syndrome
matrix \(\Omega\) 表示 circuit errors 翻转哪些 measurements，detector error matrix
\(H=D\Omega\) 表示 circuit errors 违反哪些 detectors。

## 目标函数

一次前向采样可以概念化为

```text
tau = (e_1, e_2, ..., e_M, m_1, m_2, ..., m_R, s, o)
```

其中：

- `e_l` 是第 `l` 个 noise location 的采样事件。
- `m_r` 是 measurement record。
- `s` 是由 detector declarations 生成的 detector syndrome。
- `o` 是由 observable declarations 或最终 Pauli frame 得到的 logical observable flip record。

目标函数是 shot-level loss 的期望：

```text
J(lambda) = E_tau[L_loss(tau)].
```

在 Python API 中，自定义 loss 通过 packed mask callback 表达。forward estimator 支持：

```text
loss_mask_fn(batch) -> int
loss_mask_fn(batch, corrections) -> int
```

DEM estimator 固定调用两参形式：

```text
loss_mask_fn(batch, corrections) -> int
```

如果没有 decoder 或 `correction_mask_fn`，`corrections` 是空 mapping。

返回整数的第 `k` 位表示第 `k` 个 shot 是否贡献 loss。默认 logical loss 使用 declared
observables 和 decoder correction masks：

```text
failure_mask = any_observable(observable_mask xor correction_mask)
```

## 噪声率导数

每个 noise location `l` 有一个事件分布：

```text
e_l ~ p_l(e; lambda_l)
```

对采样事件定义 score：

```text
s_l(tau) = d log p_l(e_l; lambda_l) / d lambda_l
```

FaultScope 的 Pauli-compatible 噪声模型都实现同一类 score。对 Bernoulli 型事件：

```text
s_l(tau) =
  1 / lambda_l       if an error event occurred
 -1 / (1-lambda_l)   otherwise
```

实际实现会对 rate 做数值裁剪，避免 `1 / lambda_l` 或 `1 / (1-lambda_l)` 发散。

score-function estimator 为：

```text
dJ / d lambda_l = E[(L_loss(tau) - b) s_l(tau)]
```

其中 `b` 是 baseline，默认取 batch mean loss。baseline 不改变真实敏感度，因为
`E[s_l] = 0`。有限 batch 中 `mean(s_l)` 不会严格为 0，因此 baseline 可能改变单次
Monte Carlo 排序，但通常降低估计方差。

## 热点分数

FaultScope 输出 signed sensitivity：

```text
sensitivity_l = dJ / d lambda_l
```

默认热点分数用于排序：

```text
hotspot_l = |sensitivity_l|
```

结果对象还按 `NoiseLocation.tags` 聚合：

- `by_qubit`
- `by_round`
- `by_gate`
- `by_operation`

## 支持范围

当前模型限制在 stabilizer-compatible stochastic workflows：

- Clifford gates: `H`, `S`, `S_DAG`, `CX`, `CZ`, `SWAP`
- Ideal Pauli gates: `X`, `Y`, `Z`, and sparse Pauli strings through
  `Operation.pauli_gate(...)`
- Single-qubit `X`/`Y`/`Z` measurement
- Pauli-string measurement
- `X`/`Y`/`Z` basis reset
- Bernoulli Pauli noise
- weighted Pauli-channel noise
- single-qubit depolarizing noise
- two-qubit depolarizing noise
- measurement bit-flip noise attached to measurements
- detector and observable declarations
- DEM generation by single-error propagation
- DEM-level sampling and hotspot estimation
- optional PyMatching decoder integration for graphlike DEMs
- flattened Stim text subset import

非 Clifford gates、非 Pauli 噪声、amplitude damping 等不直接进入 stabilizer runtime；需要先做
Pauli twirling、离散化近似，或替换为 stabilizer-compatible stochastic channel。

## Batch Runtime

`FaultScopeSimulator` 共享一个理想 stabilizer support，并把 per-shot 差异压入整数
mask。第 `k` 个 shot 存在整数的第 `k` 位中：

- `X_frame[q]`: qubit `q` 上是否有 X frame 分量。
- `Z_frame[q]`: qubit `q` 上是否有 Z frame 分量。
- `M[key]`: measurement key 的测量结果。
- `S[id]`: detector id 的 detector syndrome bit。
- `O[id]`: logical observable id 的 observable flip bit。
- `E[location_id]`: noise location 是否采样到 error/flip event。

热点聚合在 Rust 中用 `popcount` 完成。例如 Bernoulli error location 的计数来自：

```text
loss_and_event = popcount(loss_mask & event_mask)
event_count = popcount(event_mask)
```

再代入同一 score-function estimator。

该 runtime 支持确定和随机 Pauli measurement，只要后续电路不依赖单个 shot 的测量结果选择不同操作。
FaultScope 当前不暴露通用 per-shot adaptive branching simulator。

## Detector Error Model

`DetectorErrorModelGenerator` 使用 `Detector` 和 `LogicalObservable` 声明生成 DEM。`Detector`
声明是 detector matrix \(D\) 的行；每个 physical error 的 measurement flips 组成
measurement syndrome matrix \(\Omega\) 的列；生成出的 DEM edge materialize detector error
matrix \(H=D\Omega\) 的列及其 logical observable flips。声明可以显式传入，也可以作为 circuit
operations 存在：

```text
Operation.detector(measurement_keys, detector_id=...)
Operation.observable_include(observable_id, measurement_keys)
```

生成器对每个 noise location 和每个非 identity/flip 事件做单错误传播：

```text
reference effect -> s_ref, o_ref
single injected event -> s_event, o_event
edge = error(p_event) xor(s_ref, s_event) xor(o_ref, o_event)
```

输出为 Stim-like DEM 行：

```text
error(p) D0 D3 L0
```

`edges_by_location()` 返回：

```text
dict[str, list[DetectorErrorEdge]]
```

如果一个 location 产生多条 edge，location sensitivity 可以按 edge probability 权重投影到
detector graph：

```text
graph = dem.project_sensitivities_to_detector_graph(result.sensitivities)
```

`project_hotspots_to_edges(hotspots)` 接收 location-level hotspot mapping，并返回按
`(location_id, event)` keyed 的 edge-level hotspot mapping。

DEM generation 要求 detector/observable effects 在 reference 和单错误传播中可确定。随机裸测量不能直接作为
detector parity。

## PyMatching Decoder

PyMatching 接口把 detector syndrome masks 送入由 DEM 构造的 matching decoder：

```text
batch.detectors -> decoder.decode_batch_masks(batch) -> correction masks
```

给定 DEM edge 集合 `E`，构造 detector error matrix：

```text
H[i,e] = 1 iff edge e violates detector i
```

以及 logical fault matrix：

```text
G[a,e] = 1 iff edge e flips logical observable a
```

边权为：

```text
w_e = log((1-p_e) / p_e)
```

当前接口要求 DEM graphlike：每条 edge 至多连接两个 detector。没有 detector 的纯 logical
edge 会被拒绝，因为 matching decoder 无法从 syndrome 中恢复这种错误。

## DEM Hotspot Mode

`DemHotspotEstimator` 不执行 stabilizer circuit，而是在 detector error model 上直接采样。
`DemFaultScopeSimulator(circuit)` 是同一 Rust DEM sampler 的 circuit 入口：构造时先生成
DEM sampling edges，运行时仍然只采样 DEM edges，不回到 forward packed trajectory。
每条 DEM edge 是独立 Bernoulli instruction：

```text
f_e ~ Bernoulli(p_e)
s_i = xor_{e where H[i,e] = 1} f_e
o_a = xor_{e where G[a,e] = 1} f_e
```

默认 loss 为 residual logical failure：

```text
failure = any_a(o_a xor C_a)
```

edge-level sensitivity 为：

```text
dJ_DEM / dp_e =
  E[(loss - baseline) * (f_e / p_e - (1-f_e) / (1-p_e))]
```

结果包含：

- `edge_sensitivities: dict[int, float]`
- `edge_hotspots: dict[int, float]`
- `sensitivities: dict[str, float]`
- `hotspots: dict[str, float]`
- `detector_graph_hotspots`

DEM mode 采用普通 DEM 的独立 edge sampling 语义。它适合 detector-graph 或 decoder-level
hotspot 扫描，但不保留原始 forward trajectory 中同一物理 location 下多个 Pauli event 的互斥采样语义。

## Stim Import Subset

`parse_stim_circuit(...)` 和 `load_stim_file(...)` 支持 flattened Stim 文本子集。导入结果包含：

- `circuit`
- `detectors`
- `observables`
- `measurement_keys`

导入后的 `circuit` 也包含 detector/observable operations，因此可以直接用于 batch 或 DEM workflow。

支持常见 Clifford、Pauli gate、reset、measurement、MPP、Pauli/depolarizing noise、
`PAULI_CHANNEL_1`、`PAULI_CHANNEL_2`、`DETECTOR` 和 `OBSERVABLE_INCLUDE`。

不支持 `REPEAT` block、复杂 target modifiers、完整 coordinate-shift 累积、feedback targets、
`CORRELATED_ERROR` / `ELSE_CORRELATED_ERROR` 等 Stim 高级语义。遇到未支持语法会抛出
`StimImportError`。

## 验证标准

实现和文档示例应满足：

- 无噪声或零 loss 时，mean loss 和 hotspot 接近 0。
- 单比特 bit-flip toy model 中，若 `J(lambda)=lambda`，则 `dJ/dlambda=1`。
- 对称纠错电路中，几何等价 noise locations 在统计误差内给出相近 hotspot。
- 人为提高某个时空位置的噪声率后，该位置或相邻 detector 区域应在 top-k hotspot 中出现。
- API 文档中的 Python 示例应能在构建好的 `.venv` 中运行。
