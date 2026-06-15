# 前向噪声感知 Stabilizer 模拟器

本实现用于纠错协议的噪声热点研究，不使用 Heisenberg picture，也不做
observable back-propagation。模拟器按纠错电路的真实时序前向执行：

1. Clifford gate 前向更新 stabilizer tableau 和 Pauli frame。
2. 噪声位置采样 stochastic Pauli-compatible event。
3. 噪声 event 作用到当前 stabilizer state 和 Pauli frame。
4. 测量产生 measurement record。
5. detector function 生成 syndrome / detector record。
6. decoder 根据 detector record 输出 correction。
7. loss function 根据完整 trajectory 计算 logical failure。

## 目标函数

一次 trajectory 记为

```text
tau = (e_1, e_2, ..., e_M, m_1, m_2, ..., m_R)
```

其中 `e_l` 是第 `l` 个噪声位置的采样事件，`m_r` 是测量结果。
目标函数为 shot-level loss 的期望：

```text
J(lambda) = E_tau[L(tau)].
```

默认用法中 `L(tau)` 是 logical failure indicator，但 API 允许用户传入任意
`loss_fn(trajectory, decoded)`。

## 噪声率导数

每个噪声位置 `l` 绑定一个可微事件分布：

```text
e_l ~ p_l(e; lambda_l)
```

模拟器要求每个噪声模型实现：

```python
sample(rng, rate) -> event
score(event, rate) -> d log p(event; rate) / d rate
apply(event, state, frame, qubits) -> None
```

使用 score-function estimator：

```text
dJ / d lambda_l = E[(L(tau) - b) s_l(tau)]
s_l(tau) = d log p_l(e_l; lambda_l) / d lambda_l
```

其中 baseline `b` 默认取 batch mean loss。

## 热点分数

模拟器输出 signed sensitivity：

```text
S_l = dJ / d lambda_l
```

默认热点分数为：

```text
H_l = |S_l|
```

并按以下标签聚合：

- qubit-level: `result.by_qubit`
- round-level: `result.by_round`
- gate-level: `result.by_gate`
- operation-level: `result.by_operation`

## 当前支持范围

- Clifford gate: `H`, `S`, `S†`, `CX`, `CZ`, `SWAP`
- ideal Pauli gate: `X`, `Y`, `Z`, sparse Pauli strings
- Z/X/Y single-qubit measurement
- arbitrary Pauli-string measurement
- reset to Z/X/Y basis eigenstates
- Bernoulli Pauli noise
- weighted Pauli-channel noise
- single-qubit depolarizing noise
- two-qubit depolarizing noise
- measurement bit-flip noise
- bit-packed batch sampler for deterministic-syndrome QEC fast paths
- detector error model generation by single-error propagation
- Stim text subset import into `Circuit`, `Detector`, and `LogicalObservable`
- repetition-code reference decoder and experiment builder

非 Pauli、非 Clifford 噪声不直接进入 stabilizer simulator；初版应先做
Pauli twirling 或替换为 stabilizer-compatible stochastic channel。

## Batch sampler 快速路径

`BatchForwardNoiseAwareSimulator` 使用一个理想 stabilizer tableau 加上 bit-packed
Pauli frames 执行多条 trajectory。第 `k` 个 shot 存在整数 mask 的第 `k` 位中：

- `X_frame[q]`: qubit `q` 上是否有 X 分量。
- `Z_frame[q]`: qubit `q` 上是否有 Z 分量。
- `M[key]`: measurement key 的测量结果。
- `E[l]`: 噪声位置 `l` 是否采样到 error / flip event。

它使用与逐 shot 模拟器相同的 score-function estimator，只是用 `popcount`
在 bit mask 上一次性归约 loss 和 score。该快速路径适合理想 syndrome 测量确定的
QEC 电路；如果理想测量本身会产生不同 tableau 分支，应使用逐 shot 引擎。

## Detector error model

`DetectorErrorModelGenerator` 使用结构化 `Detector` 和 `LogicalObservable`
声明生成 detector error model。每个 detector 是 measurement keys 的 parity；
logical observable 可以是 measurement parity，也可以是最终 Pauli frame 上某个
Pauli observable 的翻转。

`DETECTOR` / `OBSERVABLE_INCLUDE` 也可以作为 circuit operations 存在。逐 shot
simulator 会写入 `trajectory.detectors` 和 `trajectory.observables`；batch sampler
会写入 `batch.detectors` 和 `batch.observables`。DEM 生成器在未显式传入声明时会从
circuit operations 中自动读取这些语义。

生成器对每个噪声位置和每个非 identity 事件做单错误传播：

```text
reference run -> D_ref, L_ref
single injected event -> D_event, L_event
edge = error(p_event) xor(D_ref, D_event) xor(L_ref, L_event)
```

输出为 Stim-like 文本行：

```text
error(p) D0 D3 L0
```

该 DEM 层不改变热点估计公式；它提供从 location-level sensitivity 到 detector graph
的投影方式：

```text
graph = dem.project_sensitivities_to_detector_graph(result.sensitivities)
```

如果一个 noise location 产生多条 DEM edge，`S_l` 会按 edge 概率占比分到
edge-level sensitivity。返回值同时包含 edge-level hotspot、按
`(detectors, observables)` 聚合的 detector-edge hotspot、按 detector node 聚合的
node hotspot，以及按 logical observable 聚合的 hotspot。当前 DEM 生成也要求理想/单错误
测量确定，不处理随机 tableau 分支。

## Stim import subset

`parse_stim_circuit` / `load_stim_file` 支持常见 `.stim` 文本子集导入。导入结果包含：

- `circuit`: 本项目的前向 `Circuit`
- `detectors`: 从 `DETECTOR rec[-k] ...` 生成的结构化 detector parity
- `observables`: 从 `OBSERVABLE_INCLUDE(id) rec[-k] ...` 生成的 logical observable parity
- `measurement_keys`: Stim measurement record 到内部 measurement key 的顺序映射

导入后的 `circuit` 内也包含 `detector` / `observable_include` operations，因此可以直接用于 trajectory、batch 和 DEM 工作流。

当前支持 Clifford gate、reset、measurement、MPP、Pauli/depolarizing noise、
Pauli channel noise、`DETECTOR` 和 `OBSERVABLE_INCLUDE` 的常见形式。导入器不展开
`REPEAT` block，也不尝试兼容 Stim 的完整 target/feedback/correlated-error 语义；
遇到未支持语法会抛出 `StimImportError`。
