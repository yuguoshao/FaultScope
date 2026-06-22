# NPSim 理论原理与公式细节

本文系统整理 NPSim 的数学模型、score-function estimator、packed batch 计数公式、
DEM 生成与 DEM hotspot 计算。它描述的是当前公开 runtime 的理论接口：实际 Python
回调接收 batch mask 对象，而不是逐 shot 的可变 trajectory 对象。

实现概览和运行时边界可参考 [NPSim 实现概览](implementation_overview.md)。

## 符号与对象

一次 Monte Carlo batch 有 `N` 个 shots。第 `k` 个 shot 对应一个概念上的前向采样路径：

```text
tau_k = (e_{k,1}, ..., e_{k,M}, m_{k,1}, ..., m_{k,R}, D_k, L_k)
```

主要符号：

- `l`: 一个物理 noise location，带有 id、qubits、rate 和 tags。
- `lambda_l`: noise location `l` 的事件发生率。
- `e_{k,l}`: shot `k` 在 location `l` 的 sampled event。
- `E_l`: packed event mask；第 `k` 位为 1 表示 shot `k` 在 `l` 发生非 identity/flip event。
- `m_{k,r}`: 第 `r` 个 measurement 在 shot `k` 的 classical bit。
- `M_key`: packed measurement mask；第 `k` 位为 measurement key 在 shot `k` 的值。
- `D_i`: detector `i` 的 packed detector mask。
- `O_a`: logical observable `a` 的 packed observable mask。
- `C_a`: decoder 预测的 logical correction mask。
- `F`: loss mask；第 `k` 位为 1 表示 shot `k` 贡献 loss。
- `A`: all-shot mask，低 `N` 位为 1，用来裁剪未使用 bit。

NPSim 的 batch representation 把每个 boolean shot value 存成一个 Python integer 或 Rust
`Mask`。因此：

```text
bit_k(X) = (X >> k) & 1
popcount(X) = number of set bits in X
```

## 前向目标函数

给定所有 noise rates `lambda = {lambda_l}`，目标函数是 shot-level loss 的期望：

```text
J(lambda) = E_{tau ~ P_lambda}[L_loss(tau)]
```

在 batch API 中，loss 由 packed mask 表示：

```text
F = loss_mask_fn(batch)
F = loss_mask_fn(batch, corrections)
```

默认 logical loss 使用 declared logical observables 和 decoder correction：

```text
R_a = O_a xor C_a
F = OR_a R_a
```

如果没有 decoder，correction map 为空，相当于所有 `C_a = 0`。因此默认 loss 是“任一 residual
logical observable 为 1”的 indicator。

batch 中的经验 loss 为：

```text
loss_count = popcount(F & A)
mean_loss = loss_count / N
```

## Score-function 推导

对某个 location `l`，事件分布为：

```text
e_l ~ p_l(e; lambda_l)
```

score 定义为：

```text
s_l(tau) = d log p_l(e_l; lambda_l) / d lambda_l
```

对目标函数求导：

```text
dJ / d lambda_l
  = d / d lambda_l sum_tau P_lambda(tau) L_loss(tau)
  = sum_tau P_lambda(tau) L_loss(tau) d log P_lambda(tau) / d lambda_l
```

NPSim 的 stochastic noise locations 独立采样，且 measurement 随机性在给定噪声事件后不显式依赖
`lambda_l`。因此：

```text
d log P_lambda(tau) / d lambda_l = d log p_l(e_l; lambda_l) / d lambda_l = s_l(tau)
```

于是：

```text
dJ / d lambda_l = E[L_loss(tau) s_l(tau)]
```

加入 baseline `b`：

```text
E[(L_loss(tau) - b) s_l(tau)]
  = E[L_loss(tau) s_l(tau)] - b E[s_l(tau)]
```

而：

```text
E[s_l]
  = sum_e p_l(e; lambda_l) d log p_l(e; lambda_l) / d lambda_l
  = sum_e d p_l(e; lambda_l) / d lambda_l
  = d / d lambda_l sum_e p_l(e; lambda_l)
  = d(1) / d lambda_l
  = 0
```

所以 baseline 不改变真实期望，只影响有限样本估计的方差。当前实现默认：

```text
b = mean_loss
```

也可以显式传入 numeric baseline。

## 噪声模型与 score

当前 hotspot sensitivity 对“是否发生非 identity/flip event”的 rate 求导。对所有 Pauli mixture
类噪声，非 identity event 的条件分布由模型内部权重决定；score 只依赖事件是否发生，不对条件权重求导。

实现中使用裁剪后的概率：

```text
p = clamp(rate, 1e-12, 1 - 1e-12)
score_event = 1 / p
score_no_event = -1 / (1 - p)
```

### Bernoulli Pauli

`BernoulliPauliNoise(P)`:

```text
Pr(I) = 1 - lambda
Pr(P) = lambda
```

score:

```text
s =  1 / lambda       if P occurred
s = -1 / (1-lambda)   if I occurred
```

### Measurement Bit Flip

`MeasurementBitFlip()` attached to a measurement:

```text
Pr(no flip) = 1 - lambda
Pr(flip) = lambda
```

score:

```text
s =  1 / lambda       if flip occurred
s = -1 / (1-lambda)   otherwise
```

The event mask records flip occurrence.

### Single-qubit Depolarizing

`SingleQubitDepolarizing()`:

```text
Pr(I) = 1 - lambda
Pr(X) = lambda / 3
Pr(Y) = lambda / 3
Pr(Z) = lambda / 3
```

NPSim records one event bit for `X`/`Y`/`Z` occurrence:

```text
s =  1 / lambda       if event in {X, Y, Z}
s = -1 / (1-lambda)   if event is I
```

### Two-qubit Depolarizing

`TwoQubitDepolarizing()` samples one of the 15 non-identity two-qubit Paulis when an event occurs:

```text
Pr(II) = 1 - lambda
Pr(P) = lambda / 15   for P in {IX, IY, ..., ZZ}
```

score:

```text
s =  1 / lambda       if P != II
s = -1 / (1-lambda)   if P = II
```

### Pauli Channel

`PauliChannel({P_i: w_i})`:

```text
Pr(I...I) = 1 - lambda
Pr(P_i) = lambda * w_i / sum_j w_j
```

score for rate sensitivity:

```text
s =  1 / lambda       if any non-identity channel event occurred
s = -1 / (1-lambda)   otherwise
```

The derivative is with respect to `lambda`, not `w_i`.

## Packed Forward 计数公式

Forward hotspot aggregation in `compute_packed_estimate` uses only packed counts.

For a location `l`:

```text
event_mask = E_l
loss = F & A
loss_count = popcount(loss)
event_count = popcount(event_mask)
no_event_count = N - event_count
loss_event_count = popcount(loss & event_mask)
loss_no_event_count = loss_count - loss_event_count
```

Scores:

```text
p = clamp(lambda_l, 1e-12, 1 - 1e-12)
event_score = 1 / p
no_event_score = -1 / (1 - p)
```

The summed score over loss shots:

```text
sum_loss_score =
    loss_event_count * event_score
  + loss_no_event_count * no_event_score
```

The summed score over all shots:

```text
sum_score =
    event_count * event_score
  + no_event_count * no_event_score
```

With baseline `b`:

```text
sensitivity_l = (sum_loss_score - b * sum_score) / N
hotspot_l = abs(sensitivity_l)
```

This is exactly the finite-sample estimator:

```text
sensitivity_l = (1/N) sum_k (F_k - b) s_{k,l}
```

where `F_k` is the kth bit of the clipped loss mask.

## Tag 聚合

Forward result location hotspots are aggregated by metadata tags:

```text
by_qubit[q] = sum_{l: q in location.qubits} hotspot_l
by_round[r] = sum_{l: location.tags["round"] == r} hotspot_l
by_gate[g] = sum_{l: location.tags["gate"] == g} hotspot_l
by_operation[o] = sum_{l: location.tags["operation"] == o} hotspot_l
```

These aggregations use absolute hotspot values, not signed sensitivities.

## Detector Error Model 语义

DEM 将物理 noise event 映射到 detector flips 和 logical observable flips：

```text
edge e = (p_e, DeltaD_e, DeltaL_e, location_id, event_label, tags)
```

Stim-like text:

```text
error(p_e) D0 D3 L0
```

含义：当 edge `e` 发生时，把列出的 detector bits 和 logical observable bits 全部 xor 一次。

## 单错误传播生成 DEM

`DetectorErrorModelGenerator` 对每个 physical noise event 做 single-error propagation。
概念上：

```text
reference run:
    D_ref, L_ref

single injected event (l, event):
    D_event, L_event

edge detectors = D_ref xor D_event
edge observables = L_ref xor L_event
edge probability = p_l(event)
```

只要 detector 或 observable effect 非空，就产生一条 DEM edge。DEM generation 要求相关
detector/observable effects 在 reference 和 injected run 中可确定；随机裸测量不能直接作为 detector。

对 PauliChannel 或 depolarizing noise，一个 physical location 可能产生多条 DEM edges。每条 edge
保留相同 `location_id`，event label 区分具体 Pauli event。

## DEM 独立 edge sampling

`DemBatchHotspotSimulator` 不执行原始 stabilizer circuit，而是把每条 DEM edge 作为独立 Bernoulli
instruction：

```text
f_e ~ Bernoulli(p_e)
D_i = xor_{e: i in DeltaD_e} f_e
O_a = xor_{e: a in DeltaL_e} f_e
```

默认 DEM loss：

```text
R_a = O_a xor C_a
F = OR_a R_a
```

如果没有 decoder/correction，`C_a = 0`。

注意：DEM sampling 的多个 edges 独立采样；它符合普通 DEM 语义，但不保留同一 physical
location 下多个 Pauli events 在 forward trajectory 中的互斥 categorical 关系。

## DEM hotspot 公式

DEM edge-level sensitivity 对 edge probability `p_e` 求导。实现使用与 forward path 相同的 packed
counting formula，只是把 `event_mask` 换成 `edge_event_masks[e]`，把 rate 换成 edge probability：

```text
p = clamp(p_e, 1e-12, 1 - 1e-12)
event_score = 1 / p
no_event_score = -1 / (1 - p)

sensitivity_e =
  (sum_loss_score_e - b * sum_score_e) / N

hotspot_e = abs(sensitivity_e)
```

其中：

```text
sum_loss_score_e =
    popcount(F & E_e) * event_score
  + (popcount(F) - popcount(F & E_e)) * no_event_score

sum_score_e =
    popcount(E_e) * event_score
  + (N - popcount(E_e)) * no_event_score
```

## DEM location 聚合

同一 `location_id` 的 DEM edges 被聚合成 location-level sensitivity。设 location `l` 对应 edge
集合 `G_l`：

```text
P_l = sum_{e in G_l} p_e
```

若 `P_l > 0`：

```text
weight_e = p_e / P_l
```

若 `P_l = 0`，实现使用均匀权重：

```text
weight_e = 1 / |G_l|
```

location sensitivity:

```text
sensitivity_l = sum_{e in G_l} weight_e * sensitivity_e
hotspot_l = abs(sensitivity_l)
```

DEM tag aggregations按 `location_id` 聚合后的 `hotspot_l` 计算。

## Detector graph 投影

`project_sensitivities_to_detector_graph(...)` 和 DEM result 中的 `detector_graph_hotspots`
把 location 或 edge sensitivity 投影到 detector graph。

每条 DEM edge 有 key：

```text
(detector_tuple, observable_tuple)
```

例如：

```text
((3, 4), ())      # detector graph edge
((5,), (0,))      # boundary/logical edge
```

聚合量包括：

- `by_detector_edge`: 按 `(detectors, observables)` 聚合 absolute hotspot。
- `signed_by_detector_edge`: 同一 key 的 signed sensitivity。
- `by_detector`: edge hotspot 平均分配到 touched detector nodes。
- `signed_by_detector`: signed sensitivity 平均分配到 detector nodes。
- `by_observable`: 按 logical observable 聚合 absolute hotspot。
- `signed_by_observable`: 按 logical observable 聚合 signed sensitivity。
- `by_location`: 按原始 location 聚合 absolute hotspot。
- `signed_by_location`: 按原始 location 聚合 signed sensitivity。

## PyMatching 数学接口

PyMatching decoder 从 graphlike DEM 构造 matching problem。给定 edges `e = 0..E-1`：

check matrix:

```text
H[i, e] = 1 iff detector D_i is flipped by edge e
```

fault matrix:

```text
F[a, e] = 1 iff logical observable L_a is flipped by edge e
```

edge weight:

```text
w_e = log((1 - p_e) / p_e)
```

PyMatching 输入 detector syndrome，输出 predicted logical correction masks：

```text
corrections = decoder.decode_batch_masks(batch)
```

限制：

- 每条 DEM edge 最多连接两个 detectors。
- 没有 detector 的纯 logical edge 会被拒绝，因为 syndrome 中不可见。
- Observable flips 通过 `faults_matrix` 传递给 PyMatching。

## 公式与实现对应关系

核心实现位置：

- Forward estimator: `compute_packed_estimate` in `crates/npsim-core/src/hotspot.rs`
- DEM estimator: `compute_dem_estimate` in `crates/npsim-core/src/hotspot.rs`
- DEM sampler: `run_dem_batch` in `crates/npsim-core/src/dem_sampling.rs`
- Noise event masks: sampling functions in `crates/npsim-core/src/packed.rs`

理论页中的 `event_count`、`loss_event_count`、`sum_loss_score`、`sum_score`、`baseline`
和 `sensitivity` 公式逐项对应这些实现。
