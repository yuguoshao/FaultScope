# NPSim 理论原理与公式细节

本文系统整理 NPSim 的数学模型、score-function estimator、packed batch 计数公式、
DEM 生成与 DEM hotspot 计算。它描述的是当前公开 runtime 的理论接口：实际 Python
回调接收 batch mask 对象，而不是逐 shot 的可变 trajectory 对象。

实现概览和运行时边界可参考 [NPSim 实现概览](implementation_overview.md)。

## 符号与对象

一次 Monte Carlo batch 有 \(N\) 个 shots。第 \(k\) 个 shot 对应一个概念上的前向采样路径：

\[
\tau_k =
\left(
e_{k,1}, \ldots, e_{k,M},
m_{k,1}, \ldots, m_{k,R},
D_k,
L_k
\right).
\]

主要符号：

- \(l\)：一个物理 noise location，带有 id、qubits、rate 和 tags。
- \(\lambda_l\)：noise location \(l\) 的事件发生率。
- \(e_{k,l}\)：shot \(k\) 在 location \(l\) 的 sampled event。
- \(E_l\)：packed event mask；第 \(k\) 位为 1 表示 shot \(k\) 在 \(l\) 发生非 identity/flip event。
- \(m_{k,r}\)：第 \(r\) 个 measurement 在 shot \(k\) 的 classical bit。
- \(M_{\text{key}}\)：packed measurement mask；第 \(k\) 位为 measurement key 在 shot \(k\) 的值。
- \(D_i\)：detector \(i\) 的 packed detector mask。
- \(O_a\)：logical observable \(a\) 的 packed observable mask。
- \(C_a\)：decoder 预测的 logical correction mask。
- \(F\)：loss mask；第 \(k\) 位为 1 表示 shot \(k\) 贡献 loss。
- \(A\)：all-shot mask，低 \(N\) 位为 1，用来裁剪未使用 bit。

NPSim 的 batch representation 把每个 boolean shot value 存成一个 Python integer 或 Rust
`Mask`。因此：

\[
\operatorname{bit}_k(X) = (X \gg k) \mathbin{\&} 1,
\qquad
\operatorname{popcount}(X) = \text{number of set bits in } X.
\]

## 前向目标函数

给定所有 noise rates \(\lambda = \{\lambda_l\}\)，目标函数是 shot-level loss 的期望：

\[
J(\lambda) =
\mathbb{E}_{\tau \sim P_\lambda}
\left[
L_{\mathrm{loss}}(\tau)
\right].
\]

在 batch API 中，loss 由 packed mask 表示：

```text
F = loss_mask_fn(batch)
F = loss_mask_fn(batch, corrections)
```

默认 logical loss 使用 declared logical observables 和 decoder correction：

\[
R_a = O_a \oplus C_a,
\qquad
F = \bigvee_a R_a.
\]

如果没有 decoder，correction map 为空，相当于所有 \(C_a = 0\)。因此默认 loss 是“任一 residual
logical observable 为 1”的 indicator。

batch 中的经验 loss 为：

\[
\operatorname{loss\_count}
= \operatorname{popcount}(F \mathbin{\&} A),
\qquad
\operatorname{mean\_loss}
= \frac{\operatorname{loss\_count}}{N}.
\]

## Score-function 推导

对某个 location \(l\)，事件分布为：

\[
e_l \sim p_l(e;\lambda_l).
\]

score 定义为：

\[
s_l(\tau)
=
\frac{\partial \log p_l(e_l;\lambda_l)}
{\partial \lambda_l}.
\]

对目标函数求导：

\[
\frac{\partial J}{\partial \lambda_l}
=
\frac{\partial}{\partial \lambda_l}
\sum_\tau P_\lambda(\tau)L_{\mathrm{loss}}(\tau)
=
\sum_\tau
P_\lambda(\tau)L_{\mathrm{loss}}(\tau)
\frac{\partial \log P_\lambda(\tau)}{\partial \lambda_l}.
\]

NPSim 的 stochastic noise locations 独立采样，且 measurement 随机性在给定噪声事件后不显式依赖
\(\lambda_l\)。因此：

\[
\frac{\partial \log P_\lambda(\tau)}{\partial \lambda_l}
=
\frac{\partial \log p_l(e_l;\lambda_l)}{\partial \lambda_l}
=
s_l(\tau).
\]

于是：

\[
\frac{\partial J}{\partial \lambda_l}
=
\mathbb{E}
\left[
L_{\mathrm{loss}}(\tau)s_l(\tau)
\right].
\]

加入 baseline \(b\)：

\[
\mathbb{E}
\left[
\left(L_{\mathrm{loss}}(\tau)-b\right)s_l(\tau)
\right]
=
\mathbb{E}
\left[
L_{\mathrm{loss}}(\tau)s_l(\tau)
\right]
-
b\,\mathbb{E}[s_l(\tau)].
\]

而：

\[
\mathbb{E}[s_l]
=
\sum_e
p_l(e;\lambda_l)
\frac{\partial \log p_l(e;\lambda_l)}{\partial \lambda_l}
=
\sum_e
\frac{\partial p_l(e;\lambda_l)}{\partial \lambda_l}
=
\frac{\partial}{\partial \lambda_l}
\sum_e p_l(e;\lambda_l)
=
0.
\]

所以 baseline 不改变真实期望，只影响有限样本估计的方差。当前实现默认：

\[
b = \operatorname{mean\_loss}.
\]

也可以显式传入 numeric baseline。

## 噪声模型与 score

当前 hotspot sensitivity 对“是否发生非 identity/flip event”的 rate 求导。对所有 Pauli mixture
类噪声，非 identity event 的条件分布由模型内部权重决定；score 只依赖事件是否发生，不对条件权重求导。

实现中使用裁剪后的概率：

\[
p = \operatorname{clamp}(\lambda, 10^{-12}, 1-10^{-12}),
\qquad
s_{\mathrm{event}} = \frac{1}{p},
\qquad
s_{\mathrm{no\ event}} = -\frac{1}{1-p}.
\]

### Bernoulli Pauli

`BernoulliPauliNoise(P)`:

\[
\Pr(I)=1-\lambda,
\qquad
\Pr(P)=\lambda.
\]

score:

\[
s =
\begin{cases}
\frac{1}{\lambda}, & \text{if } P \text{ occurred},\\
-\frac{1}{1-\lambda}, & \text{if } I \text{ occurred}.
\end{cases}
\]

### Measurement Bit Flip

`MeasurementBitFlip()` attached to a measurement:

\[
\Pr(\text{no flip})=1-\lambda,
\qquad
\Pr(\text{flip})=\lambda.
\]

score:

\[
s =
\begin{cases}
\frac{1}{\lambda}, & \text{if flip occurred},\\
-\frac{1}{1-\lambda}, & \text{otherwise}.
\end{cases}
\]

The event mask records flip occurrence.

### Single-qubit Depolarizing

`SingleQubitDepolarizing()`:

\[
\Pr(I)=1-\lambda,
\qquad
\Pr(X)=\Pr(Y)=\Pr(Z)=\frac{\lambda}{3}.
\]

NPSim records one event bit for \(X/Y/Z\) occurrence:

\[
s =
\begin{cases}
\frac{1}{\lambda}, & \text{if event is in } \{X,Y,Z\},\\
-\frac{1}{1-\lambda}, & \text{if event is } I.
\end{cases}
\]

### Two-qubit Depolarizing

`TwoQubitDepolarizing()` samples one of the 15 non-identity two-qubit Paulis when an event occurs:

\[
\Pr(II)=1-\lambda,
\qquad
\Pr(P)=\frac{\lambda}{15}
\quad
\text{for } P\in\{IX,IY,\ldots,ZZ\}.
\]

score:

\[
s =
\begin{cases}
\frac{1}{\lambda}, & \text{if } P \ne II,\\
-\frac{1}{1-\lambda}, & \text{if } P=II.
\end{cases}
\]

### Pauli Channel

`PauliChannel({P_i: w_i})`:

\[
\Pr(I\cdots I)=1-\lambda,
\qquad
\Pr(P_i)=
\lambda
\frac{w_i}{\sum_j w_j}.
\]

score for rate sensitivity:

\[
s =
\begin{cases}
\frac{1}{\lambda}, & \text{if any non-identity channel event occurred},\\
-\frac{1}{1-\lambda}, & \text{otherwise}.
\end{cases}
\]

The derivative is with respect to \(\lambda\), not \(w_i\).

## Packed Forward 计数公式

Forward hotspot aggregation in `compute_packed_estimate` uses only packed counts.

For a location \(l\):

\[
\begin{aligned}
E &= E_l,\\
F_A &= F \mathbin{\&} A,\\
\operatorname{loss\_count} &= \operatorname{popcount}(F_A),\\
\operatorname{event\_count} &= \operatorname{popcount}(E),\\
\operatorname{no\_event\_count} &= N-\operatorname{event\_count},\\
\operatorname{loss\_event\_count} &= \operatorname{popcount}(F_A \mathbin{\&} E),\\
\operatorname{loss\_no\_event\_count}
&=
\operatorname{loss\_count}
-
\operatorname{loss\_event\_count}.
\end{aligned}
\]

Scores:

\[
p = \operatorname{clamp}(\lambda_l, 10^{-12}, 1-10^{-12}),
\qquad
s_1 = \frac{1}{p},
\qquad
s_0 = -\frac{1}{1-p}.
\]

The summed score over loss shots:

\[
\operatorname{sum\_loss\_score}
=
\operatorname{loss\_event\_count}\,s_1
+
\operatorname{loss\_no\_event\_count}\,s_0.
\]

The summed score over all shots:

\[
\operatorname{sum\_score}
=
\operatorname{event\_count}\,s_1
+
\operatorname{no\_event\_count}\,s_0.
\]

With baseline \(b\):

\[
\operatorname{sensitivity}_l
=
\frac{
\operatorname{sum\_loss\_score}
-
b\,\operatorname{sum\_score}
}{N},
\qquad
\operatorname{hotspot}_l
=
\left|\operatorname{sensitivity}_l\right|.
\]

This is exactly the finite-sample estimator:

\[
\operatorname{sensitivity}_l
=
\frac{1}{N}
\sum_{k=0}^{N-1}
(F_k-b)s_{k,l},
\qquad
F_k=\operatorname{bit}_k(F_A).
\]

## Tag 聚合

Forward result location hotspots are aggregated by metadata tags:

\[
\begin{aligned}
\operatorname{by\_qubit}[q]
&=
\sum_{l:\ q\in\operatorname{qubits}(l)}
\operatorname{hotspot}_l,\\
\operatorname{by\_round}[r]
&=
\sum_{l:\ \operatorname{tags}(l)[\text{"round"}]=r}
\operatorname{hotspot}_l,\\
\operatorname{by\_gate}[g]
&=
\sum_{l:\ \operatorname{tags}(l)[\text{"gate"}]=g}
\operatorname{hotspot}_l,\\
\operatorname{by\_operation}[o]
&=
\sum_{l:\ \operatorname{tags}(l)[\text{"operation"}]=o}
\operatorname{hotspot}_l.
\end{aligned}
\]

These aggregations use absolute hotspot values, not signed sensitivities.

## Detector Error Model 语义

DEM 将物理 noise event 映射到 detector flips 和 logical observable flips。每条 edge 记录：

\[
e =
(p_e,\Delta D_e,\Delta L_e,\operatorname{location\_id},\operatorname{event\_label},\operatorname{tags}).
\]

Stim-like text:

```text
error(p_e) D0 D3 L0
```

含义：当 edge \(e\) 发生时，把列出的 detector bits 和 logical observable bits 全部 xor 一次。

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

\[
f_e \sim \operatorname{Bernoulli}(p_e),
\qquad
D_i = \bigoplus_{e:\ i\in\Delta D_e} f_e,
\qquad
O_a = \bigoplus_{e:\ a\in\Delta L_e} f_e.
\]

默认 DEM loss：

\[
R_a = O_a \oplus C_a,
\qquad
F = \bigvee_a R_a.
\]

如果没有 decoder/correction，\(C_a=0\)。

注意：DEM sampling 的多个 edges 独立采样；它符合普通 DEM 语义，但不保留同一 physical
location 下多个 Pauli events 在 forward trajectory 中的互斥 categorical 关系。

## DEM hotspot 公式

DEM edge-level sensitivity 对 edge probability \(p_e\) 求导。实现使用与 forward path 相同的 packed
counting formula，只是把 `event_mask` 换成 `edge_event_masks[e]`，把 rate 换成 edge probability：

\[
\begin{aligned}
p &= \operatorname{clamp}(p_e, 10^{-12}, 1-10^{-12}),\\
s_1 &= \frac{1}{p},\\
s_0 &= -\frac{1}{1-p},\\
\operatorname{sensitivity}_e
&=
\frac{
\operatorname{sum\_loss\_score}_e
-
b\,\operatorname{sum\_score}_e
}{N},\\
\operatorname{hotspot}_e
&=
\left|\operatorname{sensitivity}_e\right|.
\end{aligned}
\]

其中：

\[
\begin{aligned}
\operatorname{sum\_loss\_score}_e
&=
\operatorname{popcount}(F_A \mathbin{\&} E_e)\,s_1
+
\left(
\operatorname{popcount}(F_A)
-
\operatorname{popcount}(F_A \mathbin{\&} E_e)
\right)s_0,\\
\operatorname{sum\_score}_e
&=
\operatorname{popcount}(E_e)\,s_1
+
\left(N-\operatorname{popcount}(E_e)\right)s_0.
\end{aligned}
\]

## DEM location 聚合

同一 `location_id` 的 DEM edges 被聚合成 location-level sensitivity。设 location \(l\) 对应 edge
集合 \(G_l\)：

\[
P_l = \sum_{e\in G_l} p_e.
\]

若 \(P_l>0\)：

\[
\operatorname{weight}_e = \frac{p_e}{P_l}.
\]

若 \(P_l=0\)，实现使用均匀权重：

\[
\operatorname{weight}_e = \frac{1}{|G_l|}.
\]

location sensitivity:

\[
\operatorname{sensitivity}_l
=
\sum_{e\in G_l}
\operatorname{weight}_e\,
\operatorname{sensitivity}_e,
\qquad
\operatorname{hotspot}_l
=
\left|\operatorname{sensitivity}_l\right|.
\]

DEM tag aggregations 按 `location_id` 聚合后的 \(\operatorname{hotspot}_l\) 计算。

## Detector graph 投影

`project_sensitivities_to_detector_graph(...)` 和 DEM result 中的 `detector_graph_hotspots`
把 location 或 edge sensitivity 投影到 detector graph。

每条 DEM edge 有 key：

\[
(\operatorname{detector\_tuple},\operatorname{observable\_tuple}).
\]

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

PyMatching decoder 从 graphlike DEM 构造 matching problem。给定 edges \(e=0,\ldots,E-1\)，
check matrix 为：

\[
H_{i,e} =
\begin{cases}
1, & \text{if detector } D_i \text{ is flipped by edge } e,\\
0, & \text{otherwise}.
\end{cases}
\]

fault matrix 为：

\[
F_{a,e} =
\begin{cases}
1, & \text{if logical observable } L_a \text{ is flipped by edge } e,\\
0, & \text{otherwise}.
\end{cases}
\]

edge weight:

\[
w_e = \log \frac{1-p_e}{p_e}.
\]

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
