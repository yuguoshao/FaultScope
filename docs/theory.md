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
- \(D_k\)：shot \(k\) 的 detector record；它是所有 detector bits \(D_{k,i}\) 组成的向量。
- \(D_i\)：detector \(i\) 的 packed detector mask。
- \(D_{k,i}\)：shot \(k\) 上 detector \(i\) 的 bit，满足 \(D_{k,i}=\operatorname{bit}_k(D_i)\)。
- \(L_k\)：shot \(k\) 的 logical observable record；它是所有 logical observable bits \(L_{k,a}\) 组成的向量。
- \(L_{k,a}\)：shot \(k\) 上 logical observable \(a\) 的 bit，满足 \(L_{k,a}=\operatorname{bit}_k(O_a)\)。
- \(O_a\)：logical observable \(a\) 的 packed observable mask。
- \(C_a\)：decoder 预测的 logical correction mask。
- \(F\)：loss mask；第 \(k\) 位为 1 表示 shot \(k\) 贡献 loss。
- \(A\)：all-shot mask，低 \(N\) 位为 1，用来裁剪未使用 bit。

这里的 \(L_k\) 表示 logical observable record，不是 loss。本文把 shot-level loss 写成
\(L_{\mathrm{loss}}(\tau)\)，把 batch-level loss 写成 packed mask \(F\)。

NPSim 的 batch representation 把每个 boolean shot value 存成一个 Python integer 或 Rust
`Mask`。因此：

\[
\operatorname{bit}_k(X) = (X \gg k) \mathbin{\&} 1,
\qquad
\operatorname{popcount}(X) = \text{number of set bits in } X.
\]

## Packed masks: measurements, detectors, observables

Python API 暴露的 batch 结果不是逐 shot 的列表，而是一组 keyed packed masks。measurement、
detector、observable 都遵循同一个约定：dict 的 key 标识一个物理或逻辑量，dict 的 value 是一个
整数；整数第 \(k\) 位就是第 \(k\) 个 shot 上这个量的 boolean 值。

### Measurement masks

`batch.measurements` 的类型是 `dict[str, int]`。每个 key 是一次 measurement 或 reset-with-key
记录出的 measurement key，每个 value 是一个 packed measurement mask。对 key `m`，记这个整数为
\(M_m\)。它把同一个 measurement key 在 \(N\) 个 shots 中的 boolean 结果压到一个整数里：

\[
M_m = \sum_{k=0}^{N-1} m_{k,m}2^k,
\qquad
m_{k,m} = \operatorname{bit}_k(M_m).
\]

也就是说，\(M_m\) 的第 \(k\) 位就是第 \(k\) 个 shot 上 key `m` 的测量结果。若
`batch.measurements["m"] == 0b1010`，则 shot 1 和 shot 3 的结果为 1，shot 0 和 shot 2 的结果为
0。实际使用时仍应通过 bit operation 读取：

```text
((batch.measurements["m"] >> k) & 1)
```

measurement key 必须唯一。显式写 `Operation.measure(..., key="m")` 时使用给定 key；没有显式 key
的测量会按运行时顺序生成类似 `m0`, `m1`, ... 的 key。重复 key 会报错，因为一个 key 只能对应一个
packed mask。

`MeasurementBitFlip` noise 会在 measurement mask 记录前翻转相应 shots 的 measurement bit，因此
下游 detector、observable 和 `loss_mask_fn` 看到的都是已经包含 measurement noise 的
packed measurement mask。

### Detector masks

`batch.detectors` 的类型是 `dict[int, int]`。每个 key 是 detector id，每个 value 是 packed
detector mask。detector 是若干 measurement masks 的 bitwise XOR parity。若 detector \(i\)
依赖 measurement keys \(K_i\)，则：

\[
D_i = \bigoplus_{m\in K_i} M_m.
\]

这里的 XOR 是逐 bit 的：对每个 shot \(k\)，\(D_i\) 的第 \(k\) 位等于该 shot 上所有依赖测量结果的
parity。也就是：

\[
\operatorname{bit}_k(D_i)
=
\bigoplus_{m\in K_i}
\operatorname{bit}_k(M_m).
\]

### Observable masks

`batch.observables` 的类型是 `dict[int, int]`。每个 key 是 logical observable id，每个 value 是
packed observable mask。记 id 为 \(a\) 的 observable mask 为 \(O_a\)。它的第 \(k\) 位表示第
\(k\) 个 shot 上该 logical observable 是否翻转：

\[
O_a = \sum_{k=0}^{N-1} o_{k,a}2^k,
\qquad
o_{k,a} = \operatorname{bit}_k(O_a).
\]

observable mask 可以来自 measurement keys、最终 Pauli frame projection，或二者的 XOR。对只由
measurement keys \(K_a\) 定义的 observable：

\[
O_a = \bigoplus_{m\in K_a} M_m.
\]

如果 observable 还包含 final Pauli frame 项，设该 frame projection 产生的 packed mask 为
\(Q_a\)，则：

\[
O_a =
\left(
\bigoplus_{m\in K_a} M_m
\right)
\oplus Q_a.
\]

`Operation.observable_include(a, keys)` 是电路内声明形式：它把这些 measurement keys 的 parity XOR
到 `batch.observables[a]`。同一个 observable id 可以通过多条 include 逐次 XOR 累积。通过
`BatchForwardNoiseAwareSimulator(..., observables=(LogicalObservable(...),))` 传入的
`LogicalObservable` 则是在 batch 末尾从 `measurement_keys` 和可选 final Pauli frame projection
计算出 \(O_a\)。

observable mask 不是 decoder correction，也不是 residual logical loss。decoder 返回的 correction
mask \(C_a\) 使用同样的 packed convention；默认 logical loss 先计算 residual：

\[
R_a = O_a \oplus C_a.
\]

然后把所有 residual observables 做 OR 得到 loss mask \(F\)。因此 \(O_a\) 表示 simulator 观测到的
logical observable，\(C_a\) 表示 decoder 预测的修正，\(F\) 才是最终参与 hotspot 估计的 loss。

## 前向目标函数

给定所有 noise rates \(\lambda = \{\lambda_l\}\)，目标函数是 shot-level loss 的期望：

\[
J(\lambda) =
\mathbb{E}_{\tau \sim P_\lambda}
\left[
L_{\mathrm{loss}}(\tau)
\right].
\]

在 forward batch API 中，自定义 loss 由 packed mask 表示。forward estimator 支持一参或两参
callback：

```text
F = loss_mask_fn(batch)
F = loss_mask_fn(batch, corrections)
```

DEM estimator 的 custom loss callback 固定接收两参：

```text
F = loss_mask_fn(batch, corrections)
```

如果没有 decoder 或 `correction_mask_fn`，`corrections` 是空 mapping。

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

Detector Error Model, 简写 DEM，是把原始 circuit 的噪声过程压缩成 detector syndrome 和 logical
observable flips 的二进制概率模型。它不再保存完整 stabilizer state、逐 shot measurement history
或 final Pauli frame；它只保存“哪些独立错误机制会以多大概率触发，以及触发后会翻转哪些
detectors 和 logical observables”。

形式上，一个 DEM 可以写成：

\[
\mathcal{M}_{\mathrm{DEM}}
=
(\mathcal{D},\mathcal{O},\mathcal{E}),
\]

其中：

- \(\mathcal{D}\)：detector 集合。detector bit 是 syndrome bit，表示一组 measurement parity 是否异常。
- \(\mathcal{O}\)：logical observable 集合。logical observable bit 表示错误是否导致对应 logical observable 翻转。
- \(\mathcal{E}\)：DEM edge 集合。每条 edge 是一个独立的 Bernoulli 错误机制。

在一个 DEM shot 中，模型先为每条 edge \(e\in\mathcal{E}\) 采样一个发生变量 \(f_e\)。随后所有发生的
edges 通过 XOR 叠加出 detector syndrome \(D\) 和 logical observable flip record \(O\)：

\[
D_i = \bigoplus_{e:\ i\in\Delta D_e} f_e,
\qquad
O_a = \bigoplus_{e:\ a\in\Delta L_e} f_e.
\]

因此 DEM 描述的是 \(P(D,O)\)，也就是 detector syndrome 和 logical flips 的联合分布。decoder 只能
看到 detector syndrome \(D\)，并尝试预测 logical correction \(C\)；默认 loss 比较的是 residual
logical flips \(O\oplus C\)。

这里的 edge \(e\) 不是 circuit gate，也不是某个 shot 中已经发生的错误；它是 DEM 中的一条错误机制
instruction。运行 DEM sampler 时，每条 edge 会被独立采样一次，决定这一类错误机制在当前 shot
是否发生。

一条 edge \(e\) 记录：

\[
e =
(p_e,\Delta D_e,\Delta L_e,\operatorname{location\_id},\operatorname{event\_label},\operatorname{tags}).
\]

各字段含义：

- \(p_e\)：edge probability，即这条 DEM instruction 在一个 shot 中发生的概率。
- \(\Delta D_e\)：如果 edge \(e\) 发生，需要翻转的 detector id 集合。
- \(\Delta L_e\)：如果 edge \(e\) 发生，需要翻转的 logical observable id 集合。
- `location_id`：产生这条 edge 的原始 `NoiseLocation.id`，用于把 edge-level sensitivity 聚合回物理位置。
- `event_label`：原始噪声事件标签，例如 Pauli event `"X"`、`"YZ"`，或 measurement bit flip 的 `true`。
- `tags`：从原始 noise location 继承的 metadata，用于按 qubit、round、gate、operation 等维度聚合。

在公式里，\(e\) 常同时被当作 edge 的索引使用。例如 `edge_event_masks[e]` 表示第 \(e\) 条 DEM
edge 在一批 shots 中的 packed occurrence mask。若引入 Bernoulli 发生变量 \(f_e\)，则：

\[
f_e =
\begin{cases}
1, & \text{edge } e \text{ occurred in this shot},\\
0, & \text{otherwise},
\end{cases}
\qquad
f_e \sim \operatorname{Bernoulli}(p_e).
\]

Stim-like text:

```text
error(p_e) D0 D3 L0
```

含义是：这条 edge 的 \(p_e\) 是 `p_e`，\(\Delta D_e=\{0,3\}\)，\(\Delta L_e=\{0\}\)。当
\(f_e=1\) 时，把 detector bits `D0`、`D3` 和 logical observable bit `L0` 全部 xor 一次；当
\(f_e=0\) 时，它不产生任何 flip。

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

Stim 的 `Circuit.detector_error_model(...)` 也默认采用这个严格约束：detectors 必须在 noiseless
execution 下是 deterministic 的。Stim 另有 `allow_gauge_detectors=True` 选项；开启后，某些
non-deterministic detectors 会被当作 gauge degrees of freedom 处理，通过 Gaussian elimination 从
error model 中消去，并可能引入类似 `error(0.5) D_i D_j` 的 gauge relation。这个功能针对的是
gauge detectors，不等同于把任意无法预测的裸随机测量直接保留成普通 detector。logical observables
仍然必须是 deterministic。

NPSim 当前 DEM generation 走严格路径：不暴露 Stim 式 gauge detector elimination。也就是说，
detector 和 observable 声明必须能在 reference / injected propagation 中得到确定 effect；否则应该
调整 detector 定义或先把随机自由度改写成确定的 syndrome relation。

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
