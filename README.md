# 前向噪声感知 Stabilizer 模拟器

本文档描述一个用于量子纠错协议噪声热点分析的前向 stabilizer 模拟器。模拟器不使用 Heisenberg picture，也不对观测量做反向传播；它在 Schrodinger picture 中按纠错电路的真实时间顺序执行 trajectory，记录噪声事件、测量结果、syndrome、decoder 输出和逻辑失败事件，并通过对局部噪声率求导得到噪声热点。

核心目标是估计每个局部噪声位置 `l` 对纠错协议 logical failure probability 的边际影响：

```math
S_l = \frac{\partial J}{\partial \lambda_l},
```

其中 `J` 是由完整前向 trajectory 定义的 loss 期望，`\lambda_l` 是第 `l` 个时空噪声位置的错误率。热点分数定义为：

```math
H_l = |S_l|.
```

## 1. 前向 trajectory 模型

考虑一个 stabilizer-compatible 纠错电路：

```math
C = O_T O_{T-1} \cdots O_1,
```

其中每个操作 `O_t` 可以是 Clifford gate、reset、measurement、classical record update、decoder-relevant detector construction，或局部噪声通道。

一次 Monte Carlo shot 生成一条前向 trajectory：

```math
\tau =
(e_1,e_2,\ldots,e_M,\;m_1,m_2,\ldots,m_R,\;s,\;\hat{\ell},\;\ell).
```

各符号含义为：

- `e_l`：第 `l` 个噪声位置采样到的噪声事件。
- `m_r`：第 `r` 个测量结果。
- `s`：由测量记录生成的 syndrome 或 detector record。
- `\hat{\ell}`：decoder 根据 `s` 给出的 logical correction。
- `\ell`：由前向 Pauli frame 得到的真实 logical error。

模拟器状态由两部分组成：

```math
\mathcal S_t,\quad F_t,
```

其中 `\mathcal S_t` 是 stabilizer tableau，表示当前 stabilizer state；`F_t` 是 Pauli frame，表示已经发生但可以用经典方式跟踪的 Pauli error。

Clifford gate 按前向共轭规则更新 tableau 和 Pauli frame。Pauli 噪声事件直接作用到当前 tableau，并同步更新 Pauli frame。测量按 stabilizer 测量规则产生 classical bit，并写入 measurement record。

## 2. 噪声模型与 score function

每个局部噪声位置 `l` 有一个可微事件分布：

```math
e_l \sim p_l(e;\lambda_l).
```

例如单比特 bit-flip noise：

```math
p_l(I)=1-\lambda_l,\qquad p_l(X)=\lambda_l.
```

单比特 depolarizing noise：

```math
p_l(I)=1-\lambda_l,\qquad
p_l(X)=p_l(Y)=p_l(Z)=\frac{\lambda_l}{3}.
```

测量 bit-flip noise：

```math
p_l(\text{no flip})=1-\lambda_l,\qquad
p_l(\text{flip})=\lambda_l.
```

对每个采样到的事件 `e_l`，定义 score：

```math
s_l(\tau)
=
\frac{\partial}{\partial \lambda_l}
\log p_l(e_l;\lambda_l).
```

典型 score 为：

```math
s_l(\tau)=
\begin{cases}
-\frac{1}{1-\lambda_l}, & e_l = I,\\
\frac{1}{\lambda_l}, & e_l \ne I.
\end{cases}
```

测量 bit-flip noise 同理：

```math
s_l(\tau)=
\begin{cases}
-\frac{1}{1-\lambda_l}, & \text{no flip},\\
\frac{1}{\lambda_l}, & \text{flip}.
\end{cases}
```

实际实现中对 `\lambda_l` 做数值裁剪，避免 `1 / \lambda_l` 或 `1 / (1-\lambda_l)` 发散。

## 3. 目标函数与噪声敏感度

纠错协议的目标函数定义为完整 trajectory loss 的期望：

```math
J(\lambda)
=
\mathbb E_{\tau\sim P_\lambda}[L(\tau)].
```

默认 loss 是 logical failure indicator：

```math
L(\tau)
=
\mathbf 1[\hat{\ell}(s(\tau)) \ne \ell(\tau)].
```

这里 `\hat{\ell}` 是 decoder 输出，`\ell` 是由前向 Pauli frame 计算出的真实 logical error。

由于 trajectory 概率可分解为局部噪声事件概率和 stabilizer 测量采样概率：

```math
P_\lambda(\tau)
=
\prod_l p_l(e_l;\lambda_l)\;P(m\mid e),
```

并且给定噪声事件后，stabilizer 测量采样本身不显式依赖 `\lambda_l`，因此可用 score-function estimator：

```math
\frac{\partial J}{\partial \lambda_l}
=
\mathbb E_\tau
\left[
L(\tau)
\frac{\partial}{\partial \lambda_l}
\log p_l(e_l;\lambda_l)
\right].
```

加入 baseline `b` 降低方差：

```math
S_l
=
\frac{\partial J}{\partial \lambda_l}
=
\mathbb E_\tau[(L(\tau)-b)s_l(\tau)].
```

在 batch 估计中，默认取：

```math
b=\frac{1}{N}\sum_{k=1}^N L(\tau_k).
```

因此有限样本估计为：

```math
\hat S_l
=
\frac{1}{N}
\sum_{k=1}^N
\left(L(\tau_k)-\bar L\right)
s_l(\tau_k).
```

其中：

```math
\bar L=\frac{1}{N}\sum_{k=1}^N L(\tau_k).
```

热点分数为：

```math
\hat H_l = |\hat S_l|.
```

## 4. 前向 stabilizer trajectory 算法

**Algorithm 1: Forward Noise-Aware Stabilizer Shot**

输入：

- stabilizer-compatible 纠错电路 `C`。
- 初始 stabilizer state `\mathcal S_0`。
- 初始 Pauli frame `F_0=I`。
- 局部噪声模型集合 `{p_l(e;\lambda_l)}`。
- detector construction `D`。
- decoder `Dec`。
- logical loss `L`。

输出：

- 一条 trajectory `\tau`。
- 每个噪声位置的 score `s_l(\tau)`。
- shot loss `L(\tau)`。

过程：

```text
1.  Initialize stabilizer tableau S <- S_0.
2.  Initialize Pauli frame F <- I.
3.  Initialize measurement record M <- empty.
4.  Initialize score record R <- empty.

5.  For each operation O_t in time order:

6.      If O_t is a Clifford gate:
7.          Forward-update S by the Clifford action.
8.          Forward-update F by the same Clifford action.

9.      If O_t is a noise location l:
10.         Sample event e_l ~ p_l(e; lambda_l).
11.         Apply e_l to S.
12.         Apply e_l to F.
13.         Record score R[l] += d log p_l(e_l; lambda_l) / d lambda_l.

14.     If O_t is a reset:
15.         Measure/reset the target qubit in S.
16.         Clear the corresponding component in F.

17.     If O_t is a measurement:
18.         Sample the stabilizer measurement result m.
19.         If measurement noise exists:
20.             Sample measurement flip event e_l.
21.             Flip m if required.
22.             Record its score in R[l].
23.         Append m to measurement record M.

24. Construct detector record s <- D(M).
25. Decode logical correction ell_hat <- Dec(s).
26. Compute true logical error ell from the final Pauli frame F.
27. Compute shot loss L(tau) = 1[ell_hat != ell].
28. Return tau, R, and L(tau).
```

这个算法保持纠错协议的前向时序：噪声先影响状态和后续 syndrome，syndrome 再影响 decoder，decoder 最后影响 logical failure 判断。

## 5. 噪声热点估计算法

**Algorithm 2: Score-Function Hotspot Estimation**

输入：

- 电路 `C`。
- shot 数 `N`。
- 噪声位置集合 `\mathcal L`。
- detector construction `D`。
- decoder `Dec`。
- logical loss `L`。

输出：

- logical failure rate `\hat J`。
- signed sensitivity `\hat S_l`。
- hotspot score `\hat H_l`。
- 聚合热点图。

过程：

```text
1.  For k = 1, ..., N:
2.      Run Algorithm 1 and obtain:
3.          loss L_k = L(tau_k),
4.          scores R_k[l] for all sampled noise locations.

5.  Compute empirical logical failure rate:
6.      J_hat = (1/N) sum_k L_k.

7.  Use baseline:
8.      b = J_hat.

9.  For each noise location l:
10.     S_hat[l] = (1/N) sum_k (L_k - b) R_k[l].
11.     H_hat[l] = abs(S_hat[l]).

12. Aggregate hotspots by metadata:
13.     H_qubit[q] = sum_{l: q in qubits(l)} H_hat[l].
14.     H_round[r] = sum_{l: round(l)=r} H_hat[l].
15.     H_gate[g]  = sum_{l: gate(l)=g} H_hat[l].
16.     H_op[o]    = sum_{l: operation(l)=o} H_hat[l].

17. Return J_hat, S_hat, H_hat, and aggregated hotspot maps.
```

## 6. 高性能 bit-packed batch sampler

高性能 batch sampler 不改变第 3 节的数学估计器。它仍然估计：

```math
\hat S_l
=
\frac{1}{N}
\sum_{k=1}^N
\left(L(\tau_k)-\bar L\right)
s_l(\tau_k).
```

区别只在执行方式：逐 shot 引擎为每条 trajectory 创建独立 tableau、Pauli frame 和 measurement record；batch 引擎共享 stabilizer generator 的 Pauli 支撑，把 generator sign、Pauli frame、measurement record 和 noise event 压进整数 bit mask 中，并一次性执行相同的电路操作。

对 `N` 个 shot，batch sampler 用一个整数的第 `k` 位表示第 `k` 条 trajectory：

```text
X_frame[q]: bit k = shot k has X component on qubit q.
Z_frame[q]: bit k = shot k has Z component on qubit q.
Sign[i]:    bit k = shot k has negative sign on stabilizer generator i.
M[key]:     bit k = shot k measured 1 for measurement key.
E[l]:       bit k = shot k sampled an error event at noise location l.
```

Clifford gate 对全部 shot 同时更新：

```text
H(q):       swap X_frame[q], Z_frame[q]
S(q):       Z_frame[q] ^= X_frame[q]
CX(c,t):    X_frame[t] ^= X_frame[c]
            Z_frame[c] ^= Z_frame[t]
CZ(a,b):    Z_frame[a] ^= X_frame[b]
            Z_frame[b] ^= X_frame[a]
SWAP(a,b):  swap X_frame[a], X_frame[b]
            swap Z_frame[a], Z_frame[b]
```

测量时，batch sampler 维护一个无噪声理想 tableau `\mathcal S_t^{ideal}`。如果被测 Pauli 在当前 stabilizer span 中，理想测量 mask 由已打包的 generator signs 决定；如果它与某些 generator 反对易，则采样一个 50/50 outcome mask，并用该 mask 更新被替换 generator 的 sign。随后用 Pauli frame 决定每个 shot 的翻转：

```math
m_k
=
m^{ideal}_k
\oplus
\langle F_k, P_{meas}\rangle
\oplus
f_k,
```

其中 `f_k` 是 measurement bit-flip noise。该 batch sampler 支持随机 Pauli measurement，只要所有 shot 共享同一个 Clifford/stabilizer 支撑演化；也就是说，不支持基于单个 shot 测量结果选择不同后续电路的 adaptive branching。

**Algorithm 3: Bit-Packed Batch Hotspot Estimation**

输入：

- 电路 `C`。
- shot 数 `N`。
- batch loss mask function `B`，返回 logical failure mask。
- 局部噪声位置集合 `\mathcal L`。

输出：

- logical failure rate `\hat J`。
- signed sensitivity `\hat S_l`。
- hotspot score `\hat H_l`。

过程：

```text
1.  Initialize ideal stabilizer tableau S_ideal <- S_0.
2.  Initialize bit-packed stabilizer signs and frames:
        Sign[i] <- 0 for all stabilizer generators
        X_frame[q] <- 0 for all q
        Z_frame[q] <- 0 for all q
3.  Initialize measurement masks M and event masks E.

4.  For each operation O_t in time order:

5.      If O_t is a Clifford gate:
6.          Update S_ideal once.
7.          Update all shot frames by bit operations.

8.      If O_t is noise location l:
9.          Sample event mask E[l].
10.         Apply the masked Pauli event to X_frame / Z_frame.

11.     If O_t is measurement of Pauli P:
12.         If P is deterministic, compute ideal mask from stabilizer span signs.
13.         Else sample a random ideal outcome mask and update S_ideal signs.
14.         Compute frame flip mask using symplectic product <F, P>.
15.         Apply measurement-noise flip mask if present.
16.         Store M[key].

17. Compute loss mask Loss <- B(M, X_frame, Z_frame).
18. J_hat <- popcount(Loss) / N.

19. For each noise location l:
20.     Use E[l] and Loss to count:
            error-and-loss shots,
            error-and-no-loss shots,
            no-error-and-loss shots,
            no-error-and-no-loss shots.
21.     Compute S_hat[l] from the same score-function formula.
22.     H_hat[l] <- abs(S_hat[l]).
```

这个 batch sampler 当前是快速路径，而不是通用 adaptive tableau 分支引擎。它适合 repetition code、surface-code syndrome extraction 这类所有 shot 共享同一 stabilizer 支撑演化、差异由 generator sign mask 和 Pauli frame mask 表示的 QEC 电路。

### Required native packed sampler

`npsim.native.compile_native_sampler` 是面向 Rust packed sampler 的稳定入口。
batch/DEM/hotspot 的运行时快速路径现在要求 `npsim._npsim_native` 可导入，并且
电路能被 native 编译：

```python
from npsim.native import compile_native_sampler

sampler = compile_native_sampler(circuit, backend="auto")
batch = sampler.sample(shots=100_000, seed=1)
```

普通采样吞吐基准使用 measurement-only fast path，避免为 hotspot 额外生成
`noise_event_masks` 和 final Pauli frame：

```python
measurement_masks = sampler.sample_measurements(shots=100_000, seed=1)
```

热点识别同样强制走 native 路径。`BatchForwardNoiseAwareSimulator.estimate()` 和
`DemBatchHotspotSimulator.estimate()` 会调用 Rust：采样 batch 保留在 Rust
packed words 中，Python loss/decoder 回调只读取按需暴露的 bit helper 或 mask
属性，最终 `loss_mask` 交回 Rust 计算 sensitivity、hotspot、metadata aggregation
和 top-k cache。没有 native 扩展或 native 不支持该电路时会抛
`UnsupportedNativeCircuitError`，不再回退到 Python reference。

`backend="auto"` 和 `backend="native"` 都要求 native 成功；保留 `"auto"` 只是为了
兼容旧调用。`backend="python"` 不再支持。向 `BatchForwardNoiseAwareSimulator.run_batch()`
或 `DemBatchHotspotSimulator.run_batch()` 传入 Python `rng` 时，运行时会用
`rng.getrandbits(64)` 派生 native seed；这只保持随机分布，不保证旧 Python 路径的
bit-for-bit 序列一致。

```python
sampler = compile_native_sampler(circuit, backend="native")
```

原生扩展源码位于 `native/`，使用 PyO3/maturin：

```bash
(cd native && ../.venv/bin/python -m maturin develop --release)
```

采样吞吐 benchmark 位于 `benchmarks/sampling_throughput.py`：

```bash
.venv/bin/python benchmarks/sampling_throughput.py --distances 15 21 31 --rounds 3
.venv/bin/python benchmarks/sampling_throughput.py --family random-clifford --qubits 128 256 512 --depth 20
.venv/bin/python benchmarks/sampling_throughput.py --family random-clifford --qubits 128 256 512 --depth 20 --noise-rate 0.001
.venv/bin/python benchmarks/dem_throughput.py --distances 9 13 21 --rounds 3
.venv/bin/python benchmarks/hotspot_throughput.py --distances 9 13 21 --rounds 3 --shots 100000
```

默认场景是 rotated surface-code memory；`--family random-clifford` 会生成固定种子的随机 Clifford layer circuit，最后测量所有 qubits。random Clifford benchmark 可用 `--noise-rate` 和 `--noise-model depolarizing1|x` 在每层后加入单比特噪声，并用 `--measurement-noise-rate` 加测量 bit-flip。`dem_throughput.py` 比较 native Rust 与 test-only Python reference 的 DEM generation 和默认 DEM hotspot estimate；若安装了 `stim`，也会报告 Stim DEM generation 和 detector bit-packed sampling 基线。若安装了 `stim`，sampling benchmark 会同时报告 Stim bit-packed sampler 吞吐和 NPSim/Stim 比值；未安装时只报告 NPSim 并标记 `stim-skip`。
`hotspot_throughput.py` 同时报告全链路 estimate 和预生成 batch 上的纯 hotspot
聚合阶段。全链路包含 test-only Python reference 的采样和 loss callback；纯聚合阶段仍包含
把完整 public result payload 转成 Python mapping 的兼容成本。

## 7. Stabilizer 更新规则

模拟器内部使用二进制 symplectic 表示。一个 `n` 比特 Pauli 写成：

```math
P(x,z)=i^\kappa X^x Z^z,
\qquad
x,z\in \mathbb F_2^n.
```

两个 Pauli 是否反对易由 symplectic product 判断：

```math
\langle (x,z),(x',z')\rangle
=
x\cdot z' + z\cdot x'
\pmod 2.
```

Clifford gate 对 `(x,z)` 的前向更新为：

```text
H(q):       x_q <-> z_q

S(q):       z_q <- z_q xor x_q

S†(q):      z_q <- z_q xor x_q
            phase/sign differs from S on tableau rows, but the x/z frame map is identical.

CX(c,t):    x_t <- x_t xor x_c
            z_c <- z_c xor z_t

CZ(a,b):    implemented as H(b), CX(a,b), H(b)
            equivalently:
            z_a <- z_a xor x_b
            z_b <- z_b xor x_a

SWAP(a,b):  implemented as CX(a,b), CX(b,a), CX(a,b)
            equivalently swaps x_a <-> x_b and z_a <-> z_b.
```

对 stabilizer tableau，以上规则作用到每个 stabilizer generator。对 Pauli frame，同样规则作用到已累计的物理错误 frame。

Pauli measurement 的规则：

- 若被测 Pauli 与所有 stabilizer generator 对易，则结果确定，由 stabilizer span 中的符号决定。
- 若它与某些 generator 反对易，则结果随机；选择一个反对易 generator 替换为被测 Pauli，并用它消去其他 generator 的反对易关系。

## 8. Detector Error Model 生成

Detector error model 是把局部物理错误事件映射成 detector flips 和 logical observable flips 的稀疏图模型。它不替代 score-function 热点估计，而是新增一个中间表示：

```math
(l,e)
\longmapsto
\left(
p_l(e),
\Delta D(l,e),
\Delta L(l,e)
\right).
```

其中 `l` 是噪声位置，`e` 是该位置的非 identity 错误事件，`\Delta D` 是被翻转的 detector 集合，`\Delta L` 是被翻转的 logical observable 集合。

Detector 被声明为若干 measurement key 的 parity：

```math
D_j(\tau)
=
\bigoplus_{r\in A_j} m_r.
```

Logical observable 也声明为 measurement parity 和/或最终 Pauli frame 上某个 Pauli observable 的翻转：

```math
L_a(\tau)
=
\left(\bigoplus_{r\in B_a}m_r\right)
\oplus
\langle F_\tau, P_a\rangle.
```

对每个单错误事件，生成 DEM edge：

```text
error(p_l(e)) D_i D_j ... L_a ...
```

**Algorithm 4: Single-Error DEM Construction**

输入：

- 电路 `C`。
- 结构化 detector 声明 `{D_j}`。
- 结构化 logical observable 声明 `{L_a}`。
- 局部噪声位置集合 `\mathcal L`。

输出：

- detector error model edges。

过程：

```text
1.  Run the circuit with all stochastic noise disabled.
2.  Record reference detector values D_ref and logical values L_ref.

3.  For each noise occurrence l:
4.      For each non-identity event e in the noise model at l:
5.          Run the circuit again with only event e injected at l.
6.          Record D_injected and L_injected.
7.          detector_flips <- {j | D_ref[j] xor D_injected[j] = 1}
8.          logical_flips  <- {a | L_ref[a] xor L_injected[a] = 1}
9.          p <- probability of event e under the noise model.
10.         If detector_flips or logical_flips is non-empty:
11.             Add DEM edge error(p) detector_flips logical_flips.
```

这个过程要求理想电路和单错误注入后的相关测量是确定的；如果测量本身会产生随机 tableau 分支，当前 DEM 生成器会报错并要求使用更通用的逐 shot 分析。

DEM 和热点可以通过噪声位置 id 连接。已有 location-level 热点：

```math
H_l = \left|\frac{\partial J}{\partial \lambda_l}\right|
```

可按 DEM edge 的事件概率投影：

```math
H_{(l,e)}
=
H_l
\frac{p_l(e)}{\sum_{e'}p_l(e')}.
```

这样可以把噪声敏感度从物理时空位置投影到 detector graph edge 上，用于分析哪些 syndrome graph 边对应的物理错误最影响 logical failure。

实现中提供两个投影入口：

```text
dem.project_sensitivities_to_detector_graph(sensitivities)
dem.project_result_to_detector_graph(simulation_result)
```

输出 `DetectorGraphHotspots`，包含：

```text
edge_hotspots              # 每条 DEM edge 的 signed sensitivity / hotspot
by_detector_edge           # 按 (detectors, observables) 聚合的 hotspot
signed_by_detector_edge    # 按 (detectors, observables) 聚合的 signed sensitivity
by_detector                # 按 detector node 聚合的 hotspot
signed_by_detector         # 按 detector node 聚合的 signed sensitivity
by_observable              # 按 logical observable 聚合的 hotspot
signed_by_observable       # 按 logical observable 聚合的 signed sensitivity
by_location                # 按原始噪声位置聚合的 hotspot
signed_by_location         # 按原始噪声位置聚合的 signed sensitivity
```

对一个 location `l`，若它产生多条 DEM edge，投影权重为：

```math
w_{l,e}
=
\frac{p_l(e)}{\sum_{e'}p_l(e')}.
```

于是 edge-level signed sensitivity 为：

```math
S_{l,e}^{edge}=w_{l,e}S_l,
\qquad
H_{l,e}^{edge}=|S_{l,e}^{edge}|.
```

按 detector graph edge 聚合时，key 是：

```text
(detector_tuple, observable_tuple)
```

例如：

```text
((3, 8), ())      # detector D3-D8 graph edge
((5,), (0,))      # boundary/logical edge involving D5 and L0
```

按 detector node 聚合时，一条包含多个 detector 的 edge 会把 hotspot 平均分给这些 detector，避免多 detector edge 在 node heatmap 中被重复计数。

`DETECTOR` 和 `OBSERVABLE_INCLUDE` 在电路中是一等 operation。逐 shot 模拟器执行到这些 operation 时会立即计算并记录：

```text
trajectory.detectors[id]
trajectory.observables[id]
trajectory.detector_record
```

batch sampler 也会生成对应 bit mask：

```text
batch.detectors[id]
batch.observables[id]
```

DEM 生成器既可以接受显式传入的 `Detector` / `LogicalObservable` 声明，也可以直接从 circuit 内的 detector / observable operations 自动读取声明。

### PyMatching batch decoder 接口

PyMatching 接口位于 DEM 和 loss 计算之间，不改变前向 trajectory 或 score-function
梯度估计。它的作用是把 detector record 解码成 predicted logical correction：

```math
\text{forward batch}
\longrightarrow
s^{(k)}
\longrightarrow
\hat{\ell}^{(k)}
\longrightarrow
L^{(k)}.
```

给定 DEM edge 集合 `E`，构造二元校验矩阵：

```math
H_{i,e}
=
\mathbf 1[D_i\in \Delta D_e],
```

以及 logical fault 矩阵：

```math
F_{a,e}
=
\mathbf 1[L_a\in \Delta L_e].
```

其中 `H` 的行是 detector，列是 DEM edge；`F` 的行是 logical observable，
列也是 DEM edge。每条 edge 的 matching 权重为：

```math
w_e
=
\log\frac{1-p_e}{p_e}.
```

**Algorithm 5: PyMatching Batch Decoding from DEM**

输入：

- graphlike detector error model。
- batch detector masks `B_i`，其中第 `k` 位是 shot `k` 的 detector `D_i`。
- shot 数 `N`。

输出：

- logical correction masks `C_a`。

过程：

```text
1.  Enumerate detector ids D_i and logical observable ids L_a.
2.  Build H[i,e] from the detector set of each DEM edge e.
3.  Build F[a,e] from the logical observable set of each DEM edge e.
4.  Set edge weight w_e <- log((1-p_e)/p_e).
5.  Construct PyMatching from H, F, and w.
6.  For each shot k and detector i:
7.      S[k,i] <- bit_k(B_i).
8.  Decode all rows of S with PyMatching.decode_batch.
9.  Pack predicted logical correction bits back into C_a masks.
```

随后 loss 可以继续保持 bit mask 形式。例如单 logical observable 时：

```text
failure_mask = batch.observables[0] xor correction_masks[0]
```

这个 decoder 接口要求 DEM 是 graphlike：每条 edge 最多连接两个 detector。没有
detector 的纯 logical edge 表示 syndrome 不可见的 logical fault，当前接口会拒绝它，
因为 matching decoder 无法从 detector record 中恢复这种错误。

### DEM mode hotspot 模拟

DEM mode 不执行 stabilizer 电路，而是直接在 detector error model 上采样。每条
DEM edge `e` 被视为一个独立 Bernoulli error instruction：

```math
f_e \sim \mathrm{Bernoulli}(p_e).
```

一次 DEM shot 的 detector record 和真实 logical observable flip 为：

```math
D_i
=
\bigoplus_{e: D_i\in \Delta D_e} f_e,
\qquad
L_a
=
\bigoplus_{e: L_a\in \Delta L_e} f_e.
```

decoder 只看 detector record：

```math
\hat L = \mathrm{Dec}(D),
```

默认 loss 是任一 logical observable correction 后仍翻转：

```math
L_{\mathrm{shot}}
=
\mathbf 1
\left[
\exists a:\; L_a\oplus \hat L_a = 1
\right].
```

DEM-level 目标函数为：

```math
J_{\mathrm{DEM}}(p)
=
\mathbb E_{f\sim \prod_e \mathrm{Bernoulli}(p_e)}
\left[L_{\mathrm{shot}}(f)\right].
```

对每条 DEM edge 的 hotspot 定义为：

```math
S_e
=
\frac{\partial J_{\mathrm{DEM}}}{\partial p_e}.
```

仍使用 score-function estimator：

```math
\hat S_e
=
\frac1N
\sum_{k=1}^N
\left(L_k-\bar L\right)
\left(
\frac{f_e^{(k)}}{p_e}
-
\frac{1-f_e^{(k)}}{1-p_e}
\right).
```

在 bit-packed 实现中，每条 edge 的 event mask `E_e` 与 loss mask `F` 只需要
`popcount(F & E_e)`、`popcount(E_e)` 和 `popcount(F)` 就能累计梯度。

如果需要回到物理 noise location `l`，并且 DEM edge 保留了 `location_id`，使用
线性化链式聚合：

```math
S_l^{\mathrm{DEM}}
=
\sum_{e:\mathrm{loc}(e)=l}
\frac{p_e}{\sum_{e':\mathrm{loc}(e')=l}p_{e'}}
S_e.
```

实现入口：

```text
result = DemBatchHotspotSimulator(dem).estimate(
    shots=100_000,
    seed=1,
    decoder=pymatching_decoder,
)
```

其中 `decoder` 可以是 `PyMatchingBatchDecoder`，也可以是任何提供
`decode_batch_masks(batch)` 的对象。输出同时包含：

```text
edge_sensitivities          # dJ_DEM / dp_e
edge_hotspots               # |dJ_DEM / dp_e|
sensitivities               # 按 location_id 聚合后的 signed sensitivity
hotspots                    # 按 location_id 聚合后的 hotspot
detector_graph_hotspots     # 按 detector edge / node / observable 聚合
```

这个模式非常适合做 detector graph / decoder-level 的高速热点扫描。它的限制是：
同一物理 noise location 产生的多个 DEM edge 在这里按独立 error instruction 采样；
这符合普通 DEM sampling 语义，但不完全等同于原始 forward trajectory 中的互斥
categorical Pauli event。高噪声率或强相关噪声下，应回到 forward mode 校验。

## 9. Repetition Code 热点示例

对于 bit-flip repetition code，data qubit 上的 `X` 错误会改变相邻 parity-check syndrome。一次 syndrome extraction 中，第 `i` 个 check 测量：

```math
Z_i Z_{i+1}.
```

decoder 根据 syndrome 估计 correction `\hat{\ell}`。真实 residual error 由最终 Pauli frame 得到：

```math
r_i = F_i^X \oplus \hat c_i.
```

logical failure loss 为：

```math
L(\tau)
=
\mathbf 1
\left[
\sum_i r_i > \left\lfloor\frac{d}{2}\right\rfloor
\right].
```

对每个 data-noise location 和 measurement-noise location 分别估计 `S_l`。如果某一轮 measurement error 或某个 data qubit error 被人为提高，其对应位置应在 `H_l` 排序中显著上升。

## 10. Stim 子集导入

为了和 Stim 工作流衔接，项目提供 `.stim` 文本子集导入器。导入器输出：

```text
StimImportResult(
    circuit,
    detectors,
    observables,
    measurement_keys,
)
```

其中：

- `circuit` 是本项目的前向 `Circuit`。
- `detectors` 是结构化 `Detector` 声明。
- `observables` 是结构化 `LogicalObservable` 声明。
- `measurement_keys` 是 Stim measurement record 到内部 key 的顺序映射。

导入器会把 `DETECTOR` 和 `OBSERVABLE_INCLUDE` 同时保留为 circuit operation，因此导入后的电路本身已经包含 detector / observable 语义；额外返回的 `detectors` / `observables` 主要用于显式检查或兼容旧接口。

支持的 Stim 指令子集：

```text
H, S, S_DAG, SQRT_Z_DAG
X, Y, Z
CX, CNOT, CZ, SWAP
R, RX, RY
M, MX, MY
MPP
X_ERROR, Y_ERROR, Z_ERROR
DEPOLARIZE1, DEPOLARIZE2
PAULI_CHANNEL_1, PAULI_CHANNEL_2
DETECTOR
OBSERVABLE_INCLUDE
TICK, QUBIT_COORDS, SHIFT_COORDS  # accepted as metadata/no-op subset
```

measurement record 引用支持 `rec[-k]`。导入器将其解析为内部 measurement key：

```text
M 0
M 1
DETECTOR rec[-1] rec[-2]
```

对应：

```text
Detector(measurement_keys=("m1", "m0"))
```

噪声指令映射为带唯一 id 的 `NoiseLocation`：

```text
X_ERROR(p) q          -> BernoulliPauliNoise("X")
DEPOLARIZE1(p) q      -> SingleQubitDepolarizing()
DEPOLARIZE2(p) a b    -> TwoQubitDepolarizing()
PAULI_CHANNEL_1(...)  -> PauliChannel(...)
M(p) q                -> MeasurementBitFlip() attached to measurement
```

导入后的电路可以直接用于 trajectory 模拟、batch sampler 或 DEM 生成：

```text
imported = parse_stim_circuit(stim_text)
dem = DetectorErrorModelGenerator(
    imported.circuit,
    detectors=imported.detectors,
    observables=imported.observables,
).generate()
```

当前不支持 `REPEAT` block、复杂 target modifier、坐标平移语义的完整累积、非整数 qubit target、复杂 feedback target、`CORRELATED_ERROR` / `ELSE_CORRELATED_ERROR` 等 Stim 高级语义。遇到这些语法会抛出 `StimImportError`，避免静默生成错误电路。

## 11. 验证标准

实现应满足以下校验：

- 无噪声或零 loss 时，`\hat J` 和热点分数应接近 0。
- 单比特 bit-flip toy model 中，若 `J(\lambda)=\lambda`，则 `\partial J/\partial\lambda=1`。
- 有限差分校验：

```math
\frac{J(\lambda_l+\epsilon)-J(\lambda_l-\epsilon)}{2\epsilon}
\approx
\hat S_l.
```

- 对称纠错电路中，几何等价的噪声位置应在统计误差内给出相近 hotspot score。
- 人为提高某个时空位置的噪声率后，该位置或相邻 detector 区域应在 top-k hotspot 中出现。

## 12. 当前算法边界

当前模型限制在 stabilizer-compatible stochastic noise：

- Clifford gate：`H`、`S`、`S†`、`CX`、`CZ`、`SWAP`。
- 理想 Pauli gate：`X`、`Y`、`Z` 以及任意 sparse Pauli string。
- Measurement：`X`、`Y`、`Z` basis 单比特测量，以及任意 Pauli-string measurement。
- Reset：`X`、`Y`、`Z` basis reset。
- Pauli noise：固定 Pauli 事件、通用 Pauli mixture、single-qubit depolarizing、two-qubit depolarizing。
- Classical noise：measurement bit-flip noise。
- Idle / reset / gate-local 错误：只要能表示为 stabilizer-compatible stochastic Pauli channel，就可以作为带 score 的噪声位置。
- 高性能 batch sampler：支持确定和随机 Pauli measurement 的 bit-packed 快速路径；遇到按 shot 测量结果选择不同后续电路的 adaptive branching 时，需要使用逐 shot 通用模拟器。
- Detector error model：支持结构化 detector / logical observable 声明，并通过单错误传播生成 Stim-like `error(p) D... L...` edge；当前不支持需要随机 tableau 分支的 DEM 构造。
- PyMatching batch decoder：可从 graphlike DEM 构造 matching decoder，并批量解码 detector record / bit-packed detector masks；需要可选依赖 `pymatching`、`numpy`、`scipy`，且不接受 hyperedge 或 syndrome 不可见的纯 logical edge。
- DEM hotspot mode：支持独立 DEM edge sampling、edge-level sensitivity、按 `location_id` 链式聚合的 physical-location hotspot，以及 detector-graph hotspot；不保留同一原始 noise location 下多个 Pauli event 的互斥采样语义。
- Stim import：支持常见 Clifford、reset、measurement、Pauli/depolarizing noise、`DETECTOR rec[-k]` 和 `OBSERVABLE_INCLUDE(k) rec[-k]` 子集；不支持 `REPEAT` 和 Stim 完整语义。

非 Clifford 门、非 Pauli 噪声、amplitude damping 等非 stabilizer-preserving channel 不直接进入初版算法；需要先做 Pauli twirling、离散化近似，或替换为可由 stabilizer trajectory 采样的等效噪声模型。
