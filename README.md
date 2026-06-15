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

## 6. Stabilizer 更新规则

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

## 7. Repetition Code 热点示例

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

## 8. 验证标准

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

## 9. 当前算法边界

当前模型限制在 stabilizer-compatible stochastic noise：

- Clifford gate：`H`、`S`、`S†`、`CX`、`CZ`、`SWAP`。
- 理想 Pauli gate：`X`、`Y`、`Z` 以及任意 sparse Pauli string。
- Measurement：`X`、`Y`、`Z` basis 单比特测量，以及任意 Pauli-string measurement。
- Reset：`X`、`Y`、`Z` basis reset。
- Pauli noise：固定 Pauli 事件、通用 Pauli mixture、single-qubit depolarizing、two-qubit depolarizing。
- Classical noise：measurement bit-flip noise。
- Idle / reset / gate-local 错误：只要能表示为 stabilizer-compatible stochastic Pauli channel，就可以作为带 score 的噪声位置。

非 Clifford 门、非 Pauli 噪声、amplitude damping 等非 stabilizer-preserving channel 不直接进入初版算法；需要先做 Pauli twirling、离散化近似，或替换为可由 stabilizer trajectory 采样的等效噪声模型。
