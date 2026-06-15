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
- repetition-code reference decoder and experiment builder

非 Pauli、非 Clifford 噪声不直接进入 stabilizer simulator；初版应先做
Pauli twirling 或替换为 stabilizer-compatible stochastic channel。
