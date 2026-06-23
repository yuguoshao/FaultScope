# NPSim

NPSim 是一个面向量子纠错工作流的前向噪声感知 stabilizer 模拟器。当前产品运行时由 Rust
core 提供，并通过 Python API 暴露；主要能力包括 bit-packed batch sampling、detector
error model 生成、DEM 层采样、decoder 集成和噪声热点估计。

文档站点见 [NPSim Documentation](https://yuguoshao.github.io/NPSim/)。
本地文档入口：

- [User Guide](docs/user_guide.md)：安装、示例、工作流和排错。
- [API Reference](docs/api_reference.md)：当前 Python/Rust API surface。
- [Theory](docs/theory.md)：理论原理、公式推导和实现中的计算细节。

## 项目结构

- `crates/npsim-core`：Python 无关的 Rust core，包含 circuit/DEM 数据模型、packed
  sampling 和 hotspot 聚合。
- `crates/npsim-python`：PyO3 binding crate，构建 `npsim._npsim_native`。
- `npsim/`：公共 Python import surface、decoder/Stim/visualization adapters 和示例构建器。
- `docs/`：MkDocs 文档站点。
- `tests/`、`benchmarks/`：回归测试、Stim 对照和吞吐基准。

## 安装与构建

从源码 checkout 直接安装：

```bash
python -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install .
```

可选依赖按需通过 extras 安装：

```bash
.venv/bin/python -m pip install ".[pymatching,visualization]"
.venv/bin/python -m pip install ".[test]"
```

`pip install .` 会按 `pyproject.toml` 自动获取 build dependency `maturin>=1.7,<2`，并构建
`npsim._npsim_native`。离线安装或使用 `--no-build-isolation` 时，需要提前准备好 maturin。

常用验证：

```bash
.venv/bin/python -c "import npsim; print(npsim.Circuit)"
cargo test --workspace
.venv/bin/python -m unittest discover -s tests -q
```

如果修改了 Rust extension 或 Python package 后需要刷新当前环境，重新安装即可：

```bash
.venv/bin/python -m pip install --force-reinstall .
```

## 最小示例

下面的例子构造一个单比特 X 噪声位置，采样测量结果，并估计该噪声率对 loss mask 的敏感度。

```python
from npsim import (
    BernoulliPauliNoise,
    BatchForwardNoiseAwareSimulator,
    Circuit,
    NoiseLocation,
    Operation,
)

x_noise = NoiseLocation(
    id="data_x0",
    model=BernoulliPauliNoise("X"),
    rate=0.02,
    qubits=(0,),
    tags={"round": 0, "gate": "idle", "qubit": 0},
)

circuit = Circuit(
    n_qubits=1,
    operations=(
        Operation.noise(x_noise),
        Operation.measure(0, key="m0", basis="Z"),
    ),
)

simulator = BatchForwardNoiseAwareSimulator(circuit)
batch = simulator.run_batch(shots=1024, seed=1)
result = simulator.estimate(
    shots=2048,
    seed=2,
    loss_mask_fn=lambda batch: batch.measurements["m0"],
    top_k=5,
)

print(batch.measurement_bit("m0", 0))
print(result.hotspot_table(top_k=5))
```

## 当前 API 要点

- `Operation.pauli_gate(...)` 是 Pauli gate 构造器；`Operation.pauli` 是只读属性。
- `PauliFrame` 和 `StabilizerState` 从 `npsim.core` 导入，不是顶层 `npsim` export。
- `BatchForwardNoiseAwareSimulator` 是前向 packed batch runtime 的主要入口。
- `DetectorErrorModelGenerator` 和 `generate_native_dem(...)` 生成 DEM；未显式传入 detector /
  observable 时，会读取 circuit 中的 `Operation.detector(...)` 和
  `Operation.observable_include(...)`。
- `DemBatchHotspotSimulator` 在 DEM 层采样，每条 DEM edge 按独立 Bernoulli instruction 处理。
- `edge_sensitivities` 和 `edge_hotspots` 是按 DEM edge index keyed 的 dict。
- `edges_by_location()` 返回 `dict[str, list[DetectorErrorEdge]]`。
- `materialize_dem=False` 的 native DEM sampler 是轻量采样路径，`sampler.dem is None`，
  需要完整 DEM metadata 的 estimate/hotspot API 会抛出 `ValueError`。

## 工作流选择

| 需求 | 推荐工作流 |
| --- | --- |
| 查看原始 measurement/noise masks | Forward sampling |
| 自定义 measurement-history loss | Forward estimate + `loss_mask_fn` |
| detector-level decoder | Forward 或 DEM estimate + decoder |
| graphlike matching decoder | DEM + `PyMatchingBatchDecoder` |
| DEM edge 级热点排序 | DEM hotspot estimate |
| 重复 detector-level sampling | 生成 DEM 后复用 DEM sampler |
| Rust 集成 | `npsim-core` |

NPSim 当前产品路径是 packed batch engine，不暴露通用的 per-shot adaptive branching simulator。

## 文档开发

本地预览文档站：

```bash
.venv/bin/python -m pip install mkdocs-material
.venv/bin/python -m mkdocs serve
```

严格构建：

```bash
.venv/bin/mkdocs build --strict --site-dir /private/tmp/npsim-doc-review-site
```

## Benchmarks

构建 extension 后可从仓库根目录运行：

```bash
.venv/bin/python benchmarks/sampling_throughput.py --distances 15 21 31 --rounds 3
.venv/bin/python benchmarks/sampling_throughput.py --family random-clifford --qubits 128 256 512 --depth 20
.venv/bin/python benchmarks/dem_throughput.py --distances 9 13 21 --rounds 3
.venv/bin/python benchmarks/hotspot_throughput.py --distances 9 13 21 --rounds 3 --shots 100000
.venv/bin/python benchmarks/surface_code_threshold.py --distances 3 5 7 --shots 10000
```

Stim/PyMatching 相关 benchmark 会在对应可选依赖安装后启用对照。
