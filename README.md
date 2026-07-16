# FaultScope

FaultScope 是一个面向量子纠错工作流的噪声感知 fault attribution 工具包。它使用 detector error
model formalism 描述 noisy Clifford circuits：detectors 是 measurement outcomes 上的 parity
constraints，detector matrix \(D\) 汇总这些 constraints，measurement syndrome matrix
\(\Omega\) 描述每个 circuit error 会翻转哪些 measurements，detector error matrix
\(H=D\Omega\) 描述每个 error 会违反哪些 detectors。当前产品运行时由 Rust core 提供，并通过
Python API 暴露；主要能力包括 bit-packed stabilizer batch sampling、detector error model
生成、DEM 层采样、decoder 集成和噪声热点估计。

文档站点见 [FaultScope Documentation](https://yuguoshao.github.io/FaultScope/)。
本地文档入口：

- [User Guide](docs/user_guide.md)：安装、示例、工作流和排错。
- [API Reference](docs/api_reference.md)：当前 Python/Rust API surface。
- [Release And Compatibility](docs/release.md)：支持矩阵、版本策略和发布流程。
- [Theory](docs/theory.md)：理论原理、公式推导和实现中的计算细节。

## 项目结构

- `crates/faultscope-core`：Python 无关的 Rust core，包含 circuit/DEM 数据模型、detector
  syndrome sampling 和 hotspot 聚合。
- `crates/faultscope-collection`：logical error-rate / threshold collection 的 Rust 调度和计数层。
- `crates/faultscope-python`：PyO3 binding crate，构建 `faultscope._native`。
- `faultscope/`：公共 Python import surface、decoder/Stim/visualization adapters 和示例构建器。
- `docs/`：MkDocs 文档站点。
- `tests/`、`benchmarks/`：回归测试、Stim 对照和吞吐基准。

## 安装与构建

FaultScope `0.2.x` 支持 CPython 3.10–3.14。预构建 wheels 的目标平台是
manylinux x86_64/aarch64、macOS 11+ x86_64/arm64 和 Windows x86_64；源码构建需要
Rust 1.85 或更新版本。

当前 workspace 版本为 `0.2.2`。项目仍处于 pre-1.0，补丁版本也可能收口或重命名 Python/Rust
API；native decoder ABI 则使用独立版本号管理。

从源码 checkout 直接安装：

```bash
python -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install .
```

可选依赖按需通过 extras 安装：

```bash
.venv/bin/python -m pip install ".[pymatching,visualization]"
.venv/bin/python -m pip install ".[collection]"  # plotting/fitting helpers
.venv/bin/python -m pip install ".[test]"
```

`pip install .` 会按 `pyproject.toml` 自动获取 build dependency `maturin>=1.7,<2`，并构建
`faultscope._native`。离线安装或使用 `--no-build-isolation` 时，需要提前准备好 maturin。

常用验证：

```bash
.venv/bin/python -c "import faultscope; print(faultscope.Circuit)"
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
from faultscope import (
    BernoulliPauliNoise,
    FaultScopeSimulator,
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

simulator = FaultScopeSimulator(circuit)
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
- `PauliFrame` 和 `StabilizerState` 从 `faultscope.core` 导入，不是顶层 `faultscope` export。
- `PauliFrame(x, z)` 在构造边界校验并持有 binary x/z 不变量；`StabilizerState` 只能通过
  `StabilizerState.zero(n_qubits)` 构造。无效 dense/sparse Pauli 输入会在计算、RNG 调用和
  state mutation 前抛出 `ValueError`。
- `FaultScopeSimulator` 是前向 packed batch runtime 的主要入口。
- `DemFaultScopeSimulator` 是同形的 DEM runtime 入口：从 circuit 直接生成 DEM sampler，
  再按 DEM edge 概率采样 detector syndrome / logical observable flips。它不会逐门执行
  forward trajectory，也不会返回 measurement 或 Pauli-frame masks。
- `DetectorErrorModelGenerator` 和 `generate_native_dem(...)` 生成 detector error model；未显式
  传入 detector / observable 时，会读取 circuit 中的 `Operation.detector(...)` 和
  `Operation.observable_include(...)`。每个 generated edge 对应 detector error matrix
  \(H\) 的一列及其 logical observable flips。
- `DemHotspotEstimator` 在 DEM 层采样，每条 DEM edge 按独立 Bernoulli instruction 处理，并把
  sampled edge vector 映射成 detector syndrome 和 logical observable flip record。
- `faultscope.collection.Collector` 提供可复用的 threshold-style logical error-rate
  collection 配置；`collect(...)` 是一次性薄封装。Rust 负责 native sampler/decoder 调度和
  计数，Python 负责任务解析、strong id 和 CSV resume。
- `NativeNoCorrectionDecoder` 和后续 native decoder handle 可通过
  `estimate(..., decoder=decoder)` 自动走 native fast path；传入 Python loss/correction
  callback 时回退到兼容路径。普通 Python decoder 或 subclass 不会自动获得 native hot path；
  可用 native backend 通过 `faultscope.decoders.available_native_decoders()` 查看。
- `DetectorErrorModel.compile_indexed()`、`compile_graphlike_problem()` 和
  `compile_binary_linear_problem()` 提供面向后续 fusion-blossom、BP+OSD 等 decoder 的 native
  problem views；这些 views 暴露 detector error matrix \(H\) 和 logical fault matrix 的稀疏结构。
- 可选 native decoder backend 通过统一后装命令管理，例如
  `python -m faultscope.backends status` 查看 catalog/status，
  `python -m faultscope.backends install pymatching --dry-run` 查看安装步骤；FaultScope 不会在
  `import` 或 `estimate(...)` 时隐式联网、clone 或编译。
- `mwpm` 仍可在 catalog/status 中发现，但外部 `faultscope-mwpm` 包还是 ABI v1，尚未迁移到
  FaultScope native decoder ABI v2，因此目前不可安装；`bposd` 是 catalog/status 中不可安装、
  未实现的预留项。通用 `install bposd --dry-run` 子命令会报告它不可用，但不会生成安装计划或
  步骤，也不会执行安装。
- 开发中的 PyMatching 和 fusion-blossom backend 可在激活 venv 后通过
  `.venv/bin/python -m pip install -e backends/faultscope-pymatching --no-build-isolation` 和
  `.venv/bin/python -m pip install -e backends/faultscope-fusion-blossom` 本地安装；当前是最小
  native MWPM backend；PyMatching backend 避免 FaultScope batch/correction masks 经 Python 转换。
- `edge_sensitivities` 和 `edge_hotspots` 是按 DEM edge index keyed 的 dict。
- `edges_by_location()` 返回 `dict[str, list[DetectorErrorEdge]]`。
- `materialize_dem=False` 的 native DEM sampler 是轻量采样路径，`sampler.dem is None`，
  需要完整 DEM metadata 的 estimate/hotspot API 会抛出 `ValueError`。
  `DemFaultScopeSimulator(circuit, materialize_dem=False)` 暴露同样的轻量路径。

## 工作流选择

| 需求 | 推荐工作流 |
| --- | --- |
| 查看原始 measurement/noise masks | Forward sampling |
| 自定义 measurement-history loss | Forward estimate + `loss_mask_fn` |
| detector-syndrome decoder | Forward 或 DEM estimate + decoder |
| graphlike matching decoder | 原型用 `PyMatchingDecoder`；高性能路径安装 `faultscope-pymatching` 后使用 `NativePyMatchingDecoder`，或使用 `faultscope-fusion-blossom`；`mwpm` 等待 ABI v2 迁移 |
| circuit 入口的 DEM 采样 | `DemFaultScopeSimulator(circuit)` |
| DEM edge 级热点排序 | `DemFaultScopeSimulator` 或 `DemHotspotEstimator(dem)` |
| 重复 detector syndrome sampling | `DemFaultScopeSimulator(circuit)` 或生成 DEM 后复用 DEM sampler |
| Rust 集成 | `faultscope-core` |

FaultScope 当前产品路径是 packed batch engine，不暴露通用的 per-shot adaptive branching simulator。

## 文档开发

本地预览文档站：

```bash
.venv/bin/python -m pip install mkdocs-material
.venv/bin/python -m mkdocs serve
```

严格构建：

```bash
.venv/bin/mkdocs build --strict --site-dir /private/tmp/faultscope-doc-review-site
```

## Benchmarks

性能 benchmark 应使用 release 构建的 extension：

```bash
.venv/bin/maturin develop --release --skip-install
```

然后从仓库根目录运行：

```bash
.venv/bin/python benchmarks/compiler_throughput.py --distances 5 10 15 20 --representation m-plus-r --json-out compiler-throughput.json
.venv/bin/python benchmarks/sampling_throughput.py --distances 15 21 31 --rounds 3
.venv/bin/python benchmarks/sampling_throughput.py --family random-clifford --qubits 128 256 512 --depth 20
.venv/bin/python benchmarks/dem_throughput.py --distances 9 13 21 --rounds 3
.venv/bin/python benchmarks/hotspot_throughput.py --distances 9 13 21 --rounds 3 --shots 100000
.venv/bin/python benchmarks/collection_throughput.py --shots 10000 --batch-size 1000 --adaptive-start-batch-size 100 --adaptive-max-batch-size 1000 --max-batch-seconds 0.25 --workers 1 2 4
.venv/bin/python benchmarks/native_decoder_fast_path.py
.venv/bin/python benchmarks/surface_code_decoder_performance.py --distances 3 5 7 --shots 10000
.venv/bin/python benchmarks/surface_code_threshold.py --distances 3 5 7 --shots 10000
```

Stim/PyMatching 相关 benchmark 会在对应可选依赖安装后启用对照；
surface-code decoder performance benchmark 会在安装 `faultscope-pymatching` 或
`faultscope-fusion-blossom` 后额外输出对应 native path。传入
`--split-native-baseline` 可把 native no-correction packed-row baseline 与
decoder 增量分开显示。需要比较不同 native decoder 的 mean-loss 时，使用
`--same-seed-across-paths` 让同一个 distance/rate 点复用相同 seed。
