# NPSim Documentation

NPSim is a Rust-core forward noise-aware stabilizer simulator with a Python
API. It supports packed batch sampling, detector error model generation,
detector-level sampling, decoder integration, and noise hotspot estimation for
stabilizer-compatible quantum error correction workflows.

## Start Here

- [User Guide](user_guide.md): installation, workflows, examples, results,
  troubleshooting, and best practices.
- [API Reference](api_reference.md): public Python and Rust API surfaces.
- [Theory](forward_noise_aware_stabilizer.md): score-function estimator and
  forward trajectory model.

## Quick Build

From a source checkout:

```bash
python -m venv .venv
.venv/bin/python -m pip install -U pip maturin
.venv/bin/python -m maturin develop --release
```

Optional integrations:

```bash
.venv/bin/python -m pip install numpy scipy pymatching pillow stim
```

## Documentation Development

Preview this documentation site locally:

```bash
.venv/bin/python -m pip install mkdocs-material
.venv/bin/python -m mkdocs serve
```
