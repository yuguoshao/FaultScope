# NPSim Documentation

NPSim is a Rust-core forward noise-aware stabilizer simulator with a Python
API. It supports packed batch sampling, detector error model generation,
detector-level sampling, decoder integration, and noise hotspot estimation for
stabilizer-compatible quantum error correction workflows.

## Start Here

- [User Guide](user_guide.md): installation, runnable workflows, examples,
  packed-mask basics, troubleshooting, and best practices.
- [Decoder Development](decoder_development.md): how to prototype Python
  decoders and add native decoder backends without moving hot-path data through
  Python.
- [API Reference](api_reference.md): current public Python and Rust API
  surfaces, callback contracts, and result object fields.
- [Theory](theory.md): theory, derivations, packed estimator formulas, DEM
  equations, and implementation-level calculation details.
- [Implementation Overview](implementation_overview.md): current packed
  runtime behavior, supported workflows, and implementation boundaries.

## Quick Build

From a source checkout:

```bash
python -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install .
```

Optional integrations:

```bash
.venv/bin/python -m pip install ".[pymatching,visualization]"
.venv/bin/python -m pip install ".[test]"
```

## Documentation Development

Preview this documentation site locally:

```bash
.venv/bin/python -m pip install mkdocs-material
.venv/bin/python -m mkdocs serve
```

Strict build:

```bash
.venv/bin/mkdocs build --strict --site-dir /private/tmp/npsim-doc-review-site
```
