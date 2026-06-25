# npsim-fusion-blossom

Official NPSim fusion-blossom backend scaffold.

This package proves the post-install backend discovery path. It does not link
or run the real fusion-blossom solver yet. Its decoder class constructs an
external Rust/PyO3 graphlike smoke decoder and exposes it through the NPSim
native decoder PyCapsule ABI. Returned decoder handles can enter the NPSim fast
path, but they are only smoke test backends.

Install locally during development:

```bash
.venv/bin/python -m pip install -e backends/npsim-fusion-blossom
```
