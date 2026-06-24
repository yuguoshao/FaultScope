# npsim-fusion-blossom

Official NPSim fusion-blossom backend scaffold.

This package proves the post-install backend discovery path. It does not link
or run the real fusion-blossom solver yet. Its decoder class currently delegates
to NPSim's in-tree `NativeGraphlikeDetectorCopyDecoder`, so returned decoder
handles are native and can enter the NPSim fast path, but they are only smoke
test backends.

Install locally during development:

```bash
.venv/bin/python -m pip install -e backends/npsim-fusion-blossom
```
