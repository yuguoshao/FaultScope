# faultscope-collection

`faultscope-collection` provides native logical error-rate collection and the
global fixed/adaptive batch scheduler used by FaultScope.

Persistence, CSV formatting, threshold analysis, and CLI orchestration remain
in the Python package and are intentionally outside the sampling hot path.
