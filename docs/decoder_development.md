# Decoder Development

This page is for developers who want to prototype a decoder in Python or add a
native decoder backend to NPSim. For user-facing workflows, see the
[User Guide](user_guide.md). For exact signatures, see the
[API Reference](api_reference.md).

## Mental Model

Decoder integration has two phases:

```text
Circuit -> DetectorErrorModel -> decoding problem -> decoder object
```

and then, on the hot path:

```text
packed detector syndrome masks -> decoder -> packed observable correction masks
```

A decoder can and usually should use circuit-derived information, but only
during construction. The decoder object should compile that information into
its own graph, matrices, weights, or lookup tables. Batch decoding should not
read the original circuit or DEM again.

The runtime expects detector and observable ids to be stable:

- `detector_ids` define the syndrome order consumed by the decoder.
- `observable_ids` define the logical correction ids that the decoder may
  return.
- A correction mask bit `k` is the decoder's predicted logical correction for
  shot `k`.

The default residual logical loss is:

```text
sampled observable mask XOR decoder correction mask
```

over all declared logical observables.

## Python Prototype Decoders

Python decoders are the quickest way to validate an algorithm. They use the
compatibility path, where NPSim materializes a Python batch object and calls
`decode_batch_masks(batch)`.

Recommended shape:

```python
from npsim.runtime import generate_native_dem


class MyDecoder:
    @classmethod
    def from_circuit(cls, circuit, detectors=None, observables=None):
        dem = generate_native_dem(
            circuit,
            detectors=detectors,
            observables=observables,
        )
        return cls.from_dem(dem)

    @classmethod
    def from_dem(cls, dem):
        # Use graphlike problems for MWPM/fusion-blossom-style decoders.
        problem = dem.compile_graphlike_problem()

        # Use this instead for BP+OSD/LDPC-style decoders:
        # problem = dem.compile_binary_linear_problem()

        return cls(
            detector_ids=problem.detector_ids,
            observable_ids=problem.observable_ids,
            problem=problem,
        )

    def __init__(self, detector_ids, observable_ids, problem):
        self.detector_ids = tuple(detector_ids)
        self.observable_ids = tuple(observable_ids)
        self.problem = problem

        # Compile the problem into the decoder's internal representation here.
        # For example: matching graph, sparse parity-check matrix, weights, etc.

    def decode_batch_masks(self, batch):
        syndrome_masks = [
            batch.detectors.get(detector_id, 0)
            for detector_id in self.detector_ids
        ]

        # Replace this with the actual algorithm.
        # Return one packed correction mask per corrected observable id.
        correction_for_first_observable = 0

        return {
            self.observable_ids[0]: correction_for_first_observable,
        }
```

Use the decoder like any other Python decoder:

```python
decoder = MyDecoder.from_circuit(
    circuit,
    detectors=detectors,
    observables=observables,
)

result = sampler.estimate(
    shots=10_000,
    decoder=decoder,
)
```

`batch.detectors` is a mapping from detector id to a Python integer containing
packed syndrome bits. The least significant bit is shot 0. The return value
must be `dict[int, int]`, mapping observable id to a packed correction mask.

Python decoders are ideal for correctness prototypes and small experiments.
They are not the final high-performance path, because detector masks and
correction masks cross the Python boundary.

## Getting Circuit And DEM Information

When starting from a circuit, first generate a detector error model:

```python
from npsim.runtime import generate_native_dem

dem = generate_native_dem(
    circuit,
    detectors=detectors,
    observables=observables,
)
```

The DEM can then be compiled into decoder-ready views:

```python
indexed = dem.compile_indexed()
graphlike = dem.compile_graphlike_problem()
binary = dem.compile_binary_linear_problem()
```

Use the view that matches the decoder family:

| View | Intended backend |
| --- | --- |
| `IndexedDem` | General DEM indexing, edge metadata, and stable order inspection |
| `GraphlikeDecodingProblem` | MWPM-style decoders, including future fusion-blossom adapters |
| `BinaryLinearDecodingProblem` | BP+OSD/LDPC-style decoders using sparse binary `H` and `F` |

`GraphlikeDecodingProblem` rejects hyperedges and undetectable pure logical
edges, because matching-style backends cannot infer those errors from syndrome
data. `BinaryLinearDecodingProblem` does not require graphlike edges; each DEM
edge is an independent binary error variable.

The problem views expose ids, counts, `edge_summary`, probabilities, weights,
and sparse binary matrix entries for construction-time inspection. They
intentionally do not expose `to_numpy_*` hot-path helpers. A native decoder
should consume the native problem representation instead of moving batch data
through Python.

## Native Decoder Backends

Native decoders are Python-owned handles around Rust decoder objects. They are
the intended path for production backends because the batch syndrome masks,
correction masks, default residual loss, and hotspot aggregation stay in native
memory.

The recommended ownership model is:

```text
Python passes circuit / DEM / options
        -> PyO3 factory reads those Python objects
        -> Rust compiles a native decoder backend
        -> Python receives an opaque decoder handle
        -> estimate(..., decoder=decoder) uses the native fast path
```

Python is responsible for selecting and configuring the decoder. Rust is
responsible for constructing the backend and running batch decode. After the
decoder is constructed, Python should not participate in the syndrome or
correction-mask hot path.

### Python Input, Rust Construction

A native decoder should expose constructors that accept Python objects:

```python
decoder = MyNativeDecoder.from_dem(
    dem,
    option_a=...,
    option_b=...,
)

decoder = MyNativeDecoder.from_circuit(
    circuit,
    detectors=detectors,
    observables=observables,
    option_a=...,
)
```

`from_circuit(...)` is a convenience constructor. It should generate or compile
a DEM first, then delegate to the same Rust construction path as
`from_dem(...)`.

The PyO3 implementation should convert Python input into Rust core structures
once:

```rust
#[pyclass(name = "MyNativeDecoder", module = "npsim._npsim_native")]
pub struct PyMyNativeDecoder {
    inner: Arc<dyn npsim_core::NativeBatchDecoder>,
}

#[pymethods]
impl PyMyNativeDecoder {
    #[staticmethod]
    pub fn from_dem(
        py: Python<'_>,
        dem: &Bound<'_, PyDetectorErrorModel>,
        option_a: Option<f64>,
    ) -> PyResult<Self> {
        let core_dem = dem.borrow().to_core_dem(py)?;

        let problem = core_dem
            .compile_graphlike_problem()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        let backend = MyNativeDecoder::from_graphlike_problem(
            problem,
            option_a,
        )
        .map_err(|err| PyValueError::new_err(err.to_string()))?;

        Ok(Self {
            inner: Arc::new(backend),
        })
    }
}
```

For BP+OSD/LDPC-style decoders, use
`core_dem.compile_binary_linear_problem()` instead. For a decoder that needs
more DEM metadata, use `compile_indexed()` or extend the construction-time DEM
view. Do not add per-batch circuit or DEM reads to `decode_batch(...)`.

The Rust backend stores all construction-time information:

```rust
pub struct MyNativeDecoder {
    detector_ids: Vec<i64>,
    observable_ids: Vec<i64>,
    // Compiled graph / H matrix / F matrix / weights / LLR / backend state.
}
```

The Python object only owns the handle. It does not own batch detector masks or
correction masks during `estimate(...)`.

The core Rust contract is `NativeBatchDecoder`:

```rust
use npsim_core::{
    CorrectionMaskBatch,
    DetectorMaskBatchView,
    Mask,
    NativeBatchDecoder,
    NpResult,
};

pub struct MyNativeDecoder {
    detector_ids: Vec<i64>,
    observable_ids: Vec<i64>,
    // Store the compiled graph, matrix, weights, or backend object here.
}

impl NativeBatchDecoder for MyNativeDecoder {
    fn name(&self) -> &'static str {
        "my-decoder"
    }

    fn detector_ids(&self) -> &[i64] {
        &self.detector_ids
    }

    fn observable_ids(&self) -> &[i64] {
        &self.observable_ids
    }

    fn decode_batch(
        &self,
        detectors: DetectorMaskBatchView<'_>,
    ) -> NpResult<CorrectionMaskBatch> {
        let words = npsim_core::word_count(detectors.shots);

        // detectors.masks are ordered exactly like self.detector_ids().
        // Run the backend and fill one Mask per corrected observable id.
        let correction_for_observable_0 = Mask::zero(words);

        CorrectionMaskBatch::new(
            vec![self.observable_ids[0]],
            vec![correction_for_observable_0],
            detectors.shots,
        )
    }
}
```

The PyO3 layer should expose a Python handle with constructors such as:

```python
decoder = MyNativeDecoder.from_dem(dem)
# or
decoder = MyNativeDecoder.from_circuit(
    circuit,
    detectors=detectors,
    observables=observables,
)
```

Internally, those constructors should compile the DEM/problem once and store an
`Arc<dyn NativeBatchDecoder + Send + Sync>`. The runtime recognizes native
handles explicitly; a Python subclass that only implements
`decode_batch_masks(batch)` remains a Python decoder and does not enter the
native fast path.

Native backends should be added as in-tree optional backends first. The current
feature slots are:

```text
decoder-fusion-blossom
decoder-bposd
```

They are Cargo/maturin build features, disabled by default. Python extras do
not currently enable third-party native decoder builds.

Python can inspect compiled native backend names:

```python
from npsim.decoders import available_native_decoders

print(available_native_decoders())
```

The default build currently exposes only the no-correction smoke-test backend.

## Validation And Performance Rules

NPSim validates native decoder output before using it:

- correction `shots` must match the sampled batch;
- correction mask word count must match the shot count;
- correction observable ids must be unique;
- correction observable ids must be declared by the decoder;
- missing observable corrections are treated as all-zero correction masks.

The native fast path is used only when:

- `decoder` is a native decoder handle;
- no `loss_mask_fn` is supplied;
- no `correction_mask_fn` is supplied.

If a Python loss or correction callback is supplied, NPSim uses the
compatibility path and may call `decoder.decode_batch_masks(batch)`. This is
useful for debugging and comparison, but it moves batch data through Python.

`materialize_dem=False` creates a light DEM sampler without full DEM metadata.
That mode can sample, but APIs that need DEM metadata, including hotspot result
construction and decoder problem construction, require a materialized DEM.

## Development Checklist

For a new decoder family:

1. Prototype with a Python class implementing `from_dem(...)` and
   `decode_batch_masks(batch)`.
2. Choose the construction view: graphlike for MWPM/fusion-blossom-style
   decoders, binary linear for BP+OSD/LDPC-style decoders.
3. Add an in-tree Rust backend implementing `NativeBatchDecoder`.
4. Expose a PyO3 handle with `from_dem(...)` and optionally
   `from_circuit(...)`.
5. Register the backend name in `available_native_decoders()` when the backend
   feature is enabled.
6. Test both paths: Python compatibility behavior and native no-callback fast
   path.
