# Native decoder ABI v3

FaultScope native decoder plugins use one strict runtime contract:

```text
numeric ABI: 3
manifest and capsule name: faultscope.native_decoder_plugin.v3
entry-point group: faultscope.native_decoders
capsule method: __faultscope_native_decoder_capsule__
thread-safe factory flag: NATIVE_DECODER_FACTORY_FLAG_THREAD_SAFE = 1 << 0
```

The entry-point group and capsule method names are unchanged from ABI v2, but
there is no runtime compatibility with ABI v2. A v2 manifest, capsule, or
numeric ABI is rejected with the observed value and the expected v3 value.

## Pure factory/worker model

The capsule contains a `FaultScopeNativeDecoderFactoryV3`. The factory owns
immutable construction data and metadata, and a factory cannot decode. Its
only hot-path role is to create an exclusive worker. Every decode callback is
on `FaultScopeNativeDecoderWorkerV3`, whose mutable state is private to that
worker.

On 64-bit targets the exact C layouts are:

```c
typedef struct FaultScopeNativeDecoderFactoryV3 {
    uint32_t abi_version;                 /* offset 0 */
    size_t struct_size;                   /* offset 8 */
    uint64_t flags;                       /* offset 16 */
    void *factory_state;                  /* offset 24 */
    void (*drop_factory_state)(void *);   /* offset 32 */
    FaultScopeNativeDecoderStatusV1 (*name)(
        const void *, FaultScopeNativeDecoderStringViewV1 *
    );                                        /* offset 40 */
    FaultScopeNativeDecoderStatusV1 (*detector_ids)(
        const void *, FaultScopeNativeDecoderI64SliceV1 *
    );                                        /* offset 48 */
    FaultScopeNativeDecoderStatusV1 (*observable_ids)(
        const void *, FaultScopeNativeDecoderI64SliceV1 *
    );                                        /* offset 56 */
    FaultScopeNativeDecoderStatusV1 (*create_worker)(
        const void *factory_state,
        FaultScopeNativeDecoderWorkerV3 *out_worker,
        size_t out_worker_capacity
    );                                        /* offset 64 */
} FaultScopeNativeDecoderFactoryV3;            /* 72 bytes, align 8 */

typedef struct FaultScopeNativeDecoderWorkerV3 {
    size_t struct_size;                     /* offset 0 */
    void *worker_state;                     /* offset 8 */
    void (*drop_worker_state)(void *);      /* offset 16 */
    FaultScopeNativeDecoderStatusV1 (*decode_batch)(
        void *,
        const FaultScopeNativeDetectorMaskBatchViewV1 *,
        FaultScopeNativeCorrectionMaskBatchMutViewV1 *
    );                                          /* offset 24 */
    FaultScopeNativeDecoderStatusV1 (*decode_packed_batch)(
        void *,
        const FaultScopeNativePackedDetectorShotBatchViewV1 *,
        FaultScopeNativePackedObservableShotBatchMutViewV1 *
    );                                          /* offset 32 */
    FaultScopeNativeDecoderStatusV1 (*decode_detector_event_batch)(
        void *,
        const FaultScopeNativeDetectorEventShotBatchViewV1 *,
        FaultScopeNativePackedObservableShotBatchMutViewV1 *
    );                                          /* offset 40 */
} FaultScopeNativeDecoderWorkerV3;             /* 48 bytes, align 8 */
```

The stable v1-named view structs used inside v3 retain their 64-bit sizes:
`FaultScopeNativeDecoderStringViewV1` is 16 bytes,
`FaultScopeNativeDetectorMaskBatchViewV1` is 40 bytes,
`FaultScopeNativeCorrectionMaskBatchMutViewV1` is 40 bytes,
`FaultScopeNativePackedDetectorShotBatchViewV1` is 40 bytes,
`FaultScopeNativeDetectorEventShotBatchViewV1` is 56 bytes, and
`FaultScopeNativeDecoderStatusV1` is 24 bytes. Their names identify frozen
layouts; they do not provide compatibility with an older decoder ABI.

`create_worker` receives a host-provided capacity for `out_worker`. A producer
must not write beyond that capacity and must set `out_worker.struct_size` to
the bytes it actually initializes. FaultScope validates the required prefix
and reads optional callback fields only when covered by both the producer's
`struct_size` and the host-provided capacity.

## Canonical observable layout

ABI v3 has one observable-output layout. When a native decoder is bound to a
sampler, the factory's complete `observable_ids` sequence must equal the
sampler's canonical observable ids. Missing ids, extra ids, and reordered ids
are rejected before sampling and before `create_worker` or a decode callback is
called. Detector ids remain decoder-defined and may use a different order, but
each detector id must occur exactly once in a decoder's input layout. FaultScope
rejects duplicate detector ids when binding an ABI factory and again when
preparing a sampling plan.

The host allocates every correction output using that canonical sequence and
passes the same ids to all three decode callbacks. A callback must not replace
the output pointers, dimensions, or ids:

- `decode_batch` writes one correction mask for every canonical observable;
- `decode_packed_batch` and `decode_detector_event_batch` use bit `k` for
  canonical observable `k`; and
- unused high bits in the final packed byte of every shot must remain zero.

Output buffers are zero-initialized, so a zero correction does not require an
explicit write. FaultScope validates the full layout and packed padding after
each callback. It never realigns ids or treats a missing native correction as
an implicit zero correction. The no-decoder path remains a separate explicit
all-zero correction case.

## Threading and worker selection

`NATIVE_DECODER_FACTORY_FLAG_THREAD_SAFE` promises that the factory metadata
callbacks and `create_worker` may be called concurrently. It does not make a
worker shareable: each worker remains exclusively owned by its caller.

FaultScope creates exclusive workers for all native decode consumers:

- every collection task, including a collection configured with one worker;
- each estimate path;
- each debug/helper decode path; and
- every child of a composite decoder.

Collection keeps a thread-local worker cache keyed by task so repeated batches
reuse that task's worker on the same collection thread. Backends therefore do
not need a decoder pool or mutex around mutable solver state. They may share
immutable graph or matrix data from the factory.

## Ownership, errors, and FFI safety

The capsule owns the factory descriptor and `factory_state`. FaultScope keeps
the capsule alive while the factory handle or any worker exists. It calls
`drop_factory_state` exactly once when the last owner releases the factory.

A successful `create_worker` transfers one non-null `worker_state` and its
callbacks to FaultScope. FaultScope calls `drop_worker_state` exactly once. If
creation fails after returning a sufficiently complete partial worker,
FaultScope calls its declared `drop_worker_state` before reporting the error;
the producer must not retain or free transferred state after returning.

Output buffers belong to the host. Decode callbacks may write only within the
provided dimensions and capacity and must not ask FaultScope to free
backend-allocated output memory.

An error status carries a borrowed UTF-8 error string. It must remain valid
until the associated factory or worker state is dropped, including across
overlapping calls on that state. FaultScope copies error text promptly; a
consumer must not use the pointer after the owning state is dropped.

Every extern callback must catch backend failures. Rust callbacks must not unwind or panic across FFI, and C/C++ callbacks must not let exceptions cross
the boundary. Report failures through `FaultScopeNativeDecoderStatusV1`.

## Backend availability

`pymatching`, `fusion-blossom`, and `bpdecoder` are ABI v3 packages. `mwpm` is
discoverable but unavailable: its package is ABI v1 and
not yet migrated to FaultScope native decoder ABI v3. `bposd` is a reserved,
unimplemented, non-installable catalog/status entry. The generic install
subcommand accepts these unavailable catalog names but only reports why they are
unavailable; it returns no install plan or steps and installs nothing.

Public Python decoder class names do not change. The public Python decoder classes are factory handles; workers are private implementation objects and
are never returned through the Python API.

## Migrating a third-party ABI v2 backend

This section is migration guidance, not a compatibility promise.

1. Change the manifest and capsule names to
   `faultscope.native_decoder_plugin.v3` and numeric ABI to `3`.
2. Rebuild the descriptor as `FaultScopeNativeDecoderFactoryV3` and workers as
   `FaultScopeNativeDecoderWorkerV3`; there are no V2 aliases.
3. Return the sampler's complete canonical observable-id sequence and write all
   correction outputs in exactly that order with zero packed padding.
4. Keep `create_worker` host-capacity and partial-failure cleanup rules.
5. Set `NATIVE_DECODER_FACTORY_FLAG_THREAD_SAFE` only when metadata and worker
   creation are safe concurrently.
6. Keep `faultscope.native_decoders` and
   `__faultscope_native_decoder_capsule__()` unchanged.
7. Test distinct worker state, all three canonical output forms, layout and
   padding rejection, concurrent creation, drops, error-string lifetime, and
   panic/exception containment.

FaultScope deliberately does not adapt an ABI v2 descriptor at runtime. The
backend must be rebuilt and republished for ABI v3.
