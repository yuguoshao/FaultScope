# Native decoder ABI v2

FaultScope native decoder plugins use one strict runtime contract:

```text
numeric ABI: 2
manifest and capsule name: faultscope.native_decoder_plugin.v2
entry-point group: faultscope.native_decoders
capsule method: __faultscope_native_decoder_capsule__
thread-safe factory flag: NATIVE_DECODER_FACTORY_FLAG_THREAD_SAFE = 1 << 0
```

The entry-point group and capsule method names are unchanged from ABI v1, but
there is no runtime compatibility with ABI v1. A v1 manifest, capsule, or
numeric ABI is rejected with the observed value and the expected v2 value.

## Pure factory/worker model

The capsule contains a `FaultScopeNativeDecoderFactoryV2`. The factory owns
immutable construction data and metadata, and a factory cannot decode. Its
only hot-path role is to create an exclusive worker. Every decode callback is
on `FaultScopeNativeDecoderWorkerV2`, whose mutable state is private to that
worker.

On 64-bit targets the exact C layouts are:

```c
typedef struct FaultScopeNativeDecoderFactoryV2 {
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
        FaultScopeNativeDecoderWorkerV2 *out_worker,
        size_t out_worker_capacity
    );                                        /* offset 64 */
} FaultScopeNativeDecoderFactoryV2;            /* 72 bytes, align 8 */

typedef struct FaultScopeNativeDecoderWorkerV2 {
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
} FaultScopeNativeDecoderWorkerV2;             /* 48 bytes, align 8 */
```

The stable v1-named view structs used inside v2 retain their 64-bit sizes:
`FaultScopeNativeDecoderStringViewV1` is 16 bytes,
`FaultScopeNativeDetectorMaskBatchViewV1` is 40 bytes,
`FaultScopeNativeCorrectionMaskBatchMutViewV1` is 40 bytes,
`FaultScopeNativePackedDetectorShotBatchViewV1` is 40 bytes,
`FaultScopeNativeDetectorEventShotBatchViewV1` is 56 bytes, and
`FaultScopeNativeDecoderStatusV1` is 24 bytes. Their names identify frozen
layouts; they do not provide ABI v1 decoder compatibility.

`create_worker` receives a host-provided capacity for `out_worker`. A producer
must not write beyond that capacity and must set `out_worker.struct_size` to
the bytes it actually initializes. FaultScope validates the required prefix
and reads optional callback fields only when covered by both the producer's
`struct_size` and the host-provided capacity.

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

`pymatching`, `fusion-blossom`, and `bpdecoder` are ABI v2 packages. `mwpm` is
discoverable but unavailable: its package is ABI v1 and
not yet migrated to FaultScope native decoder ABI v2. `bposd` is a reserved,
unimplemented, non-installable catalog/status entry. The generic install
subcommand accepts these unavailable catalog names but only reports why they are
unavailable; it returns no install plan or steps and installs nothing.

Public Python decoder class names do not change. The public Python decoder classes are factory handles; workers are private implementation objects and
are never returned through the Python API.

## Migrating a third-party ABI v1 backend

This section is migration guidance, not a compatibility promise.

1. Change the manifest and capsule names to
   `faultscope.native_decoder_plugin.v2` and numeric ABI to `2`.
2. Split immutable construction data into `FaultScopeNativeDecoderFactoryV2`
   and mutable decode state into `FaultScopeNativeDecoderWorkerV2`.
3. Move all decode callbacks to the worker and implement `create_worker` with
   host capacity and partial-failure cleanup rules.
4. Set `NATIVE_DECODER_FACTORY_FLAG_THREAD_SAFE` only when metadata and worker
   creation are safe concurrently.
5. Keep `faultscope.native_decoders` and
   `__faultscope_native_decoder_capsule__()` unchanged.
6. Test distinct worker state, one-worker collection, concurrent creation,
   drops, error-string lifetime, and panic/exception containment.

FaultScope deliberately does not adapt an ABI v1 descriptor at runtime. The
backend must be rebuilt and republished for ABI v2.
