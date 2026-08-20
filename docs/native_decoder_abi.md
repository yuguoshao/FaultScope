# Native decoder ABI V4

FaultScope 0.2.10 accepts exactly one native decoder plugin ABI:

```text
entry-point group: faultscope.native_decoders
manifest and capsule name: faultscope.native_decoder_plugin.v4
numeric ABI version: 4
capsule method: __faultscope_native_decoder_capsule__
```

V4 is a breaking ABI. FaultScope does not reinterpret V1, V2, or V3
descriptors and does not export compatibility aliases for their factory or
worker types. An old capsule name or numeric version is rejected with an
`expected ABI v4` error before a worker is created.

## Design

The factory owns immutable decoder construction data. It is thread-safe and
creates private mutable workers. A worker exposes one tagged `decode_batch`
callback; there are no per-layout callbacks or `supports_*` flags.

Detector syndrome layout and FaultScope attribution are separate concerns.
The decoder receives only a detector batch. When hotspot attribution is
requested, FaultScope records an internal edge-major `DemAttributionTrace`
sidecar from the same sampling pass. That trace is never placed in the decoder
view and is not part of this ABI.

## Format tags and fixed correction mapping

```c
#define FAULTSCOPE_NATIVE_DECODER_FORMAT_MASKS  1u
#define FAULTSCOPE_NATIVE_DECODER_FORMAT_PACKED 2u
#define FAULTSCOPE_NATIVE_DECODER_FORMAT_EVENTS 3u
```

The factory returns a stable, non-empty, duplicate-free ordered list of these
tags. The order is the decoder's preference order. Unknown tags are errors.

The output layout is fixed by the input tag:

| Detector input | Required correction output |
| --- | --- |
| `MASKS` | detector-major mask words -> observable-major mask words |
| `PACKED` | shot-major packed detector rows -> shot-major packed observable rows |
| `EVENTS` | sparse per-shot detector indices -> shot-major packed observable rows |

`EVENTS` is a GF(2) representation. Repeating a detector index within one shot
toggles that detector again; an even number of occurrences cancels. Converting
Masks or Packed to Events emits only the final non-zero detector indices.

FaultScope chooses the first decoder preference that a producer can emit
directly. If no direct format intersects, it converts to the decoder's first
preference. The DEM sampler can directly emit all three formats, so it always
uses the decoder's first preference. A `Problem` constructs the decoder; it
does not select the runtime batch layout.

## C layout

The existing V1-named leaf views remain reusable layout primitives. V4 adds a
U32 slice and tagged unions:

```c
typedef struct {
    const uint32_t *ptr;
    size_t len;
} FaultScopeNativeDecoderU32SliceV1;

typedef union {
    FaultScopeNativeDetectorMaskBatchViewV1 masks;
    FaultScopeNativePackedDetectorShotBatchViewV1 packed;
    FaultScopeNativeDetectorEventShotBatchViewV1 events;
} FaultScopeNativeDetectorBatchPayloadV4;

typedef struct {
    uint32_t format;
    uint32_t reserved;
    FaultScopeNativeDetectorBatchPayloadV4 payload;
} FaultScopeNativeDetectorBatchViewV4;

typedef union {
    FaultScopeNativeCorrectionMaskBatchMutViewV1 masks;
    FaultScopeNativePackedObservableShotBatchMutViewV1 packed;
} FaultScopeNativeCorrectionBatchPayloadV4;

typedef struct {
    uint32_t format;
    uint32_t reserved;
    FaultScopeNativeCorrectionBatchPayloadV4 payload;
} FaultScopeNativeCorrectionBatchMutViewV4;
```

The factory and worker descriptors are:

```c
typedef struct FaultScopeNativeDecoderWorkerV4 FaultScopeNativeDecoderWorkerV4;

typedef struct {
    uint32_t abi_version;                  /* offset 0  */
    size_t struct_size;                    /* offset 8  */
    uint64_t flags;                        /* offset 16 */
    void *factory_state;                   /* offset 24 */
    void (*drop_factory_state)(void *);    /* offset 32 */
    FaultScopeNativeDecoderStatusV1 (*name)(
        const void *, FaultScopeNativeDecoderStringViewV1 *
    );                                     /* offset 40 */
    FaultScopeNativeDecoderStatusV1 (*detector_ids)(
        const void *, FaultScopeNativeDecoderI64SliceV1 *
    );                                     /* offset 48 */
    FaultScopeNativeDecoderStatusV1 (*observable_ids)(
        const void *, FaultScopeNativeDecoderI64SliceV1 *
    );                                     /* offset 56 */
    FaultScopeNativeDecoderStatusV1 (*batch_formats)(
        const void *, FaultScopeNativeDecoderU32SliceV1 *
    );                                     /* offset 64 */
    FaultScopeNativeDecoderStatusV1 (*create_worker)(
        const void *, FaultScopeNativeDecoderWorkerV4 *, size_t
    );                                     /* offset 72 */
} FaultScopeNativeDecoderFactoryV4;

struct FaultScopeNativeDecoderWorkerV4 {
    size_t struct_size;                    /* offset 0  */
    void *worker_state;                    /* offset 8  */
    void (*drop_worker_state)(void *);     /* offset 16 */
    FaultScopeNativeDecoderStatusV1 (*decode_batch)(
        void *,
        const FaultScopeNativeDetectorBatchViewV4 *,
        FaultScopeNativeCorrectionBatchMutViewV4 *
    );                                     /* offset 24 */
};
```

On supported 64-bit targets the frozen layouts are:

| Type | Size | Alignment |
| --- | ---: | ---: |
| `FaultScopeNativeDecoderU32SliceV1` | 16 | 8 |
| `FaultScopeNativeDetectorBatchViewV4` | 64 | 8 |
| `FaultScopeNativeCorrectionBatchMutViewV4` | 48 | 8 |
| `FaultScopeNativeDecoderFactoryV4` | 80 | 8 |
| `FaultScopeNativeDecoderWorkerV4` | 32 | 8 |

The detector tagged fields are at offsets `0`, `4`, and `8`; correction tagged
fields use the same offsets. Plugins should use compile-time size and offset
assertions. `struct_size` is a descriptor-size check, not permission to expose
a shortened V4 descriptor: a V4 factory must be at least 80 bytes and a worker
must report exactly 32 bytes.

## Callback contract

All factory and worker callbacks are required. `factory_state`, `worker_state`,
the corresponding drop callbacks, and the thread-safe factory flag must be
non-null/present. Factory metadata must remain valid and unchanged for the
factory lifetime. In particular, `batch_formats` must return the same pointer,
length, values, and order before and after worker creation.

For every decode call:

- both `reserved` fields must be zero on entry and return;
- input and output tags must match the fixed mapping above;
- detector and observable IDs use the exact declared order;
- `shots`, counts, row widths, word widths, offsets, and byte lengths must be
  exact and must not overflow `size_t`;
- unused bits in the final mask word or packed row byte must be zero;
- Events offsets have length `shots + 1`, start at zero, are nondecreasing, and
  end at `event_count`; every index is less than `detector_count`;
- FaultScope initializes every correction data word or byte to zero before the
  callback; a plugin may leave zero corrections untouched or XOR corrections
  into the provided buffers;
- the callback may write only correction data words/bytes;
- it must not rewrite either tagged header, a payload pointer or dimension, or
  a nested mask pointer/word count.

FaultScope validates the input, ID order, output variant, observable layout,
shots, padding, and unchanged metadata around the callback. A violation fails
the batch and the worker is still destroyed normally.

## Ownership and lifetime

The plugin owns its factory and worker states. FaultScope owns all input and
output batch storage. Detector views and correction buffers are borrowed only
for the duration of `decode_batch`; a plugin must not retain them.

The Python capsule owns the factory descriptor and `factory_state`. It must
have a non-null capsule destructor. When the last capsule owner is released,
that destructor calls the descriptor's `drop_factory_state` exactly once and
then releases the descriptor itself. FaultScope retains the capsule while any
bound factory or worker can use that state; it does not separately call
`drop_factory_state`. A capsule without a destructor is rejected before the
descriptor is used.

`create_worker` receives an output descriptor and its capacity. On insufficient
capacity, a null output pointer, or a null factory state it must fail without
modifying the output bytes. On success it returns a complete worker. If it
fails after publishing a non-null partial `worker_state`, it must also publish
a valid drop callback so FaultScope can clean it exactly once.

Factories can be shared across threads. Workers are exclusive mutable objects:
FaultScope caches one per task/thread where appropriate and calls one worker
serially. A plugin must not put mutable solver state in the shared factory.

## Errors

Status code zero means success. A non-zero status must provide either an empty
message or a valid UTF-8-compatible byte view. The message storage must remain
readable after the callback returns and must not be overwritten by the next
callback on another worker/thread before FaultScope has consumed it. Static
strings or factory/worker-owned synchronized storage are suitable.

Callbacks must not unwind across the C boundary. Rust plugins should catch
panics and return an error status. C/C++ plugins should translate exceptions to
status errors.

## Manifest and official packages

The Python entry-point group and public decoder class names are unchanged. A
manifest must report:

```python
{
    "abi_version": "faultscope.native_decoder_plugin.v4",
    "name": "pymatching",
    "version": "...",
    "source": "faultscope-pymatching",
    "decoders": {"pymatching": NativePyMatchingDecoder},
}
```

The official PyMatching and fusion-blossom packages declare `[PACKED]` and
require `faultscope>=0.2.10,<0.3`. The built-in detector-copy decoder declares
`[MASKS]`; no-correction declares `[MASKS, PACKED, EVENTS]`. `bpdecoder` is
temporarily non-installable while awaiting V4 migration. `mwpm` remains
non-installable and has not migrated from its old ABI.

## V3 to V4 migration

1. Change the capsule/manifest ABI string to
   `faultscope.native_decoder_plugin.v4` and numeric version to `4`.
2. Replace the factory and worker descriptors with the V4 layouts. Do not keep
   V3 aliases in production code.
3. Add the stable `batch_formats` factory callback.
4. Delete capability flags and layout-specific decode callbacks. Implement one
   tagged `decode_batch` callback.
5. Return mask-major corrections only for Masks input and packed-row
   corrections for Packed or Events input.
6. Treat repeated Events indices as XOR and enforce zero padding/reserved
   fields and immutable metadata.
7. Return a capsule with a destructor that owns the descriptor/factory state
   and calls `drop_factory_state` exactly once.
8. Rebuild against FaultScope 0.2.10+, update the dependency bound, and run the
   V4 fixture/layout/rejection tests.

Graphlike Problem ABI remains V1 and is independent of this decoder ABI change.
