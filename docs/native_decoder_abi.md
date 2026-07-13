# Native decoder ABI

FaultScope native decoder plugins use ABI version 1 and the capsule name
`faultscope.native_decoder_plugin.v1`. The descriptor is an append-only C layout:
consumers must use `struct_size` before reading an optional tail field. Appending
`create_worker_state` did not change `abi_version`.

## Collection worker state factory

The optional tail callback has this contract:

```text
create_worker_state(factory_state, out_state)
- returns a new independent mutable state
- uses the descriptor's existing decode and drop_state callbacks
- must preserve detector/observable metadata
- transfers every non-null out_state to the caller, regardless of status
- may be absent in descriptors ending before this field
```

`factory_state` is the descriptor's prototype state. It remains owned by the
plugin capsule and is borrowed for the duration of the call. The factory may read
immutable graph, problem, or option data from it. Because a descriptor declaring
the thread-safe flag permits concurrent callback use, the factory must also be
safe to call concurrently with other operations on the prototype.

On success, the callback writes a non-null pointer to a newly allocated state.
That pointer must not alias the prototype or any state returned by an earlier
factory call. Each returned state owns its mutable solver workspace, so decoder
callbacks for distinct worker states can run concurrently without sharing a
decoder-owned solver pool. Calling the descriptor's name, detector-id, and
observable-id callbacks with the worker pointer must return the same metadata as
the prototype.

Every non-null pointer written to `out_state` transfers ownership to FaultScope,
regardless of whether the callback returns success or failure. On success the
pointer becomes the worker wrapper's state and is passed to the descriptor's
decode callbacks. On failure FaultScope immediately calls the descriptor's
`drop_state` callback exactly once before returning the error. This defensive
rule prevents leaks from a producer that reports an error after allocating its
state; the producer must not retain or free a non-null pointer after returning.

Producers SHOULD initialize `out_state` to null before work begins and leave it
null on every failure path. Success still requires a non-null state. FaultScope
keeps the capsule alive while any worker wrapper exists and calls `drop_state`
exactly once when that wrapper is dropped. A descriptor that exports
`create_worker_state` must also export `drop_state`.

The descriptor itself and its prototype state remain capsule-owned. Dropping a
worker does not drop either one. Dropping the capsule eventually destroys the
descriptor and invokes `drop_state` once for the prototype state.

## Dynamic error-view lifetime

A dynamic `FaultScopeNativeDecoderStringViewV1` stored in
`FaultScopeNativeDecoderStatusV1::message` remains valid until the associated
decoder state is dropped. This includes views from overlapping callbacks on the
same state: producing a later error must not invalidate a view that another
caller may still be consuming.

Bundled backends satisfy this lifetime by retaining each dynamic error message
in state-owned append-only storage. Consequently, repeated errors retain their
message memory until state drop. Consumers should copy an error view promptly
and must not use its pointer after the state has been dropped.

## Legacy descriptors

A version-1 descriptor whose `struct_size` ends before
`create_worker_state` remains valid for direct decode. Consumers must not form a
full current-version descriptor reference or read the appended callback in this
case. FaultScope reads only fields whose complete byte range is covered by
`struct_size`. Requesting a collection worker instance from such a descriptor
returns:

```text
<decoder name> does not support collection worker instances
```

The same unsupported result applies when the field is present but the callback is
null.

## Bundled backends

The bundled pymatching and fusion-blossom plugins retain immutable construction
data in the prototype. Each factory call rebuilds one independent native solver
state. A descriptor state contains one mutex-protected mutable solver workspace;
parallelism across collection workers comes from separate descriptors rather than
from a decoder-owned state pool. Dynamic error messages use the state-owned
lifetime described above so concurrent callback consumers cannot observe an
invalidated error view.
