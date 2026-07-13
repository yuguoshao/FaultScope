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

Ownership transfers to FaultScope only after a successful status and non-null
output. FaultScope keeps the capsule alive while any worker wrapper exists, uses
the descriptor's existing decode callbacks with the worker pointer, and calls the
descriptor's `drop_state` callback exactly once when that worker wrapper is
dropped. A plugin must set `out_state` to null before work begins and leave it null
on failure. A descriptor that exports `create_worker_state` must also export
`drop_state`.

The descriptor itself and its prototype state remain capsule-owned. Dropping a
worker does not drop either one. Dropping the capsule eventually destroys the
descriptor and invokes `drop_state` once for the prototype state.

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
from a decoder-owned state pool.
