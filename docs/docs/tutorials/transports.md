---
sidebar_position: 3
title: "Transports"
---

# Choosing a Transport

Philote can carry a compute call over two different gRPC transports. Both send
exactly the same variable data; they differ only in how it is framed on the
wire.

| Transport | RPC shape | Best for |
|---|---|---|
| Streaming | bidirectional stream of `VariableMessage` | large variables, many megabytes of data |
| Unary | one `VariableSet` each way | small disciplines called many times |

By default the client picks for you. Most users never need to read the rest of
this page.

## Why a second transport exists

gRPC's blocking unary path fuses everything a call needs — send the metadata,
send the message, half-close, receive the metadata, receive the message,
receive the status — into a **single** batch operation on the underlying core.

The streaming path cannot do that. It spawns a thread per call to drain the
request iterator, does fork-lock bookkeeping around every message, and issues a
separate batch in each direction for each message. That cost is fixed per call:
it does not shrink when the payload is small, and it does not hide behind
network latency.

For a discipline like `Paraboloid` — two scalars in, one scalar out — the
framing costs far more than the data. In an optimizer loop making tens of
thousands of evaluations, that overhead dominates the wall clock.

:::note
The saving is a fixed number of microseconds per call, not a percentage. It
matters in proportion to how *often* you call, not how *much* you send.
:::

## Measured difference

From `utils/bench_transport.py`, with the server in a separate process:

| Case | Streaming | Unary | Speedup |
|---|---|---|---|
| 2 variables, 1 element | 512 µs | 216 µs | 2.4x |
| 10 variables, 1 element | 1244 µs | 299 µs | 4.2x |
| 100 variables, 1 element | 9.9 ms | 1.6 ms | 6.2x |
| 2 variables, 1 element, 16 clients | 8.5 ms | 2.4 ms | 3.6x |
| 2 variables, 10000 elements | 2.4 ms | 0.49 ms | 4.9x |
| 2 variables, 100000 elements | 19.7 ms | 3.3 ms | 6.0x |

The advantage is a fixed per-call saving of a few hundred microseconds. It
matters in proportion to how *often* you call, not how much you send — which
is why it shows up as 6x on a 100-variable discipline and as a rounding error
on a single enormous array.

Past a few hundred kilobytes a stream does pull ahead, because splitting the
payload lets the server serialize one chunk while the client reads the
previous one. Measured, the crossover sits between 256 KiB and 512 KiB, which
is where `unary_max_bytes` is set.

:::note
These numbers assume the default chunk size. Streaming a large array with
`num_double` left at its default of 1000 fragments it into hundreds of
messages and is much slower than either alternative — 19.7 ms for a 1.6 MB
payload, against 2.6 ms for the same stream unchunked. If you stream large
arrays, raise `num_double`.
:::

## How the client chooses

`DisciplineClient.transport` accepts three values:

- `"auto"` (default) — decide per RPC, as described below
- `"unary"` — force the unary transport, skipping both size checks. An
  oversized payload then fails with `RESOURCE_EXHAUSTED` and is retried on the
  stream, rather than being quietly rerouted before it is sent.
- `"stream"` — always use the streaming transport

In `"auto"` mode the client applies three checks, in order.

**1. A size gate, evaluated once per RPC.** After `get_variable_definitions()`
the client knows the shape of every continuous variable, so the payload size is
fixed and can be decided once. If either the request or the response would
exceed `unary_max_bytes` (256 KiB by default), that RPC is marked
stream-only for the life of the client and the check is not repeated.

**2. A per-call guard.** Discrete variables carry arbitrary
`google.protobuf.Value` data whose size is not known until the call is made.
The client sums the assembled message sizes and falls back to streaming if the
total is too large. This falls back for **that call only** — a large discrete
value on one call says nothing about the next.

**3. A fallback on failure.** If the unary call fails with `UNIMPLEMENTED`, the
server predates this feature; the client retries on the stream and stops
attempting unary altogether. If it fails with `RESOURCE_EXHAUSTED`, the client
retries on the stream. Any other status is a genuine server error and
propagates.

```python
import philote_mdo.general as pmdo

client = pmdo.ExplicitClient(channel=channel)
client.get_discipline_info()      # learns whether the server supports unary
client.run_setup()
client.get_variable_definitions()

# the default: let the client decide
client.transport = "auto"

# or pin one, e.g. to reproduce a bug or benchmark
client.transport = "stream"
```

Calling `get_discipline_info()` is optional but recommended: it lets the client
learn the server's capability up front instead of discovering it by attempting
a call. The OpenMDAO bindings call it for you during setup.

## Server side

`ExplicitServer` and `ImplicitServer` implement both transports, and
`attach_to_server` registers both. There is nothing to configure.

A server can opt out by setting `_supports_unary = False`, and can advertise a
capacity ceiling with `max_unary_bytes`. When a server advertises a ceiling,
the effective limit is the smaller of that and the client's `unary_max_bytes`.

```python
class MyServer(pmdo.ExplicitServer):
    max_unary_bytes = 512 * 1024
```

## Raising the size limit

Raising `unary_max_bytes` is rarely worth it: past 256 KiB a chunked stream is
faster anyway, so there is little left to gain.

If you do raise it past gRPC's 4 MiB message ceiling, you must also size the
channel and the server, or the call will fail with `RESOURCE_EXHAUSTED`:

```python
import grpc
import philote_mdo.utils as utils

limit = 16 * 1024 * 1024

channel = grpc.insecure_channel("localhost:50051",
                                options=utils.channel_options(limit))
server = grpc.server(executor, options=utils.server_options(limit))

client = pmdo.ExplicitClient(channel=channel)
client.unary_max_bytes = limit
```

## Retry safety

:::warning
When a unary call fails with `RESOURCE_EXHAUSTED` while receiving the
*response*, the server has already run the discipline. The streaming retry runs
it a second time.

This is harmless for the pure-function contract Philote assumes, but a
discipline that carries state between calls — one whose `solve_residuals`
warm-starts from the previous solution, for instance — will see two
evaluations. Pin `transport = "stream"` to disable the retry entirely.
:::

## Backward compatibility

A new client talking to an old server works: the first unary attempt returns
`UNIMPLEMENTED` and the client falls back and remembers. The only cost is one
wasted round trip, and calling `get_discipline_info()` avoids even that.

An old client talking to a new server also works, since it simply never calls
the new RPCs.
