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
| 2 variables, 1 element | 493 µs | 220 µs | 2.2x |
| 10 variables, 1 element | 1319 µs | 416 µs | 3.2x |
| 100 variables, 1 element | 9714 µs | 1618 µs | 6.0x |
| 2 variables, 1 element, 16 clients | 8398 µs | 2354 µs | 3.6x |
| 2 variables, 100000 elements | 81 ms | 111 ms | **0.7x** |

The last row is the important one: **unary loses on large payloads.** A single
large message cannot be pipelined, so the receiver sits idle until the whole
thing arrives. The crossover sits a little above 128 KiB, which is why the
client stops using unary past that size.

## How the client chooses

`DisciplineClient.transport` accepts three values:

- `"auto"` (default) — decide per RPC, as described below
- `"unary"` — always use the unary transport, skipping the size check
- `"stream"` — always use the streaming transport

In `"auto"` mode the client applies three checks, in order.

**1. A size gate, evaluated once per RPC.** After `get_variable_definitions()`
the client knows the shape of every continuous variable, so the payload size is
fixed and can be decided once. If either the request or the response would
exceed `unary_max_bytes` (128 KiB by default), that RPC is marked
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

`unary_max_bytes` is a *performance* threshold, not a capacity limit. Raising
it past 128 KiB will make the client use unary for payloads where streaming is
faster.

If you do want to raise it past gRPC's 4 MiB message ceiling, you must also
size the channel and the server, or the call will fail with
`RESOURCE_EXHAUSTED`:

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
